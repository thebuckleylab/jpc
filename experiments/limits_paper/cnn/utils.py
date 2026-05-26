import os
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax.flatten_util import ravel_pytree

import torch
import equinox as eqx

from experiments.datasets import get_dataloaders, get_tinyimagenet_loaders
from experiments.limits_paper.cnn.model import ScaledConv2d


def flatten_grads_per_layer_cnn(model, grads):
    """Flatten gradients per parameter layer for CNN. Returns list of flattened arrays.
    grads from update_pc_params is (model_grads, skip_grads); we use model_grads only.
    """
    if isinstance(grads, tuple):
        grads = grads[0]
    result = []
    for i in range(len(model.layers)):
        flat, _ = ravel_pytree(eqx.filter(grads.layers[i], eqx.is_array))
        if flat.size > 0:
            result.append(np.array(flat))
    return result


def get_tracked_cnn_param_positions_and_names(model):
    """Select a subset of parameter layers to track for grads/cos-sims.

    We track:
      - The last readout layer.
      - The *first conv layer* (top-level `ScaledConv2d`) in each of the
        3 stages of the architecture (one per stage).

    Returns:
      param_layer_indices: list mapping param-layer position -> model.layers index.
      tracked_param_positions: positions (indices into param_layer_indices) of
        the layers we keep.
      tracked_layer_names: human-readable names for those tracked layers.
    """
    layers = model.layers

    # param_layer_indices maps param-layer position -> model.layers index.
    param_layer_indices = []
    tracked_param_positions = []
    tracked_layer_names = []

    # Stage index: 0,1,2 for the three conv stages. We increment after each AvgPool2d.
    stage = 0
    conv_tracked_for_stage = {0: False, 1: False, 2: False}

    for i, layer in enumerate(layers):
        # Stage boundaries: AvgPool2d layers separate the 3 stages.
        if layer.__class__.__name__ == "AvgPool2d":
            stage = min(stage + 1, 3)
            continue

        flat, _ = ravel_pytree(eqx.filter(layer, eqx.is_array))
        has_params = flat.size > 0
        if not has_params:
            continue

        # This layer contributes a param-layer position.
        param_layer_indices.append(i)
        pos = len(param_layer_indices) - 1

        # Track the first conv in each stage (if it's a ScaledConv2d).
        is_scaled_conv2d = layer.__class__.__name__ == "ScaledConv2d"
        if is_scaled_conv2d and stage < 3 and not conv_tracked_for_stage[stage]:
            conv_tracked_for_stage[stage] = True
            tracked_param_positions.append(pos)
            tracked_layer_names.append(f"stage{stage + 1}_conv")

        # Always track the final readout layer (last module).
        if i == len(layers) - 1:
            tracked_param_positions.append(pos)
            tracked_layer_names.append("readout")

    return param_layer_indices, tracked_param_positions, tracked_layer_names


def compute_cosine_similarity(a, b):
    """Compute cosine similarity between two 1D numpy arrays."""
    if a.ndim > 1:
        a = a.reshape(-1)
    if b.ndim > 1:
        b = b.reshape(-1)
    min_dim = min(len(a), len(b))
    if min_dim == 0:
        return 0.0
    a = a[:min_dim]
    b = b[:min_dim]
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na * nb > 1e-10:
        return dot / (na * nb)
    return 0.0


def _tree_dot(a, b):
    """Sum of element-wise products over two pytrees with same structure."""
    leaves_a = jtu.tree_leaves(a)
    leaves_b = jtu.tree_leaves(b)
    return sum(jnp.sum(la * lb) for la, lb in zip(leaves_a, leaves_b))


def _tree_norm_sq(a):
    """Squared L2 norm of a pytree (sum of squares of all elements)."""
    return _tree_dot(a, a)


def _tree_normalize(a):
    """In-place-like: return a scaled so that tree_norm(a) == 1."""
    n = jnp.sqrt(_tree_norm_sq(a))
    return jtu.tree_map(lambda x: x / n, a)


def _tree_axpy(a, x, y):
    """Return a * x + y for pytrees."""
    return jtu.tree_map(lambda xi, yi: a * xi + yi, x, y)


def _tree_sub(a, b):
    """Return a - b for pytrees."""
    return jtu.tree_map(lambda ai, bi: ai - bi, a, b)


def _tree_scale(a, x):
    """Return a * x for pytree x and scalar a."""
    return jtu.tree_map(lambda xi: a * xi, x)


def hessian_vector_product(energy_fn, activities, v):
    """Compute Hessian of energy_fn(activities) applied to v (same structure as activities).
    Uses JVP on the gradient for one forward-mode pass instead of a second backward pass.
    """
    grad_energy = jax.grad(energy_fn)
    _, Hv = jax.jvp(grad_energy, (activities,), (v,))
    return Hv


def _random_same_structure(key, template):
    """Random pytree with same structure and shapes as template."""
    leaves, treedef = jtu.tree_flatten(template)
    keys = jr.split(key, len(leaves))
    new_leaves = [jr.normal(k, t.shape, dtype=t.dtype) for k, t in zip(keys, leaves)]
    return jtu.tree_unflatten(treedef, new_leaves)


def power_iteration(hvp_fn, activities_struct, key, n_iters=50, rtol=1e-4):
    """Estimate largest eigenvalue and corresponding eigenvector via power iteration.
    Returns (max_eigenval, v_max) where v_max is normalized.
    Stops early if relative change in eigenvalue is below rtol.
    """
    keys = jr.split(key, n_iters + 1)
    v = _random_same_structure(keys[0], activities_struct)
    v = _tree_normalize(v)
    max_eigenval = None
    for i in range(n_iters):
        Hv = hvp_fn(v)
        lam = _tree_dot(Hv, v)
        lam_f = float(lam)
        if max_eigenval is not None and abs(lam_f - max_eigenval) <= rtol * (abs(max_eigenval) + 1e-14):
            break
        max_eigenval = lam_f
        v = _tree_normalize(Hv)
    max_eigenval = float(_tree_dot(hvp_fn(v), v))
    return max_eigenval, v


def inverse_iteration_cg(hvp_fn, activities_struct, key, n_iters=30, cg_iters=50, cg_rtol=1e-8, rtol=1e-4):
    """Estimate smallest eigenvalue via inverse iteration: solve H w = v with CG, then v = w/|w|.
    Returns min_eigenval (so condition number = max_eigenval / min_eigenval).
    CG stops when residual norm squared is below cg_rtol * initial; outer loop stops when eigenvalue stabilizes (rtol).
    """
    v = _random_same_structure(key, activities_struct)
    v = _tree_normalize(v)
    min_eigenval = None

    for _ in range(n_iters):
        # Conjugate gradient: solve H w = v.
        w = jtu.tree_map(jnp.zeros_like, activities_struct)
        r = jtu.tree_map(jnp.array, v)
        p = jtu.tree_map(jnp.array, v)
        rs_old = _tree_dot(r, r)
        rs_0 = rs_old
        for cg_step in range(cg_iters):
            Hp = hvp_fn(p)
            alpha = rs_old / (_tree_dot(p, Hp) + 1e-14)
            w = _tree_axpy(alpha, p, w)
            r = _tree_sub(r, _tree_scale(alpha, Hp))
            rs_new = _tree_dot(r, r)
            if rs_new <= cg_rtol * (float(rs_0) + 1e-14):
                break
            beta = rs_new / (rs_old + 1e-14)
            p = _tree_axpy(1.0, r, _tree_scale(beta, p))
            rs_old = rs_new
        w_norm_sq = _tree_norm_sq(w)
        w_norm_sq = float(w_norm_sq)
        if w_norm_sq < 1e-20:
            break
        v = _tree_scale(1.0 / np.sqrt(w_norm_sq), w)
        # Rayleigh quotient
        Hv = hvp_fn(v)
        lam = _tree_dot(Hv, v)
        lam_f = float(lam)
        if min_eigenval is not None and abs(lam_f - min_eigenval) <= rtol * (abs(min_eigenval) + 1e-14):
            break
        min_eigenval = lam_f

    Hv = hvp_fn(v)
    min_eigenval = _tree_dot(Hv, v)
    return float(min_eigenval)


def load_cifar10_batch(batch_size, seed):
    """Load a single batch of CIFAR-10 (train) as JAX arrays. Shape: x (B, 3, 32, 32), y (B, 10) one-hot."""
    gen = torch.Generator().manual_seed(seed)
    train_loader, _ = get_dataloaders(
        "CIFAR10",
        batch_size=batch_size,
        flatten=False,
        generator=gen,
    )
    x_batch, y_batch = next(iter(train_loader))
    x = jnp.array(x_batch.cpu().numpy())
    y = jnp.array(y_batch.cpu().numpy())
    return x, y


def load_tinyimagenet_batch(batch_size, split="train", seed=None):
    """Load a single batch of TinyImageNet as JAX arrays.

    Returns:
      - x: (B, 3, 64, 64)
      - y: (B, 200) one-hot labels
    """
    gen = None
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
    train_loader, val_loader = get_tinyimagenet_loaders(
        batch_size=batch_size,
        generator=gen,
    )
    loader = train_loader if split == "train" else val_loader
    x_batch, y_batch = next(iter(loader))
    x = jnp.array(x_batch.cpu().numpy())
    y = jnp.array(y_batch.cpu().numpy())
    return x, y


def setup_theory_experiment(args):
    return os.path.join(
        args.results_dir,
        f"{args.in_channels}_in_channels",
        f"{args.input_size}_input_size",
        f"{args.width}_width",
        f"{args.n_res_blocks}_n_res_blocks",
        f"{args.out_features}_out_features",
        f"{args.param_type}_param_type",
        f"{args.act_fn}_act_fn",
        f"{args.batch_size}_batch_size",
        f"{args.n_steps}_n_steps",
        f"{args.param_lr}_param_lr",
        f"{args.activity_lr}_activity_lr",
        f"{args.n_infer_iters}_n_infer_iters",
        f"{args.loss_id}_loss_id",
        f"use_amortiser_{args.use_amortiser}",
        str(args.seed),
    )


def load_imagenet_batch(batch_size: int, seed: int | None = None, n_examples: int = 1000):
    """
    Load a single fixed ImageNet batch via Hugging Face streaming.

    Samples are drawn from the first `n_examples` elements of the streaming
    ImageNet-1K dataset (`timm/imagenet-1k-wds`), then a random subset of size
    `batch_size` is selected (seeded).
    """
    try:
        from datasets import load_dataset
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Failed to import `datasets.load_dataset`. "
            "Make sure Hugging Face `datasets` is installed and not shadowed."
        ) from e

    try:
        from torchvision import transforms
    except Exception as e:  # pragma: no cover
        raise ImportError("Failed to import `torchvision.transforms`.") from e

    # Stream the dataset instead of downloading 150GB.
    # NOTE: `timm/imagenet-1k-wds` is gated on the Hub; provide an auth token.
    import os as _os

    hf_token = _os.environ.get("HF_TOKEN") or _os.environ.get("HUGGINGFACE_HUB_TOKEN")
    dataset = load_dataset(
        "timm/imagenet-1k-wds",
        streaming=True,
        token=hf_token,
    )

    # Just grab the first 1000 examples
    subset = dataset["train"].take(n_examples)

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    xs, ys = [], []
    for ex in subset:
        # timm's WebDataset schema typically exposes:
        # - `jpg`: image bytes or PIL image
        # - `cls`: class index
        # but we keep fallbacks for robustness.
        img = None
        for k in ("image", "img", "pixels", "jpg"):
            v = ex.get(k, None)
            if v is not None:
                img = v
                break
        if img is None:
            raise KeyError(
                f"Could not find an image in dataset example keys: {list(ex.keys())}"
            )

        if isinstance(img, (bytes, bytearray)):
            import io
            from PIL import Image

            img = Image.open(io.BytesIO(img)).convert("RGB")
        elif hasattr(img, "convert"):
            img = img.convert("RGB")
        else:
            raise TypeError(f"Unrecognized image type for key 'jpg': {type(img)}")

        x = preprocess(img)  # torch.Tensor (3, 224, 224)
        xs.append(x.cpu().numpy().astype(np.float32))

        label = None
        for k in ("label", "labels", "cls"):
            v = ex.get(k, None)
            if v is not None:
                label = v
                break
        if label is None:
            raise KeyError(
                f"Could not find a label in dataset example keys: {list(ex.keys())}"
            )

        # label should be an int in [0, 999]
        if not isinstance(label, (int, np.integer)):
            label = int(label)

        y = np.zeros((1000,), dtype=np.float32)
        y[label] = 1.0
        ys.append(y)

    n_collected = len(xs)
    if n_collected == 0:
        raise RuntimeError("ImageNet streaming subset returned 0 examples.")

    xs = np.stack(xs, axis=0)
    ys = np.stack(ys, axis=0)

    rng = np.random.default_rng(seed)
    if batch_size <= n_collected:
        idx = rng.choice(n_collected, size=batch_size, replace=False)
    else:
        idx = rng.choice(n_collected, size=batch_size, replace=True)

    x_batch = jnp.array(xs[idx])
    y_batch = jnp.array(ys[idx])
    return x_batch, y_batch
