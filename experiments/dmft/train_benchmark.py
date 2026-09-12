"""Finite-size PC vs backprop benchmark on classification datasets.

Trains a predictive-coding network and a backprop network on the **same**
minibatches (augment once, then both update) for ``--n_epochs`` passes over
the training set, then overlays train/test loss and accuracy.

``--fixed_subset`` freezes one unaugmented train batch and one held-out
test batch, both of size ``--batch_size`` (sampled with ``--seed``, like
``analyse_alignment.py``). Every epoch is one GD step on the train
subset. Train metrics use that subset; test metrics use the equal-sized
held-out subset.

``tiny-CIFAR10`` is the alignment binary task: grayscale CIFAR-10,
classes 0 vs 1, labels ``{-1, +1}``, MLP, MSE. Without ``--fixed_subset``
it uses all 10k two-class train images; with it, ``--batch_size`` (even)
examples, balanced.

If any of ``--n_hidden``, ``--width``, ``--batch_size``, ``--n_res_blocks``,
``--param_lr``, ``--param_lr_pc``, ``--activity_lr``, or ``--n_infer_iters``
is given as a list, runs a Cartesian hyperparameter sweep. Backprop and PC
are trained separately: shared architecture/batch axes apply to both,
``--param_lr`` is BP-only, and ``--param_lr_pc`` / ``--activity_lr`` /
``--n_infer_iters`` are PC-only. ``--pc_infer_mode closed_form`` skips
activity GD and updates PC parameters from the linear equilibrium
energy (requires ``--act_fn linear``, ``--loss_id mse``, and
``--arch mlp``). Configs are ranked by mean final test accuracy across
``--n_seeds``. Sweep runs are stored under
``hp_sweep/{bp|pc}/key=value/.../seed=N`` with a compact
``sweep_summary.json``.

``--keep_npy`` keeps per-run history ``*.npy`` files after plotting
(default: delete them). ``--plot_from_npy`` skips training and rebuilds
figures from those files (same hyperparameters; does not delete npy).
Requires a prior run with ``--keep_npy``.

Energy scalings match ``train.py`` / ``train_pcn`` for MLPs:

    λ = γ² N L    (µPC output precision)
    κ = L         (µPC hidden precision)

with a single ``L = n_hidden + 1``. CNNs split this into two depths
(``--resnet_fwd_l``, ``--resnet_energy_l``; see those flags). Hidden
precision can be restricted with ``--hidden_energy_layers``.

For Adam, the PC parameter LR is divided the same way as BP
(``1/√N``, or ``1/√(N L)`` with MLP skips). GD and SGD+momentum keep the
``train.py`` convention: PC uses the raw ``--param_lr_pc`` (scale lives
in the energy); BP bakes ``γ² N`` into the optimiser. No Adam-style
``1/√N`` rescaling is applied for SGD+momentum.

CNN residual blocks still use the per-parameter Adam tree from
``configure_cnn_param_optim``, with residual Adam LRs using ``L_fwd``.
ImageNet is streamed from Hugging Face.

Default architecture: MLP for MNIST / Fashion-MNIST / tiny-CIFAR10,
CNN otherwise. Override with ``--arch``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import torch
from torch.utils.data import DataLoader, TensorDataset

import jpc

from experiments.datasets import (
    get_dataset,
    get_tinyimagenet_loaders,
    TinyImageNet,
)
from experiments.dmft.utils import (
    CIFAR_GRAY_DIM,
    MLP,
    copy_mlp_linear_params,
    create_tiny_cifar10_dataset,
    get_hidden_energy_scaling,
    get_output_energy_scaling,
)
from experiments.limits_paper.utils import configure_param_optim
from experiments.mupc_paper.utils import set_seed

try:
    from experiments.dmft import cnn_pc
except ImportError:
    import cnn_pc

_CNN_DIR = Path(__file__).resolve().parents[1] / "limits_paper" / "cnn"
if str(_CNN_DIR) not in sys.path:
    sys.path.insert(0, str(_CNN_DIR))

from model import ResNet  # noqa: E402
from optim import configure_cnn_param_optim  # noqa: E402
from experiments.limits_paper.cnn.utils import _import_hf_load_dataset  # noqa: E402

import plot_style as ps

ps.apply_paper_style()


DATASET_ALIASES = {
    "mnist": "MNIST",
    "fashion-mnist": "Fashion-MNIST",
    "fashionmnist": "Fashion-MNIST",
    "cifar10": "CIFAR10",
    "cifar-10": "CIFAR10",
    "cifar": "CIFAR10",
    "tinyimagenet": "TinyImageNet",
    "tiny-imagenet": "TinyImageNet",
    "tiny_imagenet": "TinyImageNet",
    "imagenet": "ImageNet",
    "tiny-cifar10": "tiny-CIFAR10",
    "tinycifar10": "tiny-CIFAR10",
    "tiny_cifar10": "tiny-CIFAR10",
}

DATASET_SPECS = {
    "MNIST": dict(in_channels=1, input_size=28, n_classes=10, flatten_dim=784),
    "Fashion-MNIST": dict(
        in_channels=1, input_size=28, n_classes=10, flatten_dim=784
    ),
    "CIFAR10": dict(
        in_channels=3, input_size=32, n_classes=10, flatten_dim=3072
    ),
    "tiny-CIFAR10": dict(
        in_channels=1, input_size=32, n_classes=1, flatten_dim=CIFAR_GRAY_DIM
    ),
    "TinyImageNet": dict(
        in_channels=3, input_size=64, n_classes=200, flatten_dim=12288
    ),
    "ImageNet": dict(
        in_channels=3, input_size=224, n_classes=1000, flatten_dim=150528
    ),
}

MLP_DATASETS = {"MNIST", "Fashion-MNIST", "tiny-CIFAR10"}
TINY_CIFAR10_TRAIN_SIZE = 10_000
TINY_CIFAR10_TEST_SIZE = 2_000
IMAGENET_TRAIN_SIZE = 1_281_167
IMAGENET_VAL_SIZE = 50_000
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def normalize_dataset_id(name):
    key = name.strip().lower().replace(" ", "-")
    if key in DATASET_ALIASES:
        return DATASET_ALIASES[key]
    if name in DATASET_SPECS:
        return name
    raise ValueError(
        f"Unknown dataset '{name}'. Options: MNIST, Fashion-MNIST, "
        "CIFAR10, tiny-CIFAR10, TinyImageNet, ImageNet."
    )


def default_arch_for_dataset(dataset_id):
    return "mlp" if dataset_id in MLP_DATASETS else "cnn"


def cnn_energy_depth(n_res_blocks, additive_depth_factor):
    """Energy depth ``L = n_res_blocks + additive_depth_factor``.

    Prefer ``cnn_pc.resolve_cnn_ls`` when both ``L_fwd`` and ``L_energy``
    are needed. Kept for the old ``L = R + factor`` formula.
    """
    return int(n_res_blocks) + int(additive_depth_factor)


def _cnn_depth_title(fwd_l, energy_l):
    if fwd_l == energy_l:
        return f"L={energy_l}"
    return f"L_fwd={fwd_l}, L_energy={energy_l}"


def copy_eqx_arrays(src, dst):
    """Copy array leaves from ``src`` onto ``dst`` (same pytree structure)."""
    src_params, _ = eqx.partition(src, eqx.is_array)
    _, dst_static = eqx.partition(dst, eqx.is_array)
    return eqx.combine(src_params, dst_static)


def mlp_adam_lr(param_lr, param_type, use_skips, width, depth):
    """Adam LR matching ``configure_param_optim`` (BP and, here, PC)."""
    if param_type == "sp":
        return param_lr
    if use_skips:
        return param_lr / (np.sqrt(width) * np.sqrt(depth))
    return param_lr / np.sqrt(width)


def bp_gd_style_lr(args):
    """BP GD / SGD+momentum LR: µP bakes ``γ² N`` into the optimiser."""
    if args.param_type == "sp":
        return args.param_lr
    return args.param_lr * (args.gamma ** 2) * args.width


def make_sgd_param_optim(learning_rate, args):
    """Vanilla GD or SGD+momentum. No Adam-style ``1/√N`` rescaling."""
    if args.param_optim == "sgd_momentum":
        return optax.sgd(learning_rate, momentum=args.momentum)
    return optax.sgd(learning_rate)


def supervised_loss(preds, y, loss_id):
    if loss_id == "mse":
        return jpc.mse_loss(preds, y)
    return jpc.cross_entropy_loss(preds, y)


def accuracy_pct(preds, y):
    y = jnp.asarray(y)
    preds = jnp.asarray(preds)
    if y.ndim == 1 or y.shape[-1] == 1:
        y = y.reshape(-1)
        preds = preds.reshape(-1)
        return float(jnp.mean(jnp.sign(preds) == jnp.sign(y)) * 100.0)
    return float(jpc.compute_accuracy(y, preds))


def _to_numpy_batch(x, y):
    if hasattr(x, "numpy"):
        x = x.numpy()
    if hasattr(y, "numpy"):
        y = y.numpy()
    return np.asarray(x), np.asarray(y)


def maybe_flatten(x, arch):
    if arch == "mlp" and np.ndim(x) > 2:
        return np.reshape(x, (x.shape[0], -1))
    return x


def to_jax_batch(x, y, arch):
    x, y = _to_numpy_batch(x, y)
    x = maybe_flatten(x, arch)
    return jnp.asarray(x), jnp.asarray(y)


def _parse_imagenet_example(ex, transform):
    img = None
    for key in ("image", "img", "pixels", "jpg"):
        if key in ex and ex[key] is not None:
            img = ex[key]
            break
    if img is None:
        raise KeyError(
            f"Could not find an image in example keys: {list(ex.keys())}"
        )
    if isinstance(img, (bytes, bytearray)):
        import io
        from PIL import Image

        img = Image.open(io.BytesIO(img)).convert("RGB")
    elif hasattr(img, "convert"):
        img = img.convert("RGB")
    else:
        raise TypeError(f"Unrecognized ImageNet image type: {type(img)}")

    x = transform(img).cpu().numpy().astype(np.float32)

    label = None
    for key in ("label", "labels", "cls"):
        if key in ex and ex[key] is not None:
            label = ex[key]
            break
    if label is None:
        raise KeyError(
            f"Could not find a label in example keys: {list(ex.keys())}"
        )
    label = int(label)
    y = np.zeros((1000,), dtype=np.float32)
    y[label] = 1.0
    return x, y


def _hf_imagenet_split(dataset, names):
    for name in names:
        if name in dataset:
            return dataset[name]
    raise KeyError(
        f"None of {names} found in ImageNet splits {list(dataset.keys())}"
    )


def iter_imagenet_hf(
    split,
    batch_size,
    seed,
    n_examples,
    *,
    train,
    drop_last=True,
):
    """Yield JAX ``(x, y)`` batches from streamed ImageNet-1K."""
    try:
        load_dataset = _import_hf_load_dataset()
    except Exception as exc:
        raise ImportError(
            "Failed to import Hugging Face `datasets.load_dataset`. "
            "Install `datasets` and set HF_TOKEN for gated ImageNet-1K."
        ) from exc

    from torchvision import transforms

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGINGFACE_HUB_TOKEN"
    )
    dataset = load_dataset(
        "timm/imagenet-1k-wds",
        streaming=True,
        token=hf_token,
    )
    if train:
        split_ds = _hf_imagenet_split(dataset, ("train",))
        split_ds = split_ds.shuffle(seed=int(seed), buffer_size=10_000)
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    else:
        split_ds = _hf_imagenet_split(
            dataset, ("validation", "val", "test")
        )
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    xs, ys = [], []
    n_seen = 0
    for ex in split_ds:
        if n_seen >= n_examples:
            break
        x, y = _parse_imagenet_example(ex, transform)
        xs.append(x)
        ys.append(y)
        n_seen += 1
        if len(xs) == batch_size:
            yield (
                jnp.asarray(np.stack(xs, axis=0)),
                jnp.asarray(np.stack(ys, axis=0)),
            )
            xs, ys = [], []

    if xs and not drop_last:
        yield (
            jnp.asarray(np.stack(xs, axis=0)),
            jnp.asarray(np.stack(ys, axis=0)),
        )


def _targets_2d(y):
    y = jnp.asarray(y, dtype=jnp.float32)
    if y.ndim == 1:
        return y[:, None]
    return y


def _tiny_cifar_xy(key, n_samples, train):
    X, y = create_tiny_cifar10_dataset(
        key=key, D=CIFAR_GRAY_DIM, P=n_samples, train=train
    )
    return jnp.asarray(X.T, dtype=jnp.float32), _targets_2d(y)


def _loader_from_xy(x, y, batch_size, *, shuffle, drop_last, seed=0):
    dataset = TensorDataset(
        torch.from_numpy(np.array(x, copy=True)),
        torch.from_numpy(np.array(y, copy=True)),
    )
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    if shuffle:
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        kwargs["generator"] = gen
    return DataLoader(**kwargs)


def _materialize_subset(dataset, n, seed, arch, dataset_id=None):
    if n > len(dataset):
        raise ValueError(
            f"--fixed_subset requested {n} examples but the dataset has "
            f"{len(dataset)}"
        )
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=gen)[:n].tolist()
    xs, ys = [], []
    for idx in indices:
        x, y = dataset[idx]
        xs.append(x)
        ys.append(y)
    x = torch.stack([torch.as_tensor(item) for item in xs])
    y = torch.stack([torch.as_tensor(item) for item in ys])
    if dataset_id == "CIFAR10":
        x = (x - 0.5) / 0.5
    return to_jax_batch(x, y, arch)


def _unaugmented_train_dataset(dataset_id, flatten):
    """Train split with deterministic (eval-style) transforms."""
    if dataset_id in ("MNIST", "Fashion-MNIST"):
        return get_dataset(
            id=dataset_id, train=True, normalise=True, flatten=flatten
        )
    if dataset_id == "CIFAR10":
        return get_dataset(
            id=dataset_id, train=True, normalise=False, flatten=flatten
        )
    if dataset_id == "TinyImageNet":
        dataset = TinyImageNet(split="train")
        dataset.transform = dataset._make_transform("val")
        return dataset
    raise ValueError(f"No unaugmented train set for '{dataset_id}'")


def _test_dataset(dataset_id, flatten):
    if dataset_id == "TinyImageNet":
        return TinyImageNet(split="val")
    return get_dataset(
        id=dataset_id, train=False, normalise=True, flatten=flatten
    )


def _imagenet_fixed_batch(split, n, seed):
    raw = iter_imagenet_hf(
        split=split,
        batch_size=n,
        seed=seed,
        n_examples=n,
        train=False,
        drop_last=False,
    )
    return next(raw)


def prepare_data(args):
    """Attach loaders and optional frozen train/test batches on ``args``."""
    args._fixed_train_batch = None
    args._fixed_test_batch = None
    args._train_loader = None
    args._test_loader = None
    flatten = args.arch == "mlp"
    n = int(args.batch_size)

    if args.dataset == "tiny-CIFAR10":
        data_key = jr.PRNGKey(args.seed)
        train_key, test_key = jr.split(data_key)
        if uses_fixed_subset(args):
            if n % 2 != 0:
                raise ValueError(
                    "tiny-CIFAR10 --fixed_subset needs an even --batch_size "
                    f"(got {n}) so the two classes stay balanced"
                )
            args._fixed_train_batch = _tiny_cifar_xy(train_key, n, train=True)
            args._fixed_test_batch = _tiny_cifar_xy(test_key, n, train=False)
            return
        x_train, y_train = _tiny_cifar_xy(
            train_key, TINY_CIFAR10_TRAIN_SIZE, train=True
        )
        x_test, y_test = _tiny_cifar_xy(
            test_key, TINY_CIFAR10_TEST_SIZE, train=False
        )
        args._train_loader = _loader_from_xy(
            x_train,
            y_train,
            n,
            shuffle=True,
            drop_last=True,
            seed=args.seed,
        )
        args._test_loader = _loader_from_xy(
            x_test, y_test, n, shuffle=False, drop_last=False
        )
        return

    if args.dataset == "ImageNet":
        if uses_fixed_subset(args):
            args._fixed_train_batch = _imagenet_fixed_batch(
                "train", n, args.seed
            )
            args._fixed_test_batch = _imagenet_fixed_batch(
                "val", n, args.seed + 1
            )
        return

    if uses_fixed_subset(args):
        train_data = _unaugmented_train_dataset(args.dataset, flatten)
        test_data = _test_dataset(args.dataset, flatten)
        args._fixed_train_batch = _materialize_subset(
            train_data, n, args.seed, args.arch, args.dataset
        )
        args._fixed_test_batch = _materialize_subset(
            test_data, n, args.seed + 1, args.arch, args.dataset
        )
        return

    args._train_loader, args._test_loader = make_torch_loaders(
        args.dataset, n, flatten, args.seed
    )


def make_torch_loaders(dataset_id, batch_size, flatten, seed):
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    if dataset_id in ("MNIST", "Fashion-MNIST", "CIFAR10"):
        train_data = get_dataset(
            id=dataset_id, train=True, normalise=True, flatten=flatten
        )
        test_data = get_dataset(
            id=dataset_id, train=False, normalise=True, flatten=flatten
        )
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            generator=gen,
        )
        test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
        return train_loader, test_loader

    if dataset_id == "TinyImageNet":
        train_loader, _ = get_tinyimagenet_loaders(
            batch_size=batch_size, generator=gen
        )
        val_data = TinyImageNet(split="val")
        test_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        return train_loader, test_loader

    raise ValueError(f"No torch loaders for dataset '{dataset_id}'")


def _iter_prepared(raw_batches, arch):
    for x, y in raw_batches:
        yield to_jax_batch(x, y, arch)


def _iter_fixed_batch(batch):
    x, y = batch
    yield x, y


def iter_train_batches(args, epoch):
    if getattr(args, "_fixed_train_batch", None) is not None:
        return _iter_fixed_batch(args._fixed_train_batch)
    if args.dataset == "ImageNet":
        raw = iter_imagenet_hf(
            split="train",
            batch_size=args.batch_size,
            seed=args.seed + epoch,
            n_examples=IMAGENET_TRAIN_SIZE,
            train=True,
            drop_last=True,
        )
    else:
        raw = args._train_loader
    return _iter_prepared(raw, args.arch)


def iter_eval_train_batches(args):
    """Training-set batches for eval; does not consume the shuffled train loader."""
    if getattr(args, "_fixed_train_batch", None) is not None:
        return _iter_fixed_batch(args._fixed_train_batch)
    if args.dataset == "ImageNet":
        raw = iter_imagenet_hf(
            split="train",
            batch_size=args.batch_size,
            seed=args.seed,
            n_examples=IMAGENET_TRAIN_SIZE,
            train=True,
            drop_last=False,
        )
    else:
        raw = DataLoader(
            args._train_loader.dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )
    return _iter_prepared(raw, args.arch)


def iter_test_batches(args):
    if getattr(args, "_fixed_test_batch", None) is not None:
        return _iter_fixed_batch(args._fixed_test_batch)
    if args.dataset == "ImageNet":
        raw = iter_imagenet_hf(
            split="val",
            batch_size=args.batch_size,
            seed=args.seed,
            n_examples=IMAGENET_VAL_SIZE,
            train=False,
            drop_last=False,
        )
    else:
        raw = args._test_loader
    return _iter_prepared(raw, args.arch)


def make_models(key, args, spec):
    if args.arch == "mlp":
        pc_model = jpc.make_mlp(
            key,
            input_dim=spec["flatten_dim"],
            width=args.width,
            depth=args.n_hidden + 1,
            output_dim=spec["n_classes"],
            act_fn=args.act_fn,
            use_bias=False,
            param_type=args.param_type,
        )
        bp_model = MLP(
            key=key,
            d_in=spec["flatten_dim"],
            N=args.width,
            L=args.n_hidden + 1,
            d_out=spec["n_classes"],
            act_fn=args.act_fn,
            param_type=args.param_type,
            gamma=args.gamma,
            use_bias=False,
            use_skips=args.use_skips,
        )
        bp_model = copy_mlp_linear_params(pc_model, bp_model)
        skip_model = (
            jpc.make_skip_model(len(pc_model)) if args.use_skips else None
        )
        return pc_model, bp_model, skip_model

    fwd_l, _ = cnn_pc.resolve_cnn_ls(args)
    fwd_additive = cnn_pc.resnet_fwd_additive_depth_factor(
        fwd_l, args.n_res_blocks
    )
    pc_model = ResNet(
        key=key,
        width=args.width,
        n_res_blocks=args.n_res_blocks,
        in_channels=spec["in_channels"],
        input_size=spec["input_size"],
        out_features=spec["n_classes"],
        param_type=args.param_type,
        act_fn=args.act_fn,
        scale_non_res_layers=args.scale_non_res_layers,
        additive_depth_factor=fwd_additive,
    )
    bp_model = ResNet(
        key=key,
        width=args.width,
        n_res_blocks=args.n_res_blocks,
        in_channels=spec["in_channels"],
        input_size=spec["input_size"],
        out_features=spec["n_classes"],
        param_type=args.param_type,
        act_fn=args.act_fn,
        scale_non_res_layers=args.scale_non_res_layers,
        additive_depth_factor=fwd_additive,
    )
    bp_model = copy_eqx_arrays(pc_model, bp_model)
    return pc_model, bp_model, None


def pc_jpc_kwargs(args):
    """Kwargs for jpc init / updates.

    CNN µP lives inside ``ResNet``; passing ``param_type='mupc'`` would apply
    extra MLP scalings (and index ``Linear`` weights that do not exist).
    """
    if args.arch == "cnn":
        return dict(param_type="sp", gamma=None)
    return dict(param_type=args.param_type, gamma=args.gamma)


def make_pc_param_optim(pc_model, skip_model, args, depth):
    if args.arch == "cnn":
        if args.param_optim in ("gd", "sgd_momentum"):
            # train.py PC GD: raw LR; width/gamma/depth live in the energy.
            param_optim = make_sgd_param_optim(args.param_lr_pc, args)
        else:
            param_optim = configure_cnn_param_optim(
                pc_model,
                optim_id=args.param_optim,
                param_type=args.param_type,
                param_lr=args.param_lr_pc,
                width=args.width,
                depth=depth,
                gamma_0=args.gamma,
                params_for_pc=True,
            )
        opt_state = param_optim.init(
            (eqx.filter(pc_model, eqx.is_array), skip_model)
        )
        return param_optim, opt_state

    if args.param_optim in ("gd", "sgd_momentum"):
        param_optim = make_sgd_param_optim(args.param_lr_pc, args)
    elif args.param_optim == "adam":
        param_optim = optax.adam(
            mlp_adam_lr(
                args.param_lr_pc,
                args.param_type,
                args.use_skips,
                args.width,
                depth,
            )
        )
    else:
        raise ValueError(f"Invalid optimiser: {args.param_optim}")
    opt_state = param_optim.init(
        (eqx.filter(pc_model, eqx.is_array), skip_model)
    )
    return param_optim, opt_state


def make_bp_param_optim(bp_model, args, depth):
    if args.param_optim == "sgd_momentum":
        optim = make_sgd_param_optim(bp_gd_style_lr(args), args)
        return optim, optim.init(eqx.filter(bp_model, eqx.is_array))

    if args.arch == "cnn":
        optim = configure_cnn_param_optim(
            bp_model,
            optim_id=args.param_optim,
            param_type=args.param_type,
            param_lr=args.param_lr,
            width=args.width,
            depth=depth,
            gamma_0=args.gamma,
            params_for_pc=False,
        )
        return optim, optim.init(eqx.filter(bp_model, eqx.is_array))

    optim = configure_param_optim(
        args.param_optim,
        args.param_type,
        args.use_skips,
        args.param_lr,
        args.width,
        depth,
        args.gamma,
    )
    return optim, optim.init(eqx.filter(bp_model, eqx.is_array))


def pc_ffwd_preds(model, x, skip_model, jpc_kw):
    activities = jpc.init_activities_with_ffwd(
        model=model,
        input=x,
        skip_model=skip_model,
        **jpc_kw,
    )
    return activities[-1]


def pc_batch_metrics(model, x, y, skip_model, jpc_kw, loss_id):
    preds = pc_ffwd_preds(model, x, skip_model, jpc_kw)
    return float(supervised_loss(preds, y, loss_id)), float(accuracy_pct(preds, y))


def bp_batch_metrics(model, x, y, loss_id):
    preds = jax.vmap(model)(x)
    return float(supervised_loss(preds, y, loss_id)), float(accuracy_pct(preds, y))


def uses_fixed_subset(args):
    return bool(getattr(args, "fixed_subset", False))


def n_train_batches(args):
    if uses_fixed_subset(args):
        return 1
    if args.dataset == "ImageNet":
        return IMAGENET_TRAIN_SIZE // args.batch_size
    return len(args._train_loader)


def mini_epoch_boundaries(n_batches, n_mini):
    """Batch indices (1-based) at which a mini-epoch ends, including the last."""
    n_mini = max(1, min(int(n_mini), int(n_batches)))
    bounds = []
    for i in range(n_mini):
        bound = int(round((i + 1) * n_batches / n_mini))
        bound = min(max(bound, 1), n_batches)
        if not bounds or bound > bounds[-1]:
            bounds.append(bound)
    if bounds[-1] != n_batches:
        bounds[-1] = n_batches
    return set(bounds)


def _evaluate_batches(
    batches,
    args,
    *,
    bp_model=None,
    pc_model=None,
    skip_model=None,
    jpc_kw=None,
):
    """Size-weighted feedforward loss/acc over ``batches`` (no updates)."""
    skip_pc = pc_model is None
    skip_bp = bp_model is None
    pc_loss_sum = pc_acc_sum = 0.0
    bp_loss_sum = bp_acc_sum = 0.0
    n_seen = 0
    for x, y in batches:
        b = int(x.shape[0])
        if not skip_bp:
            bp_loss, bp_acc = bp_batch_metrics(bp_model, x, y, args.loss_id)
            bp_loss_sum += bp_loss * b
            bp_acc_sum += bp_acc * b
        if not skip_pc:
            pc_loss, pc_acc = pc_batch_metrics(
                pc_model, x, y, skip_model, jpc_kw, args.loss_id
            )
            pc_loss_sum += pc_loss * b
            pc_acc_sum += pc_acc * b
        n_seen += b
    empty = (float("nan"), float("nan"))
    if n_seen == 0:
        return empty, empty
    pc_metrics = (
        empty if skip_pc else (pc_loss_sum / n_seen, pc_acc_sum / n_seen)
    )
    bp_metrics = (
        empty if skip_bp else (bp_loss_sum / n_seen, bp_acc_sum / n_seen)
    )
    return pc_metrics, bp_metrics


def evaluate_pc(model, args, skip_model, jpc_kw, batches=None):
    if batches is None:
        batches = iter_test_batches(args)
    pc_metrics, _ = _evaluate_batches(
        batches,
        args,
        pc_model=model,
        skip_model=skip_model,
        jpc_kw=jpc_kw,
    )
    return pc_metrics


def evaluate_bp(model, args, batches=None):
    if batches is None:
        batches = iter_test_batches(args)
    _, bp_metrics = _evaluate_batches(batches, args, bp_model=model)
    return bp_metrics


def evaluate_train(bp_model, args, *, pc_model=None, skip_model=None, jpc_kw=None):
    """Training-set feedforward metrics at the current parameters."""
    return _evaluate_batches(
        iter_eval_train_batches(args),
        args,
        bp_model=bp_model,
        pc_model=pc_model,
        skip_model=skip_model,
        jpc_kw=jpc_kw,
    )


def current_train_metrics(
    args,
    *,
    bp_model,
    pc_model,
    skip_model,
    jpc_kw,
    skip_pc,
    skip_bp,
    nan_metrics,
):
    """Feedforward train loss/acc of the current weights (frozen subset if set)."""
    if skip_pc:
        _, bp_train = evaluate_train(bp_model, args)
        return nan_metrics, bp_train
    if skip_bp:
        pc_train, _ = evaluate_train(
            None,
            args,
            pc_model=pc_model,
            skip_model=skip_model,
            jpc_kw=jpc_kw,
        )
        return pc_train, nan_metrics
    return evaluate_train(
        bp_model,
        args,
        pc_model=pc_model,
        skip_model=skip_model,
        jpc_kw=jpc_kw,
    )


def pc_closed_form(args):
    return getattr(args, "pc_infer_mode", "infer") == "closed_form"


def pc_infer_and_update(
    model,
    skip_model,
    x,
    y,
    activity_optim,
    param_optim,
    param_opt_state,
    args,
    jpc_kw,
    output_energy_scaling,
    hidden_energy_scaling,
):
    params = (model, skip_model)
    if pc_closed_form(args):
        energy = jpc.linear_equilib_energy(
            params=params,
            x=x,
            y=y,
            output_energy_scaling=output_energy_scaling,
            hidden_energy_scaling=hidden_energy_scaling,
            **jpc_kw,
        )
        energy = float(energy)
        if not np.isfinite(energy):
            return model, skip_model, param_opt_state, energy, False
        param_result = jpc.update_linear_equilib_energy_params(
            params=params,
            optim=param_optim,
            opt_state=param_opt_state,
            x=x,
            y=y,
            output_energy_scaling=output_energy_scaling,
            hidden_energy_scaling=hidden_energy_scaling,
            **jpc_kw,
        )
        return (
            param_result["model"],
            param_result["skip_model"],
            param_result["opt_state"],
            energy,
            True,
        )

    activities = jpc.init_activities_with_ffwd(
        model=model,
        input=x,
        skip_model=skip_model,
        **jpc_kw,
    )
    activity_opt_state = activity_optim.init(activities)
    energy = None
    update_activities = (
        cnn_pc.update_pc_activities
        if args.arch == "cnn"
        else jpc.update_pc_activities
    )
    update_params = (
        cnn_pc.update_pc_params if args.arch == "cnn" else jpc.update_pc_params
    )
    extra_kw = {} if args.arch == "cnn" else jpc_kw
    for _ in range(args.n_infer_iters):
        result = update_activities(
            params=params,
            activities=activities,
            optim=activity_optim,
            opt_state=activity_opt_state,
            output=y,
            input=x,
            loss_id=args.loss_id,
            output_energy_scaling=output_energy_scaling,
            hidden_energy_scaling=hidden_energy_scaling,
            **extra_kw,
        )
        activities = result["activities"]
        activity_opt_state = result["opt_state"]
        energy = result["energy"]

    energy = float(energy)
    if not np.isfinite(energy):
        return model, skip_model, param_opt_state, energy, False

    param_result = update_params(
        params=params,
        activities=activities,
        optim=param_optim,
        opt_state=param_opt_state,
        output=y,
        input=x,
        loss_id=args.loss_id,
        output_energy_scaling=output_energy_scaling,
        hidden_energy_scaling=hidden_energy_scaling,
        **extra_kw,
    )
    return (
        param_result["model"],
        param_result["skip_model"],
        param_result["opt_state"],
        energy,
        True,
    )


def make_bp_step(loss_id):
    @eqx.filter_jit
    def loss_fn(model, x, y):
        preds = jax.vmap(model)(x)
        return supervised_loss(preds, y, loss_id)

    @eqx.filter_jit
    def step(model, opt_state, optim, x, y):
        _, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = optim.update(
            updates=grads,
            state=opt_state,
            params=eqx.filter(model, eqx.is_array),
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state

    return step


def setup_save_dir(args, seed_tag=None):
    depth_tag = (
        f"{args.n_hidden}_n_hidden"
        if args.arch == "mlp"
        else f"{args.n_res_blocks}_n_res_blocks"
    )
    if seed_tag is None:
        seed_tag = str(args.seed)
    return os.path.join(
        args.results_dir,
        args.dataset,
        args.arch,
        args.loss_id,
        f"{args.width}_width",
        depth_tag,
        f"{args.act_fn}_act_fn",
        f"{args.param_type}_param_type",
        f"{args.gamma}_gamma",
        f"{args.param_optim}_param_optim",
        *(
            [f"{args.momentum}_momentum"]
            if args.param_optim == "sgd_momentum"
            else []
        ),
        f"{args.param_lr}_param_lr",
        f"{args.param_lr_pc}_param_lr_pc",
        f"{args.batch_size}_batch_size",
        *(
            [f"{args.fixed_subset}_fixed_subset"]
            if uses_fixed_subset(args)
            else []
        ),
        f"{args.n_epochs}_n_epochs",
        *(
            [f"{args.pc_infer_mode}_pc_infer_mode"]
            if pc_closed_form(args)
            else [
                f"{args.n_infer_iters}_n_infer_iters",
                f"{args.activity_lr}_activity_lr",
            ]
        ),
        f"{args.use_skips}_use_skips",
        *(
            [
                f"{getattr(args, 'resnet_fwd_l', 'n_res_blocks')}_resnet_fwd_l",
                f"{getattr(args, 'resnet_energy_l', 'n_weight_layers')}_resnet_energy_l",
                f"{getattr(args, 'hidden_energy_layers', 'weight')}_hidden_energy_layers",
            ]
            if args.arch == "cnn"
            else []
        ),
        *(
            [f"{args.additive_depth_factor}_additive_depth_factor"]
            if args.arch == "cnn"
            and getattr(args, "additive_depth_factor", None) is not None
            else []
        ),
        f"{args.skip_pc}_skip_pc",
        f"{getattr(args, 'skip_bp', False)}_skip_bp",
        seed_tag,
    )


def _aggregate_curves(curves):
    """Mean and SEM across seed curves, truncated to the shortest length."""
    arrays = [np.asarray(c, dtype=np.float64) for c in curves]
    n_t = min(len(a) for a in arrays)
    stacked = np.stack([a[:n_t] for a in arrays], axis=0)
    mean = stacked.mean(axis=0)
    if stacked.shape[0] == 1:
        sem = np.zeros_like(mean)
    else:
        sem = stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
    return mean, sem, n_t


_EPOCH_XLABEL = "epoch"
_STEP_XLABEL = "step"
_TEST_LOSS_YLABEL = r"test loss $\mathcal{L}$"
_TEST_ACC_YLABEL = "test accuracy (%)"
_TRAIN_ACC_YLABEL = "train accuracy (%)"

#: Combined 2x2 (train+test) is supplementary; each cell is roughly half-width.
_FIGSIZE_2x2 = ps.per_layer_figsize(2, 2)
#: One matplotlib figure spanning the text width; internal spacing comes
#: from constrained layout, not the Inkscape assembly gutter.
_FIGSIZE_1x2 = (ps.TEXT_WIDTH_IN, ps.PANEL_HALF[1])


def _markersize_for_n(n):
    return 1.6 if n > 20 else ps.MARKER_SIZE


def _plot_overlay(
    ax,
    xs_pc,
    ys_pc,
    xs_bp,
    ys_bp,
    xlabel,
    ylabel,
    *,
    yerr_pc=None,
    yerr_bp=None,
    skip_pc=False,
    skip_bp=False,
):
    n_pts = 0
    if not skip_pc:
        n_pts = max(n_pts, len(np.asarray(xs_pc)))
    if not skip_bp:
        n_pts = max(n_pts, len(np.asarray(xs_bp)))
    markersize = _markersize_for_n(n_pts)
    if not skip_pc:
        ax.plot(
            xs_pc,
            ys_pc,
            marker="o",
            markersize=markersize,
            color=ps.COLOR_PC,
            label=ps.LABEL_PC,
        )
        if yerr_pc is not None:
            ax.fill_between(
                xs_pc,
                np.asarray(ys_pc) - np.asarray(yerr_pc),
                np.asarray(ys_pc) + np.asarray(yerr_pc),
                color=ps.COLOR_PC,
                alpha=0.2,
                linewidth=0,
            )
    if not skip_bp:
        ax.plot(
            xs_bp,
            ys_bp,
            marker="s",
            markersize=markersize,
            color=ps.COLOR_BP,
            label=ps.LABEL_BP,
        )
        if yerr_bp is not None:
            ax.fill_between(
                xs_bp,
                np.asarray(ys_bp) - np.asarray(yerr_bp),
                np.asarray(ys_bp) + np.asarray(yerr_bp),
                color=ps.COLOR_BP,
                alpha=0.2,
                linewidth=0,
            )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlabel in (_EPOCH_XLABEL, _STEP_XLABEL):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.legend()
    ps.style_axes(ax)


def _metric_plot_stem(skip_pc, skip_bp=False):
    if skip_pc:
        return "bp"
    if skip_bp:
        return "pc"
    return "pc_bp"


def _panel_spec(xs_pc, ys_pc, xs_bp, ys_bp, xlabel, ylabel, yerr_pc=None, yerr_bp=None):
    return (xs_pc, ys_pc, xs_bp, ys_bp, xlabel, ylabel, yerr_pc, yerr_bp)


def _save_overlay_figure(shape, figsize, panels, save_path, plot_kw):
    fig, axes = plt.subplots(*shape, figsize=figsize)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, panel in zip(axes_flat, panels):
        xs_pc, ys_pc, xs_bp, ys_bp, xlabel, ylabel, yerr_pc, yerr_bp = panel
        _plot_overlay(
            ax,
            xs_pc,
            ys_pc,
            xs_bp,
            ys_bp,
            xlabel,
            ylabel,
            yerr_pc=yerr_pc,
            yerr_bp=yerr_bp,
            **plot_kw,
        )
    ps.save_figure(fig, save_path)
    return save_path


def _split_train_test_panels(
    *,
    xs_train,
    xs_eval,
    train_loss_pc,
    train_loss_bp,
    test_loss_pc,
    test_loss_bp,
    train_acc_pc,
    train_acc_bp,
    test_acc_pc,
    test_acc_bp,
    yerr=None,
):
    """Return (combined 2x2, test-loss, test-acc, train 1x2) panel lists."""

    def err(key):
        if yerr is None:
            return None, None
        return yerr.get(f"{key}_pc"), yerr.get(f"{key}_bp")

    e_tr_l = err("train_loss")
    e_te_l = err("test_loss")
    e_tr_a = err("train_acc")
    e_te_a = err("test_acc")
    combined = [
        _panel_spec(
            xs_train, train_loss_pc, xs_train, train_loss_bp,
            _EPOCH_XLABEL, ps.LOSS_LABEL, *e_tr_l,
        ),
        _panel_spec(
            xs_eval, test_loss_pc, xs_eval, test_loss_bp,
            _EPOCH_XLABEL, _TEST_LOSS_YLABEL, *e_te_l,
        ),
        _panel_spec(
            xs_train, train_acc_pc, xs_train, train_acc_bp,
            _EPOCH_XLABEL, _TRAIN_ACC_YLABEL, *e_tr_a,
        ),
        _panel_spec(
            xs_eval, test_acc_pc, xs_eval, test_acc_bp,
            _EPOCH_XLABEL, _TEST_ACC_YLABEL, *e_te_a,
        ),
    ]
    test_loss = [
        _panel_spec(
            xs_eval, test_loss_pc, xs_eval, test_loss_bp,
            _EPOCH_XLABEL, _TEST_LOSS_YLABEL, *e_te_l,
        )
    ]
    test_acc = [
        _panel_spec(
            xs_eval, test_acc_pc, xs_eval, test_acc_bp,
            _EPOCH_XLABEL, _TEST_ACC_YLABEL, *e_te_a,
        )
    ]
    train = [
        _panel_spec(
            xs_train, train_loss_pc, xs_train, train_loss_bp,
            _EPOCH_XLABEL, ps.LOSS_LABEL, *e_tr_l,
        ),
        _panel_spec(
            xs_train, train_acc_pc, xs_train, train_acc_bp,
            _EPOCH_XLABEL, _TRAIN_ACC_YLABEL, *e_tr_a,
        ),
    ]
    return combined, test_loss, test_acc, train


def _write_metric_figures(
    save_dir,
    stem,
    combined_name,
    combined,
    test_loss,
    test_acc,
    train,
    plot_kw,
    *,
    tag="",
    test_prefix="",
    train_name="train_metrics",
):
    """Write the combined 2x2 plus the split test / train figures.

    ``tag`` is appended to the split filenames (``_mean_sem`` for seed
    aggregates). ``test_prefix`` distinguishes mini-epoch splits from the
    main-text epoch panels.
    """
    combined_path = _save_overlay_figure(
        (2, 2),
        _FIGSIZE_2x2,
        combined,
        os.path.join(save_dir, f"{stem}_{combined_name}.png"),
        plot_kw,
    )
    test_loss_path = _save_overlay_figure(
        (1, 1),
        ps.PANEL_THIRD,
        test_loss,
        os.path.join(save_dir, f"{stem}_{test_prefix}test_loss{tag}.png"),
        plot_kw,
    )
    test_acc_path = _save_overlay_figure(
        (1, 1),
        ps.PANEL_THIRD,
        test_acc,
        os.path.join(save_dir, f"{stem}_{test_prefix}test_accuracy{tag}.png"),
        plot_kw,
    )
    train_path = _save_overlay_figure(
        (1, 2),
        _FIGSIZE_1x2,
        train,
        os.path.join(save_dir, f"{stem}_{train_name}{tag}.png"),
        plot_kw,
    )
    return combined_path, test_loss_path, test_acc_path, train_path


def plot_metrics(
    history,
    save_dir,
    title_suffix="",
    log_steps=False,
    skip_pc=False,
    skip_bp=False,
):
    del title_suffix
    os.makedirs(save_dir, exist_ok=True)
    plot_kw = dict(skip_pc=skip_pc, skip_bp=skip_bp)
    stem = _metric_plot_stem(skip_pc, skip_bp)

    combined, test_loss, test_acc, train = _split_train_test_panels(
        xs_train=history["epoch_train"],
        xs_eval=history["epoch_eval"],
        train_loss_pc=history["pc_train_loss_epoch"],
        train_loss_bp=history["bp_train_loss_epoch"],
        test_loss_pc=history["pc_test_loss"],
        test_loss_bp=history["bp_test_loss"],
        train_acc_pc=history["pc_train_acc_epoch"],
        train_acc_bp=history["bp_train_acc_epoch"],
        test_acc_pc=history["pc_test_acc"],
        test_acc_bp=history["bp_test_acc"],
    )
    epoch_path, _, _, _ = _write_metric_figures(
        save_dir,
        stem,
        "epoch_metrics",
        combined,
        test_loss,
        test_acc,
        train,
        plot_kw,
    )

    combined, test_loss, test_acc, train = _split_train_test_panels(
        xs_train=history["mini_epoch"],
        xs_eval=history["mini_epoch"],
        train_loss_pc=history["pc_train_loss_mini"],
        train_loss_bp=history["bp_train_loss_mini"],
        test_loss_pc=history["pc_test_loss_mini"],
        test_loss_bp=history["bp_test_loss_mini"],
        train_acc_pc=history["pc_train_acc_mini"],
        train_acc_bp=history["bp_train_acc_mini"],
        test_acc_pc=history["pc_test_acc_mini"],
        test_acc_bp=history["bp_test_acc_mini"],
    )
    mini_path, _, _, _ = _write_metric_figures(
        save_dir,
        stem,
        "mini_epoch_metrics",
        combined,
        test_loss,
        test_acc,
        train,
        plot_kw,
        test_prefix="mini_epoch_",
        train_name="mini_epoch_train_metrics",
    )

    step_key = "pc_train_loss_step" if skip_bp else "bp_train_loss_step"
    steps = np.arange(len(history[step_key]))
    step_path = _save_overlay_figure(
        (1, 2),
        _FIGSIZE_1x2,
        [
            _panel_spec(
                steps,
                history["pc_train_loss_step"],
                steps,
                history["bp_train_loss_step"],
                _STEP_XLABEL,
                ps.LOSS_LABEL,
            ),
            _panel_spec(
                steps,
                history["pc_train_acc_step"],
                steps,
                history["bp_train_acc_step"],
                _STEP_XLABEL,
                _TRAIN_ACC_YLABEL,
            ),
        ],
        os.path.join(save_dir, f"{stem}_step_metrics.png"),
        plot_kw,
    )
    if log_steps:
        print(f"Saved plots to {epoch_path}, {mini_path}, and {step_path}")
    return epoch_path, mini_path, step_path


def _mean_sem_method_curves(histories, pc_key, bp_key, skip_pc, skip_bp):
    """Mean ± SEM for PC and/or BP curves, matching which method was trained."""
    if skip_bp:
        mean_pc, sem_pc, n_t = _aggregate_curves([h[pc_key] for h in histories])
        return mean_pc, sem_pc, mean_pc, None, n_t
    mean_bp, sem_bp, n_t = _aggregate_curves([h[bp_key] for h in histories])
    if skip_pc:
        return mean_bp, None, mean_bp, sem_bp, n_t
    mean_pc, sem_pc, n_t = _aggregate_curves([h[pc_key] for h in histories])
    return mean_pc, sem_pc, mean_bp, sem_bp, n_t


def plot_metrics_mean_sem(
    histories,
    save_dir,
    title_suffix="",
    log_steps=False,
    skip_pc=False,
    skip_bp=False,
):
    """Plot mean ± SEM across seeds (shaded bands)."""
    del title_suffix
    os.makedirs(save_dir, exist_ok=True)
    plot_kw = dict(skip_pc=skip_pc, skip_bp=skip_bp)
    stem = _metric_plot_stem(skip_pc, skip_bp)

    def series(pc_key, bp_key, xs):
        mean_pc, sem_pc, mean_bp, sem_bp, n_t = _mean_sem_method_curves(
            histories, pc_key, bp_key, skip_pc, skip_bp
        )
        xs_use = np.asarray(xs)[:n_t]
        return xs_use, mean_pc, mean_bp, sem_pc, sem_bp

    epoch_train = np.asarray(histories[0]["epoch_train"])
    epoch_eval = np.asarray(histories[0]["epoch_eval"])
    xs_tr, tr_l_pc, tr_l_bp, tr_l_pc_e, tr_l_bp_e = series(
        "pc_train_loss_epoch", "bp_train_loss_epoch", epoch_train
    )
    xs_te, te_l_pc, te_l_bp, te_l_pc_e, te_l_bp_e = series(
        "pc_test_loss", "bp_test_loss", epoch_eval
    )
    _, tr_a_pc, tr_a_bp, tr_a_pc_e, tr_a_bp_e = series(
        "pc_train_acc_epoch", "bp_train_acc_epoch", epoch_train
    )
    _, te_a_pc, te_a_bp, te_a_pc_e, te_a_bp_e = series(
        "pc_test_acc", "bp_test_acc", epoch_eval
    )
    combined, test_loss, test_acc, train = _split_train_test_panels(
        xs_train=xs_tr,
        xs_eval=xs_te,
        train_loss_pc=tr_l_pc,
        train_loss_bp=tr_l_bp,
        test_loss_pc=te_l_pc,
        test_loss_bp=te_l_bp,
        train_acc_pc=tr_a_pc,
        train_acc_bp=tr_a_bp,
        test_acc_pc=te_a_pc,
        test_acc_bp=te_a_bp,
        yerr=dict(
            train_loss_pc=tr_l_pc_e,
            train_loss_bp=tr_l_bp_e,
            test_loss_pc=te_l_pc_e,
            test_loss_bp=te_l_bp_e,
            train_acc_pc=tr_a_pc_e,
            train_acc_bp=tr_a_bp_e,
            test_acc_pc=te_a_pc_e,
            test_acc_bp=te_a_bp_e,
        ),
    )
    epoch_path, _, _, _ = _write_metric_figures(
        save_dir,
        stem,
        "epoch_metrics_mean_sem",
        combined,
        test_loss,
        test_acc,
        train,
        plot_kw,
        tag="_mean_sem",
    )

    mini_epoch = np.asarray(histories[0]["mini_epoch"])
    xs_m, m_tr_l_pc, m_tr_l_bp, m_tr_l_pc_e, m_tr_l_bp_e = series(
        "pc_train_loss_mini", "bp_train_loss_mini", mini_epoch
    )
    _, m_te_l_pc, m_te_l_bp, m_te_l_pc_e, m_te_l_bp_e = series(
        "pc_test_loss_mini", "bp_test_loss_mini", mini_epoch
    )
    _, m_tr_a_pc, m_tr_a_bp, m_tr_a_pc_e, m_tr_a_bp_e = series(
        "pc_train_acc_mini", "bp_train_acc_mini", mini_epoch
    )
    _, m_te_a_pc, m_te_a_bp, m_te_a_pc_e, m_te_a_bp_e = series(
        "pc_test_acc_mini", "bp_test_acc_mini", mini_epoch
    )
    combined, test_loss, test_acc, train = _split_train_test_panels(
        xs_train=xs_m,
        xs_eval=xs_m,
        train_loss_pc=m_tr_l_pc,
        train_loss_bp=m_tr_l_bp,
        test_loss_pc=m_te_l_pc,
        test_loss_bp=m_te_l_bp,
        train_acc_pc=m_tr_a_pc,
        train_acc_bp=m_tr_a_bp,
        test_acc_pc=m_te_a_pc,
        test_acc_bp=m_te_a_bp,
        yerr=dict(
            train_loss_pc=m_tr_l_pc_e,
            train_loss_bp=m_tr_l_bp_e,
            test_loss_pc=m_te_l_pc_e,
            test_loss_bp=m_te_l_bp_e,
            train_acc_pc=m_tr_a_pc_e,
            train_acc_bp=m_tr_a_bp_e,
            test_acc_pc=m_te_a_pc_e,
            test_acc_bp=m_te_a_bp_e,
        ),
    )
    mini_path, _, _, _ = _write_metric_figures(
        save_dir,
        stem,
        "mini_epoch_metrics_mean_sem",
        combined,
        test_loss,
        test_acc,
        train,
        plot_kw,
        tag="_mean_sem",
        test_prefix="mini_epoch_",
        train_name="mini_epoch_train_metrics",
    )

    mean_pc, sem_pc, mean_bp, sem_bp, n_t = _mean_sem_method_curves(
        histories, "pc_train_loss_step", "bp_train_loss_step", skip_pc, skip_bp
    )
    acc_pc, acc_pc_e, acc_bp, acc_bp_e, n_t_acc = _mean_sem_method_curves(
        histories, "pc_train_acc_step", "bp_train_acc_step", skip_pc, skip_bp
    )
    n_t = min(n_t, n_t_acc)
    steps = np.arange(n_t)

    def _trim(arr):
        return None if arr is None else arr[:n_t]

    step_path = _save_overlay_figure(
        (1, 2),
        _FIGSIZE_1x2,
        [
            _panel_spec(
                steps,
                _trim(mean_pc),
                steps,
                _trim(mean_bp),
                _STEP_XLABEL,
                ps.LOSS_LABEL,
                _trim(sem_pc),
                _trim(sem_bp),
            ),
            _panel_spec(
                steps,
                _trim(acc_pc),
                steps,
                _trim(acc_bp),
                _STEP_XLABEL,
                _TRAIN_ACC_YLABEL,
                _trim(acc_pc_e),
                _trim(acc_bp_e),
            ),
        ],
        os.path.join(save_dir, f"{stem}_step_metrics_mean_sem.png"),
        plot_kw,
    )
    if log_steps:
        print(
            f"Saved mean±SEM plots to {epoch_path}, {mini_path}, and {step_path}"
        )
    else:
        print(f"Saved mean±SEM plots to {save_dir}")
    return epoch_path, mini_path, step_path
    if log_steps:
        print(
            f"Saved mean±SEM plots to {epoch_path}, {mini_path}, and {step_path}"
        )
    else:
        print(f"Saved mean±SEM plots to {save_dir}")
    return epoch_path, mini_path, step_path


HISTORY_KEYS = (
    "epoch_eval",
    "epoch_train",
    "mini_epoch",
    "pc_train_loss_epoch",
    "bp_train_loss_epoch",
    "pc_train_acc_epoch",
    "bp_train_acc_epoch",
    "pc_test_loss",
    "bp_test_loss",
    "pc_test_acc",
    "bp_test_acc",
    "pc_train_loss_mini",
    "bp_train_loss_mini",
    "pc_train_acc_mini",
    "bp_train_acc_mini",
    "pc_test_loss_mini",
    "bp_test_loss_mini",
    "pc_test_acc_mini",
    "bp_test_acc_mini",
    "pc_train_loss_step",
    "bp_train_loss_step",
    "pc_train_acc_step",
    "bp_train_acc_step",
    "pc_energy_step",
)


def save_history(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for key, value in history.items():
        np.save(os.path.join(save_dir, f"{key}.npy"), np.asarray(value))


def load_history(save_dir):
    """Load metric history ``*.npy`` files written by ``save_history``."""
    if not os.path.isdir(save_dir):
        raise SystemExit(
            f"Run directory not found: {save_dir}. "
            "Run once with --keep_npy, then replot with --plot_from_npy "
            "using the same hyperparameters."
        )
    history = {}
    missing = []
    for key in HISTORY_KEYS:
        path = os.path.join(save_dir, f"{key}.npy")
        if not os.path.isfile(path):
            missing.append(path)
            continue
        history[key] = np.asarray(np.load(path)).tolist()
    if missing:
        listed = "\n  ".join(missing)
        raise SystemExit(
            f"Missing required .npy file(s) under {save_dir}:\n  {listed}\n"
            "Run once with --keep_npy, then replot with --plot_from_npy "
            "using the same hyperparameters."
        )
    return history


def cleanup_npy_files(save_dir):
    """Remove ``*.npy`` history dumps under ``save_dir`` (plots / args kept)."""
    removed = []
    root = Path(save_dir)
    if not root.is_dir():
        return removed
    for path in sorted(root.glob("*.npy")):
        path.unlink()
        removed.append(str(path))
    return removed


def _append_epoch_point(
    history,
    epoch,
    *,
    pc_train,
    bp_train,
    pc_test,
    bp_test,
    skip_pc,
    skip_bp=False,
):
    history["epoch_train"].append(epoch)
    history["epoch_eval"].append(epoch)
    if not skip_bp:
        history["bp_train_loss_epoch"].append(bp_train[0])
        history["bp_train_acc_epoch"].append(bp_train[1])
        history["bp_test_loss"].append(bp_test[0])
        history["bp_test_acc"].append(bp_test[1])
    if not skip_pc:
        history["pc_train_loss_epoch"].append(pc_train[0])
        history["pc_train_acc_epoch"].append(pc_train[1])
        history["pc_test_loss"].append(pc_test[0])
        history["pc_test_acc"].append(pc_test[1])


def _append_mini_point(
    history,
    frac,
    *,
    pc_train,
    bp_train,
    pc_test,
    bp_test,
    skip_pc,
    skip_bp=False,
):
    history["mini_epoch"].append(frac)
    if not skip_bp:
        history["bp_train_loss_mini"].append(bp_train[0])
        history["bp_train_acc_mini"].append(bp_train[1])
        history["bp_test_loss_mini"].append(bp_test[0])
        history["bp_test_acc_mini"].append(bp_test[1])
    if not skip_pc:
        history["pc_train_loss_mini"].append(pc_train[0])
        history["pc_train_acc_mini"].append(pc_train[1])
        history["pc_test_loss_mini"].append(pc_test[0])
        history["pc_test_acc_mini"].append(pc_test[1])


def _print_eval(
    label, *, skip_pc, pc_train, pc_test, bp_train, bp_test, skip_bp=False
):
    if skip_pc:
        print(
            f"  {label}: "
            f"BP train {bp_train[0]:.4f} ({bp_train[1]:.2f}%)  "
            f"test {bp_test[0]:.4f} ({bp_test[1]:.2f}%)"
        )
        return
    if skip_bp:
        print(
            f"  {label}: "
            f"PC train {pc_train[0]:.4f} ({pc_train[1]:.2f}%)  "
            f"test {pc_test[0]:.4f} ({pc_test[1]:.2f}%)"
        )
        return
    print(
        f"  {label}: "
        f"PC train {pc_train[0]:.4f} ({pc_train[1]:.2f}%)  "
        f"test {pc_test[0]:.4f} ({pc_test[1]:.2f}%)  |  "
        f"BP train {bp_train[0]:.4f} ({bp_train[1]:.2f}%)  "
        f"test {bp_test[0]:.4f} ({bp_test[1]:.2f}%)"
    )


def run_benchmark(args, save_dir=None):
    spec = DATASET_SPECS[args.dataset]
    if not hasattr(args, "skip_bp"):
        args.skip_bp = False
    if args.skip_pc and args.skip_bp:
        raise ValueError("Cannot skip both PC and BP")
    point_kw = dict(skip_pc=args.skip_pc, skip_bp=args.skip_bp)
    nan_metrics = (float("nan"), float("nan"))
    set_seed(args.seed)
    key = jr.PRNGKey(args.seed)

    if args.arch == "mlp":
        depth = args.n_hidden + 1
        fwd_depth = energy_depth = depth
        if args.use_skips:
            print("MLP skip connections enabled (Adam LR uses 1/√(N L)).")
    else:
        if args.n_res_blocks % 3 != 0:
            raise ValueError(
                f"--n_res_blocks must be a multiple of 3, got {args.n_res_blocks}"
            )
        fwd_depth, energy_depth = cnn_pc.resolve_cnn_ls(args)
        depth = energy_depth
        args.resolved_resnet_fwd_l = fwd_depth
        args.resolved_resnet_energy_l = energy_depth
        if args.use_skips:
            print(
                "Note: --use_skips is an MLP flag; CNN residual blocks are "
                "already in the architecture. Adam CNN LRs still use the "
                "res-block vs stage split from configure_cnn_param_optim."
            )
        if getattr(args, "additive_depth_factor", None) is not None:
            print(
                f"Note: --additive_depth_factor={args.additive_depth_factor} "
                f"overrides --resnet_energy_l; L_energy={energy_depth}."
            )

    if save_dir is None:
        save_dir = setup_save_dir(args)
    os.makedirs(save_dir, exist_ok=True)

    if getattr(args, "plot_from_npy", False):
        history = load_history(save_dir)
        plot_metrics(
            history,
            os.path.join(save_dir, "plots"),
            title_suffix=(
                f" ({args.dataset}, {args.arch}, N={args.width}, "
                + (
                    _cnn_depth_title(fwd_depth, energy_depth)
                    if args.arch == "cnn"
                    else f"L={depth}"
                )
                + ")"
            ),
            log_steps=args.log_steps,
            skip_pc=args.skip_pc,
            skip_bp=args.skip_bp,
        )
        print(f"Done. Results in {save_dir}")
        return save_dir, history

    output_energy_scaling = get_output_energy_scaling(
        args.param_type, args.gamma, args.width, energy_depth
    )
    hidden_kappa = get_hidden_energy_scaling(args.param_type, energy_depth)
    jpc_kw = pc_jpc_kwargs(args)

    prepare_data(args)

    pc_model, bp_model, skip_model = make_models(key, args, spec)
    if args.arch == "cnn":
        hidden_energy_scaling = cnn_pc.cnn_hidden_energy_scales(
            pc_model,
            hidden_kappa,
            getattr(args, "hidden_energy_layers", "weight"),
        )
    else:
        hidden_energy_scaling = hidden_kappa
    if not args.skip_pc:
        pc_param_optim, pc_opt_state = make_pc_param_optim(
            pc_model, skip_model, args, fwd_depth
        )
        activity_optim = (
            None
            if pc_closed_form(args)
            else optax.sgd(args.activity_lr * args.batch_size)
        )
    else:
        pc_param_optim = pc_opt_state = activity_optim = None
    if not args.skip_bp:
        bp_param_optim, bp_opt_state = make_bp_param_optim(
            bp_model, args, fwd_depth
        )
        bp_step = make_bp_step(args.loss_id)
    else:
        bp_param_optim = bp_opt_state = bp_step = None

    args_to_save = {
        key: value
        for key, value in vars(args).items()
        if not key.startswith("_")
    }
    with open(os.path.join(save_dir, "args.json"), "w", encoding="utf-8") as handle:
        json.dump(args_to_save, handle, indent=2, default=str)

    skip_notes = []
    if args.skip_pc:
        skip_notes.append("skip_pc")
    if args.skip_bp:
        skip_notes.append("skip_bp")
    skip_note = f", {', '.join(skip_notes)}" if skip_notes else ""
    print(
        f"Benchmark {args.dataset} ({args.arch}), width={args.width}, "
        f"{_cnn_depth_title(fwd_depth, energy_depth) if args.arch == 'cnn' else f'L={depth}'}, "
        f"γ={args.gamma}, λ={output_energy_scaling}, "
        f"κ={hidden_kappa}, "
        + (
            f"hidden_energy_layers={getattr(args, 'hidden_energy_layers', 'weight')}, "
            if args.arch == "cnn"
            else ""
        )
        + f"optim={args.param_optim}, "
        f"lr_bp={args.param_lr}, lr_pc={args.param_lr_pc}, "
        f"pc_infer={getattr(args, 'pc_infer_mode', 'infer')}{skip_note}"
    )
    if uses_fixed_subset(args):
        print(
            f"Fixed subset: {args.batch_size} frozen unaugmented train "
            f"examples and {args.batch_size} held-out test examples "
            "(one GD step per epoch; metrics use these subsets)"
        )

    history = {
        "epoch_eval": [],
        "epoch_train": [],
        "mini_epoch": [],
        "pc_train_loss_epoch": [],
        "bp_train_loss_epoch": [],
        "pc_train_acc_epoch": [],
        "bp_train_acc_epoch": [],
        "pc_test_loss": [],
        "bp_test_loss": [],
        "pc_test_acc": [],
        "bp_test_acc": [],
        "pc_train_loss_mini": [],
        "bp_train_loss_mini": [],
        "pc_train_acc_mini": [],
        "bp_train_acc_mini": [],
        "pc_test_loss_mini": [],
        "bp_test_loss_mini": [],
        "pc_test_acc_mini": [],
        "bp_test_acc_mini": [],
        "pc_train_loss_step": [],
        "bp_train_loss_step": [],
        "pc_train_acc_step": [],
        "bp_train_acc_step": [],
        "pc_energy_step": [],
    }

    print("Evaluating at initialization...")
    if args.skip_pc:
        _, bp_train = evaluate_train(bp_model, args)
        bp_test = evaluate_bp(bp_model, args)
        pc_train = pc_test = nan_metrics
    elif args.skip_bp:
        pc_train, _ = evaluate_train(
            None,
            args,
            pc_model=pc_model,
            skip_model=skip_model,
            jpc_kw=jpc_kw,
        )
        pc_test = evaluate_pc(pc_model, args, skip_model, jpc_kw)
        bp_train = bp_test = nan_metrics
    else:
        pc_train, bp_train = evaluate_train(
            bp_model,
            args,
            pc_model=pc_model,
            skip_model=skip_model,
            jpc_kw=jpc_kw,
        )
        pc_test = evaluate_pc(pc_model, args, skip_model, jpc_kw)
        bp_test = evaluate_bp(bp_model, args)
    _append_epoch_point(
        history,
        0,
        pc_train=pc_train,
        bp_train=bp_train,
        pc_test=pc_test,
        bp_test=bp_test,
        **point_kw,
    )
    _append_mini_point(
        history,
        0.0,
        pc_train=pc_train,
        bp_train=bp_train,
        pc_test=pc_test,
        bp_test=bp_test,
        **point_kw,
    )
    _print_eval(
        "init",
        pc_train=pc_train,
        pc_test=pc_test,
        bp_train=bp_train,
        bp_test=bp_test,
        **point_kw,
    )
    if (
        not args.skip_pc
        and not args.skip_bp
        and abs(pc_test[0] - bp_test[0]) > 1e-3
    ):
        print(
            "  Warning: PC and BP test losses differ at init; check weight copy."
        )

    n_expected = n_train_batches(args)
    boundaries = mini_epoch_boundaries(n_expected, args.n_mini_per_epoch)

    global_step = 0
    for epoch in range(1, args.n_epochs + 1):
        pc_loss_sum = bp_loss_sum = 0.0
        pc_acc_sum = bp_acc_sum = 0.0
        mini_pc_loss_sum = mini_bp_loss_sum = 0.0
        mini_pc_acc_sum = mini_bp_acc_sum = 0.0
        n_batches = 0
        mini_n = 0

        for x, y in iter_train_batches(args, epoch):
            if not args.skip_bp:
                bp_loss, bp_acc = bp_batch_metrics(bp_model, x, y, args.loss_id)
            else:
                bp_loss = bp_acc = float("nan")

            if not args.skip_pc:
                pc_loss, pc_acc = pc_batch_metrics(
                    pc_model, x, y, skip_model, jpc_kw, args.loss_id
                )
                pc_model, skip_model, pc_opt_state, energy, pc_ok = (
                    pc_infer_and_update(
                        pc_model,
                        skip_model,
                        x,
                        y,
                        activity_optim,
                        pc_param_optim,
                        pc_opt_state,
                        args,
                        jpc_kw,
                        output_energy_scaling,
                        hidden_energy_scaling,
                    )
                )
                if not pc_ok:
                    print(
                        f"  Warning: non-finite PC energy at epoch {epoch} "
                        f"step {global_step}; skipped PC parameter update."
                    )
                history["pc_train_loss_step"].append(pc_loss)
                history["pc_train_acc_step"].append(pc_acc)
                history["pc_energy_step"].append(energy)
                pc_loss_sum += pc_loss
                pc_acc_sum += pc_acc
                mini_pc_loss_sum += pc_loss
                mini_pc_acc_sum += pc_acc
            else:
                energy = float("nan")

            if not args.skip_bp:
                bp_model, bp_opt_state = bp_step(
                    bp_model, bp_opt_state, bp_param_optim, x, y
                )
                history["bp_train_loss_step"].append(bp_loss)
                history["bp_train_acc_step"].append(bp_acc)
                bp_loss_sum += bp_loss
                bp_acc_sum += bp_acc
                mini_bp_loss_sum += bp_loss
                mini_bp_acc_sum += bp_acc

            n_batches += 1
            mini_n += 1
            global_step += 1

            if args.log_steps and global_step % args.log_every == 0:
                if args.skip_pc:
                    print(
                        f"  epoch {epoch} step {global_step}: "
                        f"BP loss={bp_loss:.4f} acc={bp_acc:.2f}%"
                    )
                elif args.skip_bp:
                    print(
                        f"  epoch {epoch} step {global_step}: "
                        f"PC loss={pc_loss:.4f} acc={pc_acc:.2f}%  |  "
                        f"energy={energy:.4f}"
                    )
                else:
                    print(
                        f"  epoch {epoch} step {global_step}: "
                        f"PC loss={pc_loss:.4f} acc={pc_acc:.2f}%  |  "
                        f"BP loss={bp_loss:.4f} acc={bp_acc:.2f}%  |  "
                        f"energy={energy:.4f}"
                    )

            if n_batches in boundaries and mini_n > 0:
                frac = (epoch - 1) + n_batches / n_expected
                at_epoch_end = n_batches == n_expected
                bp_train_mini = (
                    (mini_bp_loss_sum / mini_n, mini_bp_acc_sum / mini_n)
                    if not args.skip_bp
                    else nan_metrics
                )
                pc_train_mini = (
                    (mini_pc_loss_sum / mini_n, mini_pc_acc_sum / mini_n)
                    if not args.skip_pc
                    else nan_metrics
                )
                if at_epoch_end:
                    print(f"Evaluating after epoch {epoch}...")
                else:
                    print(f"Evaluating after mini-epoch {frac:.1f}...")
                bp_test = (
                    nan_metrics if args.skip_bp else evaluate_bp(bp_model, args)
                )
                pc_test = (
                    nan_metrics
                    if args.skip_pc
                    else evaluate_pc(pc_model, args, skip_model, jpc_kw)
                )
                if uses_fixed_subset(args) and at_epoch_end:
                    pc_train_mini, bp_train_mini = current_train_metrics(
                        args,
                        bp_model=bp_model,
                        pc_model=pc_model,
                        skip_model=skip_model,
                        jpc_kw=jpc_kw,
                        skip_pc=args.skip_pc,
                        skip_bp=args.skip_bp,
                        nan_metrics=nan_metrics,
                    )
                _append_mini_point(
                    history,
                    frac,
                    pc_train=pc_train_mini,
                    bp_train=bp_train_mini,
                    pc_test=pc_test,
                    bp_test=bp_test,
                    **point_kw,
                )
                if at_epoch_end:
                    bp_train_epoch = bp_train_mini
                    pc_train_epoch = pc_train_mini
                    if not uses_fixed_subset(args):
                        bp_train_epoch = (
                            (bp_loss_sum / n_batches, bp_acc_sum / n_batches)
                            if not args.skip_bp
                            else nan_metrics
                        )
                        pc_train_epoch = (
                            (pc_loss_sum / n_batches, pc_acc_sum / n_batches)
                            if not args.skip_pc
                            else nan_metrics
                        )
                    _append_epoch_point(
                        history,
                        epoch,
                        pc_train=pc_train_epoch,
                        bp_train=bp_train_epoch,
                        pc_test=pc_test,
                        bp_test=bp_test,
                        **point_kw,
                    )
                    _print_eval(
                        f"epoch {epoch}",
                        pc_train=pc_train_epoch,
                        pc_test=pc_test,
                        bp_train=bp_train_epoch,
                        bp_test=bp_test,
                        **point_kw,
                    )
                else:
                    _print_eval(
                        f"mini-epoch {frac:.1f}",
                        pc_train=pc_train_mini,
                        pc_test=pc_test,
                        bp_train=bp_train_mini,
                        bp_test=bp_test,
                        **point_kw,
                    )
                mini_pc_loss_sum = mini_bp_loss_sum = 0.0
                mini_pc_acc_sum = mini_bp_acc_sum = 0.0
                mini_n = 0

        if n_batches == 0:
            raise RuntimeError(
                f"No training batches in epoch {epoch}. Check the dataset path "
                "or Hugging Face token for ImageNet."
            )

        if history["epoch_eval"][-1] != epoch:
            bp_train_epoch = (
                (bp_loss_sum / n_batches, bp_acc_sum / n_batches)
                if not args.skip_bp
                else nan_metrics
            )
            pc_train_epoch = (
                (pc_loss_sum / n_batches, pc_acc_sum / n_batches)
                if not args.skip_pc
                else nan_metrics
            )
            if mini_n > 0:
                frac = (epoch - 1) + n_batches / max(n_expected, n_batches)
                bp_train_mini = (
                    (mini_bp_loss_sum / mini_n, mini_bp_acc_sum / mini_n)
                    if not args.skip_bp
                    else nan_metrics
                )
                pc_train_mini = (
                    (mini_pc_loss_sum / mini_n, mini_pc_acc_sum / mini_n)
                    if not args.skip_pc
                    else nan_metrics
                )
            else:
                frac = float(epoch)
                bp_train_mini = bp_train_epoch
                pc_train_mini = pc_train_epoch
            print(f"Evaluating after epoch {epoch}...")
            bp_test = nan_metrics if args.skip_bp else evaluate_bp(bp_model, args)
            pc_test = (
                nan_metrics
                if args.skip_pc
                else evaluate_pc(pc_model, args, skip_model, jpc_kw)
            )
            if uses_fixed_subset(args):
                pc_train_epoch, bp_train_epoch = current_train_metrics(
                    args,
                    bp_model=bp_model,
                    pc_model=pc_model,
                    skip_model=skip_model,
                    jpc_kw=jpc_kw,
                    skip_pc=args.skip_pc,
                    skip_bp=args.skip_bp,
                    nan_metrics=nan_metrics,
                )
                pc_train_mini, bp_train_mini = pc_train_epoch, bp_train_epoch
            _append_mini_point(
                history,
                frac,
                pc_train=pc_train_mini,
                bp_train=bp_train_mini,
                pc_test=pc_test,
                bp_test=bp_test,
                **point_kw,
            )
            _append_epoch_point(
                history,
                epoch,
                pc_train=pc_train_epoch,
                bp_train=bp_train_epoch,
                pc_test=pc_test,
                bp_test=bp_test,
                **point_kw,
            )
            _print_eval(
                f"epoch {epoch}",
                pc_train=pc_train_epoch,
                pc_test=pc_test,
                bp_train=bp_train_epoch,
                bp_test=bp_test,
                **point_kw,
            )

        save_history(history, save_dir)
        plot_metrics(
            history,
            os.path.join(save_dir, "plots"),
            title_suffix=(
                f" ({args.dataset}, {args.arch}, N={args.width}, "
                + (
                    _cnn_depth_title(fwd_depth, energy_depth)
                    if args.arch == "cnn"
                    else f"L={depth}"
                )
                + ")"
            ),
            log_steps=args.log_steps,
            skip_pc=args.skip_pc,
            skip_bp=args.skip_bp,
        )

    if not args.keep_npy:
        removed = cleanup_npy_files(save_dir)
        if removed:
            print(f"Removed {len(removed)} .npy file(s) under {save_dir}")
        else:
            print(f"No .npy files to remove under {save_dir}")

    print(f"Done. Results in {save_dir}")
    return save_dir, history


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "PC vs backprop finite-size dataset benchmark. "
            "Pass a list for any sweepable hyperparameter to run a "
            "Cartesian search (BP and PC independently)."
        ),
    )
    parser.add_argument("--results_dir", type=str, default="results_benchmark")
    parser.add_argument(
        "--dataset",
        type=str,
        default="MNIST",
        help="MNIST, Fashion-MNIST, CIFAR10, tiny-CIFAR10, TinyImageNet, or ImageNet.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        choices=["mlp", "cnn"],
        help=(
            "Architecture. Default: mlp for MNIST / Fashion-MNIST / "
            "tiny-CIFAR10, cnn otherwise."
        ),
    )

    parser.add_argument(
        "--width",
        type=int,
        nargs="+",
        default=[128],
        help="Hidden width. Pass multiple values to sweep.",
    )
    parser.add_argument(
        "--n_hidden",
        type=int,
        nargs="+",
        default=[3],
        help="MLP hidden layers. Pass multiple values to sweep (MLP only).",
    )
    parser.add_argument(
        "--n_res_blocks",
        type=int,
        nargs="+",
        default=[3],
        help=(
            "CNN residual blocks (multiple of 3). "
            "Pass multiple values to sweep (CNN only)."
        ),
    )
    parser.add_argument(
        "--param_type", type=str, default="mupc", choices=["sp", "mupc"]
    )
    parser.add_argument("--act_fn", type=str, default="relu")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--use_skips", action="store_true", default=False)
    parser.add_argument(
        "--scale_non_res_layers", action="store_true", default=False
    )
    parser.add_argument(
        "--resnet_fwd_l",
        type=cnn_pc.parse_resnet_l_arg,
        default="n_res_blocks",
        help=(
            "CNN only: L used for residual 1/√L and Adam residual LRs. "
            "n_res_blocks (default), n_weight_layers (R+4), n_modules "
            "(R+7, includes pools), or a positive integer. Ignored for MLP."
        ),
    )
    parser.add_argument(
        "--resnet_energy_l",
        type=cnn_pc.parse_resnet_l_arg,
        default="n_weight_layers",
        help=(
            "CNN only: L used for λ = γ² N L and κ = L. "
            "n_weight_layers (default, R+4), n_res_blocks, n_modules, "
            "or a positive integer. Ignored for MLP. Overridden by "
            "--additive_depth_factor when that flag is set."
        ),
    )
    parser.add_argument(
        "--hidden_energy_layers",
        type=str,
        default="weight",
        choices=list(cnn_pc.HIDDEN_ENERGY_LAYER_CHOICES),
        help=(
            "CNN only: which hidden modules get κ. "
            "weight (default): stems + residual blocks; pools unscaled. "
            "all: every hidden module including pools. "
            "residual: ResNetBlocks only. Ignored for MLP."
        ),
    )
    parser.add_argument(
        "--additive_depth_factor",
        type=int,
        default=None,
        help=(
            "CNN only: if set, overrides --resnet_energy_l with "
            "n_res_blocks + this factor. Does not affect L_fwd. "
            "Default: unset (use --resnet_energy_l)."
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="+",
        default=[64],
        help=(
            "Minibatch size. Pass multiple values to sweep. "
            "With --fixed_subset this is both the frozen train-set size "
            "and the held-out test subset size."
        ),
    )
    parser.add_argument(
        "--fixed_subset",
        action="store_true",
        default=False,
        help=(
            "Train on one frozen unaugmented subset of --batch_size "
            "examples (sampled with --seed). Every epoch is one GD step "
            "on that same batch. Train metrics use this subset; test "
            "metrics use a held-out subset of the same size from the "
            "official test/val split (alignment protocol)."
        ),
    )
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument(
        "--n_mini_per_epoch",
        type=int,
        default=10,
        help=(
            "Number of mini-epoch checkpoints per epoch (train window "
            "average + full test eval). Default 10, i.e. 10× epoch frequency."
        ),
    )
    parser.add_argument(
        "--param_optim",
        type=str,
        default="adam",
        choices=["gd", "adam", "sgd_momentum"],
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for --param_optim sgd_momentum (ignored otherwise).",
    )
    parser.add_argument(
        "--param_lr",
        type=float,
        nargs="+",
        default=[1e-3],
        help=(
            "Backprop parameter learning rate. "
            "Pass multiple values to sweep BP only."
        ),
    )
    parser.add_argument(
        "--param_lr_pc",
        type=float,
        nargs="+",
        default=[1e-3],
        help=(
            "PC parameter learning rate (Adam: divided like BP; "
            "GD / SGD+momentum: used as-is). "
            "Pass multiple values to sweep PC only."
        ),
    )
    parser.add_argument("--loss_id", type=str, default="ce", choices=["mse", "ce"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=1,
        help=(
            "Number of consecutive seeds starting at --seed. "
            "Each seed gets its own run directory; with n_seeds>1 also "
            "writes mean±SEM overlay plots under a seeds_* directory."
        ),
    )
    parser.add_argument("--log_every", type=int, default=100)

    parser.add_argument(
        "--pc_infer_mode",
        type=str,
        default="infer",
        choices=["infer", "optim", "closed_form"],
        help=(
            "PC inference mode. 'infer' / 'optim' (default) run "
            "--n_infer_iters steps of activity GD each training step. "
            "'closed_form' updates PC parameters from the linear "
            "equilibrium energy (requires --act_fn linear, --loss_id mse, "
            "and --arch mlp). Activity LR / n_infer_iters are unused."
        ),
    )
    parser.add_argument(
        "--activity_lr",
        type=float,
        nargs="+",
        default=[0.3],
        help=(
            "PC activity learning rate. Pass multiple values to sweep PC only. "
            "Unused with --pc_infer_mode closed_form."
        ),
    )
    parser.add_argument(
        "--n_infer_iters",
        type=int,
        nargs="+",
        default=[10],
        help=(
            "PC inference steps. Pass multiple values to sweep PC only. "
            "Unused with --pc_infer_mode closed_form."
        ),
    )
    parser.add_argument(
        "--keep_npy",
        action="store_true",
        default=False,
        help=(
            "Keep history *.npy files under the run directory. "
            "By default they are deleted after plots are written. "
            "Required for a later --plot_from_npy run."
        ),
    )
    parser.add_argument(
        "--plot_from_npy",
        action="store_true",
        default=False,
        help=(
            "Skip training; rebuild figures from saved history *.npy "
            "files (same hyperparameters as the original --keep_npy "
            "run). Does not delete .npy files."
        ),
    )
    parser.add_argument(
        "--log_steps",
        action="store_true",
        default=False,
        help=(
            "Print per-step train metrics (every --log_every steps) and "
            "plot save paths."
        ),
    )
    parser.add_argument(
        "--skip_pc",
        action="store_true",
        default=False,
        help=(
            "Skip PC training/eval and plot backprop only. "
            "In a sweep, skip the PC grid. "
            "Default: train and plot both PC and BP."
        ),
    )
    parser.add_argument(
        "--skip_bp",
        action="store_true",
        default=False,
        help=(
            "Skip backprop training/eval and plot PC only. "
            "In a sweep, skip the BP grid."
        ),
    )
    return parser.parse_args()


SWEEP_ALL_KEYS = (
    "n_hidden",
    "width",
    "batch_size",
    "n_res_blocks",
    "param_lr",
    "param_lr_pc",
    "activity_lr",
    "n_infer_iters",
)
SWEEP_BP_ONLY = ("param_lr",)
SWEEP_PC_ONLY = ("param_lr_pc", "activity_lr", "n_infer_iters")
SWEEP_PC_CLOSED_FORM = ("param_lr_pc",)


def pc_only_sweep_keys(args):
    if pc_closed_form(args):
        return SWEEP_PC_CLOSED_FORM
    return SWEEP_PC_ONLY


def _as_values(args, key):
    val = getattr(args, key)
    if isinstance(val, (list, tuple)):
        return list(val)
    return [val]


def _scalarize_sweep_fields(args):
    """Replace list-valued sweep fields with their first (or only) element."""
    for key in SWEEP_ALL_KEYS:
        val = getattr(args, key)
        if isinstance(val, (list, tuple)):
            if len(val) < 1:
                raise ValueError(f"--{key} must contain at least one value")
            setattr(args, key, val[0])
    return args


def shared_sweep_keys(arch):
    keys = ["width"]
    if arch == "mlp":
        keys.append("n_hidden")
    else:
        keys.append("n_res_blocks")
    keys.append("batch_size")
    return keys


def active_sweep_keys(args):
    return (
        shared_sweep_keys(args.arch)
        + list(SWEEP_BP_ONLY)
        + list(pc_only_sweep_keys(args))
    )


def is_hp_sweep(args):
    return any(len(_as_values(args, key)) > 1 for key in active_sweep_keys(args))


def cartesian_grid(args, keys):
    if not keys:
        return [{}]
    axes = [_as_values(args, key) for key in keys]
    return [dict(zip(keys, combo)) for combo in product(*axes)]


def _public_args_dict(args):
    return {
        key: value
        for key, value in vars(args).items()
        if not key.startswith("_")
    }


def _json_number(value):
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return value


def _energy_depth(args):
    if args.arch == "mlp":
        return args.n_hidden + 1
    _, energy_l = cnn_pc.resolve_cnn_ls(args)
    return energy_l


def _fwd_depth(args):
    if args.arch == "mlp":
        return args.n_hidden + 1
    fwd_l, _ = cnn_pc.resolve_cnn_ls(args)
    return fwd_l


def make_run_args(base_args, hparams=None, **overrides):
    run_args = argparse.Namespace(**vars(base_args))
    if hparams:
        for key, value in hparams.items():
            setattr(run_args, key, value)
    _scalarize_sweep_fields(run_args)
    for key, value in overrides.items():
        setattr(run_args, key, value)
    run_args._train_loader = None
    run_args._test_loader = None
    run_args._fixed_train_batch = None
    run_args._fixed_test_batch = None
    return run_args


def _hp_value_str(value):
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def sweep_config_dir(sweep_dir, method, hparams):
    """``hp_sweep/{bp|pc}/width=256/n_hidden=3/param_lr=0.02``."""
    parts = [sweep_dir, str(method).lower()]
    for key, value in hparams.items():
        parts.append(f"{key}={_hp_value_str(value)}")
    return os.path.join(*parts)


def plot_seed_aggregate(args, histories, save_dir=None):
    """Mean±SEM plots across seeds for a scalar hyperparameter config."""
    if args.n_seeds <= 1:
        return None
    if save_dir is None:
        seed_tag = f"seeds_{args.seed}_{args.seed + args.n_seeds - 1}"
        save_dir = setup_save_dir(args, seed_tag=seed_tag)
    depth = _energy_depth(args)
    fwd_depth = _fwd_depth(args)
    plot_metrics_mean_sem(
        histories,
        os.path.join(save_dir, "plots"),
        title_suffix=(
            f" ({args.dataset}, {args.arch}, N={args.width}, "
            + (
                _cnn_depth_title(fwd_depth, depth)
                if args.arch == "cnn"
                else f"L={depth}"
            )
            + ")"
        ),
        log_steps=args.log_steps,
        skip_pc=args.skip_pc,
        skip_bp=getattr(args, "skip_bp", False),
    )
    with open(os.path.join(save_dir, "args.json"), "w", encoding="utf-8") as handle:
        json.dump(_public_args_dict(args), handle, indent=2, default=str)
    print(f"Aggregated mean±SEM plots in {save_dir}")
    return save_dir


def _history_last(history, key):
    values = history.get(key, [])
    if not values:
        return float("nan")
    return float(values[-1])


def final_metrics_from_history(history, *, skip_pc, skip_bp):
    prefix = "pc" if skip_bp else "bp"
    return {
        "test_acc": _history_last(history, f"{prefix}_test_acc"),
        "test_loss": _history_last(history, f"{prefix}_test_loss"),
        "train_acc": _history_last(history, f"{prefix}_train_acc_epoch"),
        "train_loss": _history_last(history, f"{prefix}_train_loss_epoch"),
    }


def _mean_sem(values):
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), float("nan")
    mean = float(finite.mean())
    if finite.size == 1:
        return mean, 0.0
    return mean, float(finite.std(ddof=1) / np.sqrt(finite.size))


def aggregate_seed_records(records):
    summary = {}
    for key in (
        "test_acc",
        "test_loss",
        "train_acc",
        "train_loss",
        "wall_time_s",
    ):
        mean, sem = _mean_sem([record[key] for record in records])
        summary[f"mean_{key}"] = mean
        summary[f"sem_{key}"] = sem
    summary["n_seeds"] = len(records)
    summary["seeds"] = records
    return summary


def rank_by_mean_test_acc(results):
    def score(result):
        value = result.get("mean_test_acc", float("nan"))
        return value if np.isfinite(value) else float("-inf")

    return sorted(results, key=score, reverse=True)


def _format_hp(hparams):
    parts = []
    for key, value in hparams.items():
        parts.append(f"{key}={_hp_value_str(value)}")
    return ", ".join(parts)


def _print_ranked_table(title, ranked):
    print(f"\n=== {title} ===")
    if not ranked:
        print("  (no configs)")
        return
    header = (
        f"{'rank':>4}  {'test_acc':>10}  {'train_acc':>10}  "
        f"{'test_loss':>10}  {'train_loss':>10}  {'wall_s':>8}  hyperparams"
    )
    print(header)
    for rank, result in enumerate(ranked, start=1):
        print(
            f"{rank:4d}  "
            f"{result['mean_test_acc']:10.2f}  "
            f"{result['mean_train_acc']:10.2f}  "
            f"{result['mean_test_loss']:10.4f}  "
            f"{result['mean_train_loss']:10.4f}  "
            f"{result['mean_wall_time_s']:8.1f}  "
            f"{_format_hp(result['hyperparams'])}"
        )
    best = ranked[0]
    print(
        f"Best: {_format_hp(best['hyperparams'])}  "
        f"(mean test acc {best['mean_test_acc']:.2f}% ± {best['sem_test_acc']:.2f})"
    )


def run_config_with_seeds(
    base_args, hparams, *, skip_pc, skip_bp, method_label, sweep_dir
):
    records = []
    histories = []
    config_dir = sweep_config_dir(sweep_dir, method_label, hparams)
    seeds = range(base_args.seed, base_args.seed + base_args.n_seeds)
    for seed in seeds:
        run_args = make_run_args(
            base_args, hparams, skip_pc=skip_pc, skip_bp=skip_bp, seed=seed
        )
        seed_dir = os.path.join(config_dir, f"seed={seed}")
        print(f"\n=== {method_label} seed={seed} | {_format_hp(hparams)} ===")
        t0 = time.perf_counter()
        save_dir, history = run_benchmark(run_args, save_dir=seed_dir)
        wall_time_s = time.perf_counter() - t0
        metrics = final_metrics_from_history(
            history, skip_pc=skip_pc, skip_bp=skip_bp
        )
        record = {
            "seed": int(seed),
            "save_dir": save_dir,
            "wall_time_s": float(wall_time_s),
            "test_acc": _json_number(metrics["test_acc"]),
            "test_loss": _json_number(metrics["test_loss"]),
            "train_acc": _json_number(metrics["train_acc"]),
            "train_loss": _json_number(metrics["train_loss"]),
        }
        records.append(record)
        histories.append(history)
        print(
            f"  finished in {wall_time_s:.1f}s  "
            f"test acc={metrics['test_acc']:.2f}%  "
            f"train acc={metrics['train_acc']:.2f}%"
        )

    if base_args.n_seeds > 1:
        agg_args = make_run_args(
            base_args,
            hparams,
            skip_pc=skip_pc,
            skip_bp=skip_bp,
            seed=base_args.seed,
        )
        plot_seed_aggregate(
            agg_args, histories, save_dir=os.path.join(config_dir, "mean_sem")
        )

    summary = aggregate_seed_records(records)
    summary["hyperparams"] = {
        key: _json_number(value) for key, value in hparams.items()
    }
    summary["method"] = method_label
    summary["dir"] = os.path.relpath(config_dir, sweep_dir)
    return summary


def _grid_values(args, keys):
    return {key: [_json_number(v) for v in _as_values(args, key)] for key in keys}


def _fixed_training_args(args):
    """Non-swept training settings (omit CLI bookkeeping flags)."""
    keys = [
        "act_fn",
        "param_type",
        "gamma",
        "param_optim",
        "pc_infer_mode",
        "use_skips",
        "fixed_subset",
        "n_mini_per_epoch",
    ]
    if args.param_optim == "sgd_momentum":
        keys.append("momentum")
    if args.arch == "cnn":
        keys.extend(
            [
                "scale_non_res_layers",
                "resnet_fwd_l",
                "resnet_energy_l",
                "hidden_energy_layers",
            ]
        )
        if getattr(args, "additive_depth_factor", None) is not None:
            keys.append("additive_depth_factor")
    fixed = {}
    for key in keys:
        value = getattr(args, key)
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                continue
            value = value[0]
        if key in ("use_skips", "scale_non_res_layers", "fixed_subset") and not value:
            continue
        fixed[key] = _json_number(value)
    return fixed


def _compact_seed(record):
    return {
        "seed": record["seed"],
        "test_acc": record["test_acc"],
        "train_acc": record["train_acc"],
        "test_loss": record["test_loss"],
        "train_loss": record["train_loss"],
        "wall_time_s": round(record["wall_time_s"], 2),
    }


def _round_metric(value, digits=4):
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def _compact_result(result, rank, n_seeds):
    """Flatten hyperparams + metrics into one object for the sweep JSON."""
    row = {"rank": rank}
    row.update(result["hyperparams"])
    row["test_acc"] = _round_metric(result["mean_test_acc"], 2)
    row["train_acc"] = _round_metric(result["mean_train_acc"], 2)
    row["test_loss"] = _round_metric(result["mean_test_loss"])
    row["train_loss"] = _round_metric(result["mean_train_loss"])
    row["wall_time_s"] = _round_metric(result["mean_wall_time_s"], 2)
    if n_seeds > 1:
        row["test_acc_sem"] = _round_metric(result["sem_test_acc"], 2)
        row["train_acc_sem"] = _round_metric(result["sem_train_acc"], 2)
        row["test_loss_sem"] = _round_metric(result["sem_test_loss"])
        row["train_loss_sem"] = _round_metric(result["sem_train_loss"])
        row["wall_time_s_sem"] = _round_metric(result["sem_wall_time_s"], 2)
    row["dir"] = result["dir"]
    if n_seeds > 1:
        row["seeds"] = [_compact_seed(record) for record in result["seeds"]]
    return row


def _compact_best(result, n_seeds):
    if result is None:
        return None
    best = _compact_result(result, rank=1, n_seeds=n_seeds)
    best.pop("rank", None)
    best.pop("seeds", None)
    return best


def warn_unused_arch_sweep_axes(args):
    if args.arch == "mlp" and len(_as_values(args, "n_res_blocks")) > 1:
        print(
            "Warning: --n_res_blocks is unused for MLP; extra values are ignored."
        )
    if args.arch == "cnn" and len(_as_values(args, "n_hidden")) > 1:
        print(
            "Warning: --n_hidden is unused for CNN; extra values are ignored."
        )
    if args.arch == "mlp":
        if getattr(args, "resnet_fwd_l", "n_res_blocks") != "n_res_blocks":
            print("Warning: --resnet_fwd_l is unused for MLP.")
        if getattr(args, "resnet_energy_l", "n_weight_layers") != "n_weight_layers":
            print("Warning: --resnet_energy_l is unused for MLP.")
        if getattr(args, "hidden_energy_layers", "weight") != "weight":
            print("Warning: --hidden_energy_layers is unused for MLP.")
        if getattr(args, "additive_depth_factor", None) is not None:
            print("Warning: --additive_depth_factor is unused for MLP.")
    if pc_closed_form(args):
        for key in ("activity_lr", "n_infer_iters"):
            if len(_as_values(args, key)) > 1:
                print(
                    f"Warning: --{key} is unused with --pc_infer_mode "
                    "closed_form; extra values are ignored."
                )


def run_hp_sweep(args):
    shared_keys = shared_sweep_keys(args.arch)
    bp_keys = shared_keys + list(SWEEP_BP_ONLY)
    pc_keys = shared_keys + list(pc_only_sweep_keys(args))
    do_bp = not args.skip_bp
    do_pc = not args.skip_pc
    bp_grid = cartesian_grid(args, bp_keys) if do_bp else []
    pc_grid = cartesian_grid(args, pc_keys) if do_pc else []

    print(
        f"Hyperparameter sweep on {args.dataset} ({args.arch}): "
        f"{len(bp_grid)} BP configs × {args.n_seeds} seed(s), "
        f"{len(pc_grid)} PC configs × {args.n_seeds} seed(s)."
    )
    print(
        "Selection metric: mean final test accuracy. "
        "BP grid keys: "
        + ", ".join(f"{k}={_as_values(args, k)}" for k in bp_keys)
    )
    print(
        "PC grid keys: "
        + ", ".join(f"{k}={_as_values(args, k)}" for k in pc_keys)
    )

    sweep_dir = os.path.join(
        args.results_dir, args.dataset, args.arch, args.loss_id, "hp_sweep"
    )
    os.makedirs(sweep_dir, exist_ok=True)

    t_sweep = time.perf_counter()
    bp_results = []
    for i, hparams in enumerate(bp_grid, start=1):
        print(f"\n----- BP config {i}/{len(bp_grid)}: {_format_hp(hparams)} -----")
        bp_results.append(
            run_config_with_seeds(
                args,
                hparams,
                skip_pc=True,
                skip_bp=False,
                method_label=ps.LABEL_BP,
                sweep_dir=sweep_dir,
            )
        )
    pc_results = []
    for i, hparams in enumerate(pc_grid, start=1):
        print(f"\n----- PC config {i}/{len(pc_grid)}: {_format_hp(hparams)} -----")
        pc_results.append(
            run_config_with_seeds(
                args,
                hparams,
                skip_pc=False,
                skip_bp=True,
                method_label=ps.LABEL_PC,
                sweep_dir=sweep_dir,
            )
        )

    bp_ranked = rank_by_mean_test_acc(bp_results)
    pc_ranked = rank_by_mean_test_acc(pc_results)
    total_wall_time_s = time.perf_counter() - t_sweep
    n_seeds = args.n_seeds
    bp_rows = [
        _compact_result(result, rank, n_seeds)
        for rank, result in enumerate(bp_ranked, start=1)
    ]
    pc_rows = [
        _compact_result(result, rank, n_seeds)
        for rank, result in enumerate(pc_ranked, start=1)
    ]

    summary = {
        "dataset": args.dataset,
        "arch": args.arch,
        "loss": args.loss_id,
        "n_epochs": args.n_epochs,
        "n_seeds": n_seeds,
        "base_seed": args.seed,
        "metric": "final test accuracy (mean over seeds)",
        "wall_time_s": _round_metric(total_wall_time_s, 1),
        "fixed": _fixed_training_args(args),
        "grid": {
            "bp": _grid_values(args, bp_keys) if do_bp else {},
            "pc": _grid_values(args, pc_keys) if do_pc else {},
        },
        "best": {
            "bp": _compact_best(bp_ranked[0] if bp_ranked else None, n_seeds),
            "pc": _compact_best(pc_ranked[0] if pc_ranked else None, n_seeds),
        },
        "bp": bp_rows,
        "pc": pc_rows,
    }

    summary_path = os.path.join(sweep_dir, "sweep_summary.json")
    if not args.plot_from_npy:
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        with open(os.path.join(sweep_dir, "best_bp.json"), "w", encoding="utf-8") as handle:
            json.dump(summary["best"]["bp"], handle, indent=2)
        with open(os.path.join(sweep_dir, "best_pc.json"), "w", encoding="utf-8") as handle:
            json.dump(summary["best"]["pc"], handle, indent=2)

    _print_ranked_table("BP sweep (mean final test accuracy)", bp_ranked)
    _print_ranked_table("PC sweep (mean final test accuracy)", pc_ranked)
    print(f"\nSweep wall time: {total_wall_time_s:.1f}s")
    print(f"Sweep runs: {os.path.join(sweep_dir, 'bp')} and {os.path.join(sweep_dir, 'pc')}")
    if args.plot_from_npy:
        print("plot_from_npy: skipped rewriting sweep_summary.json")
    else:
        print(f"Sweep summary written to {summary_path}")
    return summary


if __name__ == "__main__":
    args = parse_args()
    args.dataset = normalize_dataset_id(args.dataset)
    if args.plot_from_npy:
        print(
            "plot_from_npy: skipping training; "
            "rebuilding figures from .npy files."
        )
    if args.dataset == "tiny-CIFAR10":
        if args.arch is None:
            args.arch = "mlp"
            print(f"Using default --arch mlp for {args.dataset}")
        if args.arch != "mlp":
            raise SystemExit(
                "tiny-CIFAR10 requires --arch mlp (grayscale binary MSE)"
            )
        if args.loss_id != "mse":
            print(
                "tiny-CIFAR10 uses MSE (labels are {-1, +1}); "
                "overriding --loss_id"
            )
            args.loss_id = "mse"
    if args.arch is None:
        args.arch = default_arch_for_dataset(args.dataset)
        print(f"Using default --arch {args.arch} for {args.dataset}")
    if args.n_seeds < 1:
        raise SystemExit("--n_seeds must be >= 1")
    if args.n_mini_per_epoch < 1:
        raise SystemExit("--n_mini_per_epoch must be >= 1")
    if args.skip_pc and args.skip_bp:
        raise SystemExit("Cannot use --skip_pc and --skip_bp together")
    if args.pc_infer_mode == "closed_form":
        if args.act_fn != "linear":
            raise SystemExit(
                "--pc_infer_mode closed_form requires --act_fn linear"
            )
        if args.loss_id != "mse":
            raise SystemExit(
                "--pc_infer_mode closed_form requires --loss_id mse"
            )
        if args.arch != "mlp":
            raise SystemExit(
                "--pc_infer_mode closed_form requires --arch mlp"
            )

    warn_unused_arch_sweep_axes(args)
    if is_hp_sweep(args):
        run_hp_sweep(args)
    else:
        _scalarize_sweep_fields(args)
        histories = []
        base_seed = args.seed
        for seed in range(base_seed, base_seed + args.n_seeds):
            run_args = make_run_args(args, seed=seed)
            _, history = run_benchmark(run_args)
            histories.append(history)
        plot_seed_aggregate(args, histories)



# # Frozen 40-example subset (full-batch GD, similar to analyse_alignment.py) - For testing
# python train_benchmark.py --dataset tiny-CIFAR10 --fixed_subset --batch_size 40 --n_epochs 100 --width 256 --n_hidden 2 --param_lr 0.05 --param_lr_pc 0.05 --activity_lr 0.1 --n_infer_iters 20 --param_optim gd --act_fn relu --loss_id mse --results_dir results_fixed_batch

# # Linear MLP (closed-form PC equilibrium; MSE), MNIST
# python train_benchmark.py --dataset MNIST --n_epochs 10 --batch_size 64 --width 256 --n_hidden 2 --param_lr 0.01 --param_lr_pc 0.01 --param_optim adam --act_fn linear --loss_id mse --pc_infer_mode closed_form --results_dir results_mnist_linear

# # MLP, MNIST
# python train_benchmark.py --dataset MNIST --n_epochs 10 --batch_size 64 --width 256 --n_hidden 2 --param_lr 0.1 --param_lr_pc 0.1 --activity_lr 0.01 --n_infer_iters 20 --param_optim adam --act_fn relu --n_seeds 3 --results_dir results_mnist

# # MLP, Fashion-MNIST
# python train_benchmark.py --dataset Fashion-MNIST --n_epochs 10 --batch_size 128 --width 256 --n_hidden 2 --param_lr 0.3 --param_lr_pc 0.3 --activity_lr 0.001 --n_infer_iters 20 --param_optim adam --act_fn relu --n_seeds 3 --results_dir results_fashion_mnist

# # CNN, CIFAR-10
# python train_benchmark.py --dataset CIFAR10 --arch cnn --n_epochs 100 --batch_size 64 --width 256 --n_res_blocks 3 --param_lr 0.1 --param_lr_pc 0.1 --activity_lr 0.01 --n_infer_iters 20 --param_optim sgd_momentum --act_fn relu --results_dir results_cifar

# # CNN L / κ variants (defaults: --resnet_fwd_l n_res_blocks, --resnet_energy_l n_weight_layers, --hidden_energy_layers weight)
# python train_benchmark.py --dataset CIFAR10 --arch cnn --resnet_fwd_l n_res_blocks --resnet_energy_l n_weight_layers --hidden_energy_layers weight --results_dir results_cifar_l_default
# python train_benchmark.py --dataset CIFAR10 --arch cnn --resnet_fwd_l n_weight_layers --resnet_energy_l n_weight_layers --hidden_energy_layers all --results_dir results_cifar_l_old
# python train_benchmark.py --dataset CIFAR10 --arch cnn --resnet_fwd_l n_res_blocks --resnet_energy_l n_modules --hidden_energy_layers all --results_dir results_cifar_l_modules
# python train_benchmark.py --dataset CIFAR10 --arch cnn --resnet_fwd_l n_res_blocks --resnet_energy_l n_weight_layers --hidden_energy_layers residual --results_dir results_cifar_l_res
# python train_benchmark.py --dataset CIFAR10 --arch cnn --additive_depth_factor 4 --results_dir results_cifar_l_energy_r_plus_4

# # CNN, ImageNet (HF streaming) - Not optimised
# python train_benchmark.py --dataset ImageNet --arch cnn --n_epochs 100 --batch_size 64 --width 256 --n_res_blocks 3 --param_lr 0.1 --param_lr_pc 0.1 --activity_lr 0.01 --n_infer_iters 20 --param_optim sgd_momentum --act_fn relu --results_dir results_imagenet


########### SWEEP (MNIST) ##############
########################################

# # MLP, MNIST: Hyperparameter sweep (Coarse) - Same for Fashion-MNIST (change dataset and name)
# # python train_benchmark.py --dataset Fashion-MNIST --n_epochs 6 --n_seeds 1 \
# python train_benchmark.py --dataset MNIST --n_epochs 6 --n_seeds 1 \
#   --width 256 --n_hidden 2 --batch_size 64 \
#   --param_lr 0.001 0.003 0.01 0.03 0.1 0.3 1.0 3.0 \
#   --param_lr_pc 0.01 0.03 0.1 0.3 1.0 3.0 \
#   --activity_lr 0.001 0.003 0.01 0.03 0.1 0.3 \
#   --n_infer_iters 20 200 \
#   --param_optim adam --act_fn relu \
#   --results_dir results_mnist_sweep_coarse
# #   --results_dir results_fashion_mnist_sweep_coarse

# # MLP, MNIST: Hyperparameter sweep (Fine) - Same for Fashion-MNIST (change dataset and name) - bs64
# # python train_benchmark.py --dataset Fashion-MNIST --n_epochs 10 --n_seeds 3 \
# python train_benchmark.py --dataset MNIST --n_epochs 10 --n_seeds 3 \
#   --width 256 --n_hidden 2 --batch_size 64 \
#   --param_lr 0.03 0.1 0.3 1.0 \
#   --param_lr_pc 0.03 0.1 0.3 1.0 \
#   --activity_lr 0.001 0.003 0.01 0.03 0.1 \
#   --n_infer_iters 20 \
#   --param_optim adam --act_fn relu \
#   --results_dir results_mnist_sweep_fine_bs64
# #   --results_dir results_fashion_mnist_sweep_fine_bs64

# # MLP, MNIST: Hyperparameter sweep (Fine) - Same for Fashion-MNIST (change dataset and name) - bs128
# # python train_benchmark.py --dataset Fashion-MNIST --n_epochs 10 --n_seeds 3 \
# python train_benchmark.py --dataset MNIST --n_epochs 10 --n_seeds 3 \
#   --width 256 --n_hidden 2 --batch_size 128 \
#   --param_lr 0.03 0.1 0.3 1.0 \
#   --param_lr_pc 0.03 0.1 0.3 1.0 \
#   --activity_lr 0.001 0.003 0.01 0.03 0.1 \
#   --n_infer_iters 20 \
#   --param_optim adam --act_fn relu \
#   --results_dir results_mnist_sweep_fine_bs128
# #   --results_dir results_fashion_mnist_sweep_fine_bs128


########### SWEEP (CIFAR) ##############
########################################

# # CNN, CIFAR-10: Hyperparameter sweep including residual-block depth (Can start lr at 0.03, probably 0.1 best)
# python train_benchmark.py --dataset CIFAR10 --arch cnn --n_epochs 100 --n_seeds 1 \
#   --width 256 --n_res_blocks 3 --batch_size 64 \
#   --param_lr 0.001 0.003 0.01 0.03 0.1 0.3 1.0 \
#   --param_lr_pc 0.01 0.03 0.1 0.3 1.0 3.0 \
#   --activity_lr 0.01 0.03 0.1 0.3 \
#   --n_infer_iters 20 200 \
#   --param_optim sgd_momentum --act_fn relu \
#   --results_dir results_cifar_sweep


