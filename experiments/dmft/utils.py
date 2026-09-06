"""
Shared training utilities live in ``experiments.limits_paper.utils``. This
module overrides the BP ``MLP`` so µPC weights are N(0, 1), matching
``jpc.make_mlp``, and hosts the finite-size PC/BP training loops.
"""

import os
import shutil
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

import jpc
from experiments.datasets import CIFAR10
from experiments.limits_paper.utils import MLP as LimitsMLP
from experiments.limits_paper.utils import configure_param_optim, flatten_grads


def mlp_adam_lr(param_lr, param_type, use_skips, width, depth):
    """Adam LR matching ``configure_param_optim`` (BP and, here, PC)."""
    if param_type == "sp":
        return param_lr
    if use_skips:
        return param_lr / (np.sqrt(width) * np.sqrt(depth))
    return param_lr / np.sqrt(width)


def bp_gd_style_lr(param_lr, param_type, gamma_0, width):
    """BP GD / SGD+momentum LR: µP bakes ``γ² N`` into the optimiser."""
    if param_type == "sp":
        return param_lr
    return param_lr * (gamma_0 ** 2) * width


def make_sgd_param_optim(learning_rate, param_optim_id, momentum=0.9):
    """Vanilla GD or SGD+momentum. No Adam-style ``1/√N`` rescaling."""
    if param_optim_id == "sgd_momentum":
        return optax.sgd(learning_rate, momentum=momentum)
    return optax.sgd(learning_rate)


def create_toy_dataset(key, D, P):
    X = jr.normal(key, (D, P))
    y = jnp.where(jnp.arange(P) < P//2, 1.0, -1.0)
    return X, y


CIFAR_GRAY_DIM = 32 * 32


def create_tiny_cifar10_dataset(key, D, P, class0=0, class1=1, train=True):
    """Subsample a binary grayscale CIFAR-10 dataset.

    Loads CIFAR-10 via ``experiments.datasets.CIFAR10``, converts RGB
    images to grayscale, and draws ``P // 2`` examples from each of two
    classes. Labels are mapped to ``{-1, +1}``. ``train=True`` samples
    the official train split; ``train=False`` samples the official test
    split.

    Args:
        key: JAX PRNG key for sampling indices.
        D: Input dimension; must be ``32 * 32`` (flattened grayscale).
        P: Number of samples (must be even).
        class0: CIFAR-10 class mapped to ``-1`` (default: airplane).
        class1: CIFAR-10 class mapped to ``+1`` (default: automobile).
        train: If True, sample the train split; else the test split.

    Returns:
        X: array of shape ``(D, P)``.
        y: array of shape ``(P,)``.
    """
    if D != CIFAR_GRAY_DIM:
        raise ValueError(
            f"Grayscale CIFAR-10 has dimension {CIFAR_GRAY_DIM}, got D={D}."
        )
    if P % 2 != 0:
        raise ValueError(f"P must be even so classes are balanced, got P={P}.")

    dataset = CIFAR10(
        train=train,
        normalise=False,
        flatten=False,
        save_dir=str(Path(__file__).resolve().parent / "datasets" / "cifar10"),
    )
    X_rgb = np.asarray(dataset.data, dtype=np.float32) / 255.0
    labels = np.asarray(dataset.targets)

    # ITU-R BT.601 luma, matching the usual CIFAR grayscale conversion.
    X_gray = (
        0.2989 * X_rgb[..., 0]
        + 0.5870 * X_rgb[..., 1]
        + 0.1140 * X_rgb[..., 2]
    )

    inds0 = np.where(labels == class0)[0]
    inds1 = np.where(labels == class1)[0]
    n = P // 2
    if n > len(inds0) or n > len(inds1):
        raise ValueError(
            f"Requested {n} samples per class but class {class0} has "
            f"{len(inds0)} and class {class1} has {len(inds1)}."
        )

    key0, key1 = jr.split(key)
    idx0 = inds0[np.asarray(jr.choice(key0, len(inds0), (n,), replace=False))]
    idx1 = inds1[np.asarray(jr.choice(key1, len(inds1), (n,), replace=False))]

    X = np.concatenate([X_gray[idx0], X_gray[idx1]], axis=0).reshape(P, D).T
    y = np.concatenate([np.full(n, -1.0), np.full(n, 1.0)])
    return jnp.asarray(X), jnp.asarray(y)


def cosine_similarity(a, b, axis=None, eps=1e-10):
    """Cosine similarity between two arrays, or pairwise over sequences.

    ``axis=None`` flattens both arrays (Frobenius / vector cosine sim).
    Pass ``axis=-1`` for a batch of vectors. Lists/tuples are compared
    pairwise, e.g. PC and BP grads over training steps.
    """
    if isinstance(a, (list, tuple)):
        return jnp.array([
            cosine_similarity(x, y, axis=axis, eps=eps)
            for x, y in zip(a, b)
        ])

    a, b = jnp.asarray(a), jnp.asarray(b)
    if axis is None:
        a, b = a.reshape(-1), b.reshape(-1)
        n = min(a.shape[0], b.shape[0])
        a, b = a[:n], b[:n]
        axis = 0

    num = jnp.sum(a * b, axis=axis)
    denom = jnp.linalg.norm(a, axis=axis) * jnp.linalg.norm(b, axis=axis) + eps
    return num / denom


def relative_frobenius_displacement(C0, Ct, eps=1e-30):
    """Relative Frobenius displacement ``||Ct - C0||_F / ||C0||_F``.

    Unlike cosine similarity, this tracks magnitude change as well as
    direction. ``C0`` / ``Ct`` are flattened; mismatched sizes are
    truncated to the shared leading length (same convention as
    ``cosine_similarity``).
    """
    C0 = np.asarray(C0, dtype=np.float64).reshape(-1)
    Ct = np.asarray(Ct, dtype=np.float64).reshape(-1)
    n = min(C0.size, Ct.size)
    C0, Ct = C0[:n], Ct[:n]
    return float(np.linalg.norm(Ct - C0) / (np.linalg.norm(C0) + eps))


def _center_gram(K):
    """Double-center a Gram matrix: ``H K H`` with ``H = I - 11^T / n``."""
    K = np.asarray(K, dtype=np.float64)
    row_mean = K.mean(axis=1, keepdims=True)
    col_mean = K.mean(axis=0, keepdims=True)
    return K - row_mean - col_mean + K.mean()


def centered_kernel_alignment(K, L, eps=1e-30):
    """Linear CKA (Kornblith et al.): cosine similarity of centered Grams.

    When both centered kernels are numerically zero (e.g. ``C(t)-C(0)``
    at ``t=0``), returns ``1.0``, matching the limit
    ``CKA(εI, εI) → 1``.
    """
    K = np.asarray(K, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    if K.shape != L.shape:
        raise ValueError(
            f"CKA requires matching shapes, got {K.shape} vs {L.shape}"
        )
    Kc = _center_gram(K)
    Lc = _center_gram(L)
    na = np.linalg.norm(Kc)
    nb = np.linalg.norm(Lc)
    denom = na * nb
    if denom < eps:
        # Both zero: CKA(εI, εI) → 1. One zero: undefined, treat as 0.
        return 1.0 if na < eps and nb < eps else 0.0
    return float(np.sum(Kc * Lc) / denom)


def kernel_eigh(C, centered=False):
    """Descending eigenvalues and eigenvectors (columns) of a kernel.

    Symmetrizes ``(C + C.T) / 2``. If ``centered`` is True, uses the
    double-centered Gram ``H C H``.
    """
    C = np.asarray(C, dtype=np.float64)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"expected a square kernel, got shape {C.shape}")
    if centered:
        C = _center_gram(C)
    C = 0.5 * (C + C.T)
    evals, evecs = np.linalg.eigh(C)
    return evals[::-1].copy(), evecs[:, ::-1].copy()


def kernel_eigs(C):
    """Descending eigenvalues of a symmetric kernel.

    Symmetrizes ``(C + C.T) / 2`` and clips numerically negative
    eigenvalues to 0 (Gram matrices are PSD).
    """
    evals, _ = kernel_eigh(C, centered=False)
    return np.maximum(evals, 0.0)


def top_k_eigenvectors(C, k, centered=False):
    """Orthonormal leading-``k`` eigenvectors (columns) of a kernel."""
    _, evecs = kernel_eigh(C, centered=centered)
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    k = min(k, evecs.shape[1])
    return evecs[:, :k]


def subspace_overlap(C_a, C_b, k, centered=True):
    """Mean squared principal cosine between leading-``k`` eigenspaces.

    ``||U_a^T U_b||_F^2 / k``, i.e. the average of ``cos^2 θ_i``. Equals
    1 iff the two ``k``-planes coincide. Uses centered kernels by
    default (the same centering as CKA).
    """
    Ua = top_k_eigenvectors(C_a, k, centered=centered)
    Ub = top_k_eigenvectors(C_b, k, centered=centered)
    k_eff = min(Ua.shape[1], Ub.shape[1])
    if k_eff < 1:
        return float("nan")
    M = Ua[:, :k_eff].T @ Ub[:, :k_eff]
    return float(np.sum(M * M) / k_eff)


def _as_sample_matrix(Y):
    """Ensure ``(P, d)`` with samples on axis 0."""
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    return Y


def leading_evec_label_overlap(C, Y, eps=1e-30):
    """Overlap of the centered kernel's leading eigenvector with labels.

    For a single label column this is ``|cos(v_1, y)|`` after centering
    ``y``. For several columns it is the cosine of ``v_1`` with the
    column space of centered ``Y`` (``||U_y^T v_1||``).
    """
    v = top_k_eigenvectors(C, 1, centered=True)[:, 0]
    v_norm = np.linalg.norm(v)
    if v_norm < eps:
        return 0.0
    v = v / v_norm
    Yc = _as_sample_matrix(Y)
    Yc = Yc - Yc.mean(axis=0, keepdims=True)
    Uy, svals, _ = np.linalg.svd(Yc, full_matrices=False)
    if svals.size == 0 or svals[0] < eps:
        return 0.0
    rank = int(np.sum(svals > eps * svals[0]))
    return float(np.linalg.norm(Uy[:, :rank].T @ v))


def participation_ratio_from_eigs(eigs, eps=1e-30):
    """Participation ratio ``(sum λ)^2 / sum λ^2`` from kernel eigenvalues."""
    eigs = np.asarray(eigs, dtype=np.float64).reshape(-1)
    eigs = eigs[np.isfinite(eigs)]
    eigs = np.maximum(eigs, 0.0)
    s1 = float(eigs.sum())
    s2 = float(np.square(eigs).sum())
    if s2 < eps:
        return 0.0
    return (s1 * s1) / s2


def participation_ratio(C, eps=1e-30):
    """Effective rank of a kernel via the participation ratio of its eigs."""
    return participation_ratio_from_eigs(kernel_eigs(C), eps=eps)


def gram_to_correlation(C, eps=1e-30):
    """Convert a Gram / kernel matrix to a correlation matrix.

    ``R_ij = C_ij / sqrt(C_ii C_jj)``. Non-positive diagonals yield a
    zero row/column (no finite correlation).
    """
    C = np.asarray(C, dtype=np.float64)
    d = np.sqrt(np.maximum(np.diag(C), 0.0))
    denom = np.outer(d, d)
    R = np.zeros_like(C)
    np.divide(C, denom, out=R, where=denom > eps)
    return R


def kernels_to_correlations(kernels, eps=1e-30):
    """Map a sequence of Gram matrices to correlation matrices."""
    return [gram_to_correlation(K, eps=eps) for K in kernels]


def _as_layer_list(model):
    if isinstance(model, (list, tuple)):
        return list(model)
    return list(model.layers)


def copy_mlp_linear_params(src_model, dst_model):
    """Copy Linear weights (and biases) from ``src_model`` onto ``dst_model``.

    ``src_model`` is a ``jpc.make_mlp`` layer list (or anything with a
    ``.layers`` attribute of ``Sequential([Lambda, Linear])`` blocks);
    ``dst_model`` is the BP ``MLP``. Returns a new ``dst_model``. Layer
    counts and weight shapes must match.
    """
    src_layers = _as_layer_list(src_model)
    dst_layers = _as_layer_list(dst_model)
    if len(src_layers) != len(dst_layers):
        raise ValueError(
            "src/dst layer counts differ: "
            f"{len(src_layers)} vs {len(dst_layers)}"
        )
    new_layers = []
    for i, (s_layer, d_layer) in enumerate(zip(src_layers, dst_layers)):
        s_lin, d_lin = s_layer[1], d_layer[1]
        if s_lin.weight.shape != d_lin.weight.shape:
            raise ValueError(
                f"layer {i} weight shape {tuple(s_lin.weight.shape)} vs "
                f"{tuple(d_lin.weight.shape)}"
            )
        d_lin = eqx.tree_at(
            lambda l: l.weight, d_lin, jnp.array(s_lin.weight)
        )
        if s_lin.bias is not None:
            if d_lin.bias is None:
                raise ValueError(
                    f"layer {i}: src has a bias but dst does not"
                )
            d_lin = eqx.tree_at(
                lambda l: l.bias, d_lin, jnp.array(s_lin.bias)
            )
        new_layers.append(eqx.tree_at(lambda s: s[1], d_layer, d_lin))
    return eqx.tree_at(lambda m: m.layers, dst_model, new_layers)


def pc_hidden_preactivations_and_errors(
    model,
    skip_model,
    activities,
    x,
    param_type,
    gamma,
):
    """Hidden pre-activations ``h`` and PC prediction errors ``Δ``.

    ``jpc.make_mlp`` stores pre-activations: layer ``ℓ`` maps
    ``h^ℓ = a_ℓ W_ℓ φ(h^{ℓ-1})``. Hidden errors are the energy residuals
    ``Δ^ℓ = h^ℓ - f_ℓ(h^{ℓ-1})``, which vanish on the feedforward init
    (``k = 0``), matching the DMFT boundary ``Δ_0 = 0``.
    """
    scalings = jpc._get_param_scalings(
        model=model,
        input=x,
        skip_model=skip_model,
        param_type=param_type,
        gamma=gamma,
    )
    n_hidden = len(model) - 1
    if skip_model is None:
        skip_model = [None] * len(model)

    hs = []
    deltas = []
    for l in range(n_hidden):
        h = activities[l]
        prev = x if l == 0 else activities[l - 1]
        pred = scalings[l] * jax.vmap(model[l])(prev)
        if skip_model[l] is not None:
            pred = pred + jax.vmap(skip_model[l])(prev)
        hs.append(h)
        deltas.append(h - pred)
    return hs, deltas


def empirical_pc_kernel(field):
    """Finite-width PC kernel from a field with trailing neuron axis ``N``.

    All leading axes are flattened (slowest-first), matching the DMFT
    ``(k, t, mu)`` convention when those axes are present.
    """
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim < 2:
        raise ValueError(f"expected at least (..., N), got shape {arr.shape}")
    n_units = arr.shape[-1]
    if n_units < 1:
        raise ValueError("width N must be positive")
    flat = arr.reshape(-1, n_units)
    return (flat @ flat.T) / n_units


def final_time_pc_kernel(
    cov, num_inference_steps, num_training_steps, num_samples, k=0, t=-1
):
    """Sample-sample kernel at inference step ``k`` and training time ``t``.

    ``t`` defaults to ``-1`` (the last training step); pass ``t=0`` for the
    kernel right after initialisation, before any parameter update.
    """
    K1 = num_inference_steps + 1
    if not (0 <= k < K1):
        raise ValueError(
            f"k={k} is out of range for K={num_inference_steps} "
            f"(valid 0,...,{num_inference_steps})."
        )
    T = num_training_steps
    P = num_samples
    tensor = np.asarray(cov, dtype=np.float64).reshape(K1, T, P, K1, T, P)
    return tensor[k, t, :, k, t, :]


def sample_traced_time_kernel(cov_tp):
    """Reduce a ``(T, P, T, P)`` kernel to ``(T, T)`` by summing the sample diagonal.

    Returns ``C_T[t, t'] = sum_mu C[t, mu, t', mu]``.
    """
    arr = np.asarray(cov_tp, dtype=np.float64)
    if arr.ndim != 4 or arr.shape[0] != arr.shape[2] or arr.shape[1] != arr.shape[3]:
        raise ValueError(f"expected (T, P, T, P), got {arr.shape}")
    return np.einsum("tmsm->ts", arr)


def sample_traced_pc_kernel(
    cov, num_inference_steps, num_training_steps, num_samples, k=0
):
    """Sample-traced time-time kernel at inference step ``k``.

    Reshapes the flattened ``((K+1)*T*P, (K+1)*T*P)`` covariance and
    returns ``C_T[t, t'] = sum_mu C[k, t, mu, k, t', mu]``.
    """
    K1 = num_inference_steps + 1
    if not (0 <= k < K1):
        raise ValueError(
            f"k={k} is out of range for K={num_inference_steps} "
            f"(valid 0,...,{num_inference_steps})."
        )
    T = num_training_steps
    P = num_samples
    tensor = np.asarray(cov, dtype=np.float64).reshape(K1, T, P, K1, T, P)
    return sample_traced_time_kernel(tensor[k, :, :, k, :, :])


def sample_traced_empirical_pc_kernel(field):
    """Sample-traced ``T x T`` kernel from a field of shape ``(T, P, N)``."""
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"expected (T, P, N), got {arr.shape}")
    T, P, _ = arr.shape
    return sample_traced_time_kernel(
        empirical_pc_kernel(arr).reshape(T, P, T, P)
    )


def bp_sample_kernel_at(kernel, t=-1):
    """Sample-sample block of a BP feature kernel ``(T, P, T, P)`` at time ``t``.

    ``t`` defaults to ``-1`` (the last training step); pass ``t=0`` for the
    kernel right after initialisation, before any parameter update. Works
    for both linear (``all_H``) and nonlinear (``all_Phi``) BP DMFT kernels,
    which share this shape.
    """
    arr = np.asarray(kernel, dtype=np.float64)
    return arr[t, :, t, :]


def bp_hidden_preactivations(model, X_input):
    """Per-layer pre-activation ``h^l`` for the finite-width BP MLP.

    Mirrors ``pc_hidden_preactivations_and_errors``'s ``h`` convention:
    each hidden layer's pre-activation output ``h^l`` is returned
    *before* the next layer's nonlinearity is applied (so ``phi(h^l)``,
    computed later, is the actual feature fed forward). This lets BP and
    PC feature kernels be compared on equal footing. Only the ``n_hidden``
    hidden layers are returned; the readout (last of ``model.L`` layers)
    is omitted, matching the PC convention.

    Args:
        model: a ``dmft.utils.MLP`` (or ``limits_paper.utils.MLP``).
        X_input: array of shape ``(P, D)``.

    Returns:
        A list of ``n_hidden`` arrays, each of shape ``(P, N)``.
    """
    param_type = model.param_type
    N, L, D = model.N, model.L, model.D
    use_skips = model.use_skips

    def _single(x):
        hs = []
        h = x
        if param_type == "mupc":
            for i, f in enumerate(model.layers):
                if (i + 1) == 1:
                    h = f(h) / jnp.sqrt(D)
                elif 1 < (i + 1) < L:
                    residual = h if use_skips else 0.0
                    rescaling = (
                        jnp.sqrt(N * L) if use_skips else jnp.sqrt(N)
                    )
                    h = (f(h) / rescaling) + residual
                else:
                    h = f(h) / N
                if (i + 1) < L:
                    hs.append(h)
        else:
            for i, f in enumerate(model.layers):
                residual = (
                    h if use_skips and (1 < (i + 1) < L) else 0.0
                )
                h = f(h) + residual
                if (i + 1) < L:
                    hs.append(h)
        return hs

    hs_batched = jax.vmap(_single)(X_input)
    return list(hs_batched)


def collect_final_pc_kernel_fields(
    model,
    skip_model,
    X_input,
    Y_target,
    width,
    param_type,
    gamma_0,
    activity_lr,
    loss_id,
    n_infer_iters,
):
    """Hidden ``h`` at ``k=0`` (feedforward) and ``Δ`` at ``k=n_infer_iters``.

    Runs ``n_infer_iters`` steps of iterative inference (mirroring the
    inference loop in ``train_pcn``) so ``Δ`` is taken at the same final
    inference step ``k=K`` used by the PC DMFT theory and finite-size
    training, rather than after a single step.

    Returns arrays of shape ``(n_hidden, P, N)``.
    """
    depth = len(model)
    output_energy_scaling = get_output_energy_scaling(
        param_type, gamma_0, width, depth
    )
    hidden_energy_scaling = get_hidden_energy_scaling(param_type, depth)
    batch_size = X_input.shape[0]
    activity_optim = optax.sgd(activity_lr * batch_size)

    activities = jpc.init_activities_with_ffwd(
        model=model,
        input=X_input,
        skip_model=skip_model,
        param_type=param_type,
        gamma=gamma_0,
    )
    hs, _ = pc_hidden_preactivations_and_errors(
        model=model,
        skip_model=skip_model,
        activities=activities,
        x=X_input,
        param_type=param_type,
        gamma=gamma_0,
    )

    activity_opt_state = activity_optim.init(activities)
    for _ in range(n_infer_iters):
        activity_update_result = jpc.update_pc_activities(
            params=(model, skip_model),
            activities=activities,
            optim=activity_optim,
            opt_state=activity_opt_state,
            output=Y_target,
            input=X_input,
            param_type=param_type,
            gamma=gamma_0,
            loss_id=loss_id,
            output_energy_scaling=output_energy_scaling,
            hidden_energy_scaling=hidden_energy_scaling,
        )
        activities = activity_update_result["activities"]
        activity_opt_state = activity_update_result["opt_state"]
    _, deltas = pc_hidden_preactivations_and_errors(
        model=model,
        skip_model=skip_model,
        activities=activities,
        x=X_input,
        param_type=param_type,
        gamma=gamma_0,
    )
    return {
        "h": np.stack([np.asarray(h, dtype=np.float32) for h in hs], axis=0),
        "delta": np.stack(
            [np.asarray(d, dtype=np.float32) for d in deltas], axis=0
        ),
    }


class MLP(LimitsMLP):
    def __init__(
            self,
            key,
            d_in,
            N,
            L,
            d_out,
            act_fn,
            param_type,
            gamma,
            use_bias=False,
            use_skips=False
    ):
        super().__init__(
            key,
            d_in,
            N,
            L,
            d_out,
            act_fn,
            param_type,
            gamma,
            use_bias=use_bias,
            use_skips=use_skips,
        )
        # µPC applies explicit forward scalings, so weights must be N(0, 1)
        # (same convention as jpc.make_mlp). Equinox's default 1/sqrt(fan_in)
        # init would otherwise double-scale and kill learning.
        if param_type != "mupc":
            return
        keys = jr.split(key, L)
        layers = []
        for i, layer in enumerate(self.layers):
            linear = layer[1]
            W = jr.normal(keys[i], linear.weight.shape)
            linear = eqx.tree_at(lambda l: l.weight, linear, W)
            layers.append(eqx.tree_at(lambda s: s[1], layer, linear))
        object.__setattr__(self, "layers", layers)


def get_output_energy_scaling(
    param_type: str, gamma_0: float, width: int, depth: int
) -> float:
    """µPC output precision λ = γ² N L (SP: 1)."""
    return (gamma_0 ** 2) * width * depth if param_type == "mupc" else 1.0


def get_hidden_energy_scaling(param_type: str, depth: int) -> float:
    """µPC hidden precision κ = L (SP: 1)."""
    return float(depth) if param_type == "mupc" else 1.0


def cleanup_experiment_dirs(results_dir: str):
    """Remove finite-sim result trees (``*_input_dim``), keeping plot pngs."""
    removed = []
    root = Path(results_dir)
    if not root.exists():
        return removed
    for path in sorted(root.glob("*_input_dim")):
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(str(path))
    return removed


def train_pcn(
      model,
      use_skips,
      X_input,
      Y_target,
      width,
      gamma_0,
      param_type,
      infer_mode,
      n_infer_iters,
      activity_lr,
      param_optim_id,
      param_lr,
      n_train_iters,
      loss_id,
      save_dir,
      store_grads=False,
      h_k0_steps=None,
      X_eval=None,
      h_k0_eval_callback=None,
      momentum=0.9,
):
    """Train a PC network.

    Parameter / activity updates follow the finite-size convention used by
    ``get_coord_data``: GD and SGD+momentum use plain ``param_lr`` with
    ``output_energy_scaling = gamma^2 * width * depth`` and
    ``hidden_energy_scaling = depth`` for µPC (scale lives in the energy).
    Adam divides the PC LR the same way as BP (``1/√N``, or ``1/√(N L)``
    with MLP skips). Note that depth includes the output layer here, as
    opposed to depth in theory_utils.py.

    Returns ``(pc_grads, model, skip_model)``. ``pc_grads`` is ``None``
    unless ``store_grads`` is True. If ``h_k0_steps`` is a list, each
    training step appends hidden ``h`` at ``k=0`` (feedforward init,
    before the parameter update) with shape ``(n_hidden, P, N)``.
    ``X_eval`` / ``h_k0_eval_callback`` optionally record the same
    feedforward ``h`` on a held-out batch at those snapshots: the
    callback is called with a list of ``(P, N)`` hidden activations.
    """
    os.makedirs(save_dir, exist_ok=True)
    if h_k0_eval_callback is not None and X_eval is None:
        raise ValueError("h_k0_eval_callback requires X_eval")

    depth = len(model)
    skip_model = jpc.make_skip_model(depth) if use_skips else None
    output_energy_scaling = get_output_energy_scaling(
        param_type, gamma_0, width, depth
    )
    hidden_energy_scaling = get_hidden_energy_scaling(param_type, depth)

    # GD / SGD+momentum: plain lr; µPC width/gamma/depth live in the energy.
    # Adam: divide like BP (``1/√N``, or ``1/√(N L)`` with skips).
    batch_size = X_input.shape[0]
    activity_optim = optax.sgd(activity_lr * batch_size)
    if param_optim_id in ("gd", "sgd_momentum"):
        param_optim = make_sgd_param_optim(
            param_lr, param_optim_id, momentum
        )
    elif param_optim_id == "adam":
        param_optim = optax.adam(
            mlp_adam_lr(param_lr, param_type, use_skips, width, depth)
        )
    else:
        raise ValueError(f"Invalid optimiser: {param_optim_id}")
    param_opt_state = param_optim.init(
        (eqx.filter(model, eqx.is_array), skip_model)
    )

    num_energies, theory_energies = [], []
    train_losses = []
    loss_rescalings = []
    pc_grads = [] if store_grads else None

    for _ in range(n_train_iters):

        # Record supervised loss on the current feedforward prediction *before*
        # the parameter update, matching get_coord_data / DMFT step indexing.
        activities = jpc.init_activities_with_ffwd(
            model=model,
            input=X_input,
            skip_model=skip_model,
            param_type=param_type,
            gamma=gamma_0
        )
        if h_k0_steps is not None:
            hs_k0, _ = pc_hidden_preactivations_and_errors(
                model=model,
                skip_model=skip_model,
                activities=activities,
                x=X_input,
                param_type=param_type,
                gamma=gamma_0,
            )
            h_k0_steps.append(
                np.stack(
                    [np.asarray(h, dtype=np.float32) for h in hs_k0],
                    axis=0,
                )
            )
        if h_k0_eval_callback is not None:
            activities_eval = jpc.init_activities_with_ffwd(
                model=model,
                input=X_eval,
                skip_model=skip_model,
                param_type=param_type,
                gamma=gamma_0,
            )
            hs_eval, _ = pc_hidden_preactivations_and_errors(
                model=model,
                skip_model=skip_model,
                activities=activities_eval,
                x=X_eval,
                param_type=param_type,
                gamma=gamma_0,
            )
            h_k0_eval_callback(hs_eval)
        if loss_id == "mse":
            train_loss = jpc.mse_loss(activities[-1], Y_target)
        else:
            train_loss = jpc.cross_entropy_loss(activities[-1], Y_target)
        train_losses.append(train_loss)

        if infer_mode == "closed_form":
            equilib_energy, S = jpc.linear_equilib_energy(
                params=(model, skip_model),
                x=X_input,
                y=Y_target,
                param_type=param_type,
                gamma=gamma_0,
                return_rescaling=True,
                output_energy_scaling=output_energy_scaling,
                hidden_energy_scaling=hidden_energy_scaling,
            )
            theory_energies.append(equilib_energy)
            loss_rescaling = jnp.linalg.norm(S, ord=2) if Y_target.ndim > 1 else S
            loss_rescalings.append(loss_rescaling)

        # inference
        if infer_mode == "optim":
            activity_opt_state = activity_optim.init(activities)
            for _ in range(n_infer_iters):
                activity_update_result = jpc.update_pc_activities(
                    params=(model, skip_model),
                    activities=activities,
                    optim=activity_optim,
                    opt_state=activity_opt_state,
                    output=Y_target,
                    input=X_input,
                    param_type=param_type,
                    gamma=gamma_0,
                    loss_id=loss_id,
                    output_energy_scaling=output_energy_scaling,
                    hidden_energy_scaling=hidden_energy_scaling,
                )
                activities = activity_update_result["activities"]
                activity_opt_state = activity_update_result["opt_state"]
                energy = activity_update_result["energy"]

            num_energies.append(energy)

            param_update_result = jpc.update_pc_params(
                params=(model, skip_model),
                activities=activities,
                optim=param_optim,
                opt_state=param_opt_state,
                output=Y_target,
                input=X_input,
                param_type=param_type,
                gamma=gamma_0,
                loss_id=loss_id,
                output_energy_scaling=output_energy_scaling,
                hidden_energy_scaling=hidden_energy_scaling,
            )

        else:
            # learning with closed form energy
            param_update_result = jpc.update_linear_equilib_energy_params(
                params=(model, skip_model),
                optim=param_optim,
                opt_state=param_opt_state,
                y=Y_target,
                x=X_input,
                param_type=param_type,
                gamma=gamma_0,
                output_energy_scaling=output_energy_scaling,
                hidden_energy_scaling=hidden_energy_scaling,
            )

        model = param_update_result["model"]
        skip_model = param_update_result["skip_model"]
        param_opt_state = param_update_result["opt_state"]
        grads = param_update_result["grads"]

        if pc_grads is not None:
            flat_grads = flatten_grads(grads)
            # Convert JAX array to numpy immediately to free memory
            pc_grads.append(np.array(flat_grads))
            del flat_grads, grads

    energies = (
        jnp.array(theory_energies)
        if infer_mode == "closed_form"
        else jnp.array(num_energies)
    )
    np.save(f"{save_dir}/energies.npy", energies)
    np.save(f"{save_dir}/train_losses.npy", np.array(train_losses))
    np.save(f"{save_dir}/loss_rescalings.npy", loss_rescalings)

    return pc_grads, model, skip_model


def train_bpn(
      model,
      use_skips,
      X_input,
      Y_target,
      width,
      gamma_0,
      param_type,
      optim_id,
      param_lr,
      n_train_iters,
      loss_id,
      save_dir,
      store_grads=False,
      h_k0_steps=None,
      X_eval=None,
      h_k0_eval_callback=None,
      momentum=0.9,
):
    """Train a finite-width BP MLP.

    GD and SGD+momentum bake ``γ² N`` into the optimiser for µP (no
    Adam-style ``1/√N`` rescaling). Adam uses ``configure_param_optim``
    (``1/√N``, or ``1/√(N L)`` with MLP skips).

    If ``h_k0_steps`` is a list, each training step appends the hidden
    pre-activations ``h^l`` (see ``bp_hidden_preactivations``), *before*
    that step's parameter update, with shape ``(n_hidden, P, N)`` -
    mirroring ``train_pcn``'s ``h_k0_steps`` so PC and BP feature-kernel
    trajectories can be compared on equal footing.
    ``X_eval`` / ``h_k0_eval_callback`` optionally record the same
    feedforward ``h`` on a held-out batch at those snapshots: the
    callback is called with a list of ``(P, N)`` hidden activations.
    """
    os.makedirs(save_dir, exist_ok=True)
    if h_k0_eval_callback is not None and X_eval is None:
        raise ValueError("h_k0_eval_callback requires X_eval")

    if optim_id == "sgd_momentum":
        optim = make_sgd_param_optim(
            bp_gd_style_lr(param_lr, param_type, gamma_0, width),
            optim_id,
            momentum,
        )
    else:
        optim = configure_param_optim(
            optim_id, param_type, use_skips, param_lr, width, model.L, gamma_0
        )
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    if loss_id == "mse":
        @eqx.filter_jit
        def loss_fn(model, x, y):
            y_pred = jax.vmap(model)(x)
            return 0.5 * jnp.mean(jnp.sum((y - y_pred) ** 2, axis=1))
    else:
        @eqx.filter_jit
        def loss_fn(model, x, y):
            y_pred = jax.vmap(model)(x)
            return jpc.cross_entropy_loss(y_pred, y)

    @eqx.filter_jit
    def make_step(model, optim, opt_state, x, y):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = optim.update(
            updates=grads,
            state=opt_state,
            params=eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, grads

    losses = []
    bp_grads = [] if store_grads else None

    for _ in range(n_train_iters):
        if h_k0_steps is not None:
            hs = bp_hidden_preactivations(model, X_input)
            h_k0_steps.append(
                np.stack(
                    [np.asarray(h, dtype=np.float32) for h in hs], axis=0
                )
            )
        if h_k0_eval_callback is not None:
            hs_eval = bp_hidden_preactivations(model, X_eval)
            h_k0_eval_callback(hs_eval)

        # Record loss before the parameter update to match get_Delta / DMFT
        # step indexing (pre-update residual).
        if loss_id == "mse":
            y_pred = jax.vmap(model)(X_input)
            train_loss = float(
                0.5 * jnp.mean(jnp.sum((Y_target - y_pred) ** 2, axis=1))
            )
        else:
            y_pred = jax.vmap(model)(X_input)
            train_loss = float(jpc.cross_entropy_loss(y_pred, Y_target))
        losses.append(train_loss)

        model, opt_state, _, grads = make_step(
            model, optim, opt_state, X_input, Y_target
        )

        if bp_grads is not None:
            flat_grads = flatten_grads(grads)
            # Convert JAX array to numpy immediately to free memory
            bp_grads.append(np.array(flat_grads))
            del flat_grads, grads

    np.save(f"{save_dir}/losses.npy", losses)

    return bp_grads
