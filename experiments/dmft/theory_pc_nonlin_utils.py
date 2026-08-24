"""Utilities for non-linear predictive-coding DMFT with Monte-Carlo sampling.

This is the non-linear counterpart of ``theory_pc_utils``: the linear solver
there inverts the single-site equations exactly, whereas here the single-site
fields are sampled and the correlation/response functions are estimated as
averages over those samples (as in ``theory_nonlin_utils`` for backprop).

Saddle point equations solved per hidden layer ``ell``, with inference step
``k = 0, ..., K`` and training step ``t = 0, ..., T-1``:

    h^ell_0(t)     = chi^ell_0(t) + eta sum_{t'<t} C^{phi,ell-1}_{0,K}(t,t')
                                              Delta^ell_K(t')
    h^ell_{k+1}(t) = h^ell_k(t) - beta_h Delta^ell_k(t) + beta_h g^ell_k(t)
    Delta^ell_k(t) = h^ell_k(t) - chi^ell_k(t)
                     - eta sum_{t'<t} C^{phi,ell-1}_{k,K}(t,t') Delta^ell_K(t')
    z^ell_k(t)     = xi^ell_k(t)
                     + eta sum_{t'<t} C^{Delta,ell+1}_{k,K}(t,t') phi(h^ell_K(t'))
    chi^ell_k(t)   = u^{chi,ell}_k(t) + sum_{k',t'} R^{phi,ell-1}_{k,k'}(t,t')
                                                    Delta^ell_{k'}(t')
    xi^ell_k(t)    = u^{xi,ell}_k(t) + sum_{k',t'} R^{Delta,ell+1}_{k,k'}(t,t')
                                                   phi(h^ell_{k'}(t'))

with ``g^ell_k = phi'(h^ell_k) z^ell_k`` and the forward-pass boundary
condition ``Delta^ell_0(t) = 0``. The Gaussian sources are independent with
``<u^chi u^chi> = C^{phi,ell-1}`` and ``<u^xi u^xi> = C^{Delta,ell+1}``, and

    C^{phi,ell}_{kk'}(t,t')   = < phi(h^ell_k(t)) phi(h^ell_{k'}(t')) >
    C^{Delta,ell}_{kk'}(t,t') = < Delta^ell_k(t) Delta^ell_{k'}(t') >
    R^{phi,ell}_{kk'}(t,t')   = < d phi(h^ell_k(t)) / d u^{xi,ell}_{k'}(t') >
    R^{Delta,ell}_{kk'}(t,t') = < d Delta^ell_k(t) / d u^{chi,ell}_{k'}(t') >.

Sample (data) indices are suppressed above; every kernel carries an extra
``(mu, nu)`` pair and the memory sums include ``1/P sum_nu``.

Maximal-update (muP / muPC) conventions, matching ``train.py``: the readout is
``f = W_L phi(h^{L-1}) / (gamma N)`` and the output energy carries a factor
``gamma^2 N``. Consequently the top-layer residual entering the last hidden
layer is ``Delta^L = gamma (y - f)``, i.e. ``C^{Delta,L} = gamma^2 C^{Delta,top}``,
while every hidden-layer memory kernel carries ``eta / P`` with no extra gamma.
Weights are initialised with unit variance, as in ``jpc.make_mlp`` for
``param_type="mupc"``.

Because ``R^phi`` is strictly causal in ``(t, k)`` and ``R^Delta`` is causal up
to and including the current ``(t, k)``, the single-site equations can be
solved *exactly* by sweeping over ``t`` and then ``k`` -- no inner fixed-point
iteration is needed even though the equations are non-linear. The responses are
obtained by forward-mode differentiation of that same sweep.

Flattening convention throughout (shared with ``theory_pc_utils``):
    compound index = (k, t, mu)
with k the slowest block index, so ``n = (K+1) * T * P``.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random
from theory_pc_utils import (
    damp,
    make_delta0_projector,
    make_endpoint_memory_operator,
    make_input_covariance,
    make_response_causality_masks,
    relative_change,
    solve_pc_output_boundary,
    symmetrise,
)


Array = jax.Array


def get_nonlinearity(
    nonlinearity: str = "tanh",
    beta: float = 1.0,
) -> Tuple[Callable[[Array], Array], Callable[[Array], Array]]:
    """Return ``(phi, phi')`` for a supported activation.

    ``beta`` is the inverse temperature/gain of the smooth activations, using
    the same conventions as ``theory_nonlin_utils`` so that ``beta=1`` for
    ``"tanh"`` reproduces ``jnp.tanh``.
    """
    if nonlinearity == "linear":
        return (lambda h: h), (lambda h: jnp.ones_like(h))

    if nonlinearity == "tanh":
        return (
            lambda h: jnp.tanh(beta * h) / beta,
            lambda h: 1.0 - jnp.tanh(beta * h) ** 2,
        )

    if nonlinearity == "relu":
        return (
            lambda h: jnp.maximum(h, 0.0),
            lambda h: (h > 0.0).astype(h.dtype),
        )

    if nonlinearity == "softplus":
        scale = jnp.sqrt(2.0)
        return (
            lambda h: scale * jax.nn.softplus(beta * h) / beta,
            lambda h: scale * jax.nn.sigmoid(beta * h),
        )

    raise ValueError(
        f"Unknown nonlinearity {nonlinearity!r}; expected one of "
        "'linear', 'tanh', 'relu', 'softplus'."
    )


def colour_normal_samples(
    covariance: Array,
    standard_normals: Array,
    jitter: float = 1e-10,
) -> Array:
    """Map iid normals to samples of ``N(0, covariance)``.

    ``standard_normals`` has shape ``(n, num_samples)`` and the returned
    samples have shape ``(num_samples, n)``. Negative eigenvalues (which the
    sampled kernels can acquire) are clipped to zero, so rank-deficient
    covariances such as the lifted input Gram or the rank-``output_dim``
    top-layer residual kernel are handled without special casing.
    """
    cov = symmetrise(jnp.asarray(covariance))
    n = cov.shape[0]
    evals, evecs = jnp.linalg.eigh(cov + jitter * jnp.eye(n, dtype=cov.dtype))
    evals = jnp.maximum(evals, 0.0)
    return (evecs @ (jnp.sqrt(evals)[:, None] * standard_normals)).T


def draw_gaussian_samples(
    covariance: Array,
    key: Array,
    num_samples: int,
    jitter: float = 1e-10,
) -> Array:
    """Draw ``num_samples`` samples of ``N(0, covariance)``; shape ``(S, n)``."""
    cov = jnp.asarray(covariance)
    z = random.normal(key, (cov.shape[0], num_samples), dtype=cov.dtype)
    return colour_normal_samples(cov, z, jitter=jitter)


def initialise_phi_kernels(
    Kx: Array,
    depth: int,
    num_inference_steps: int,
    num_training_steps: int,
    phi: Callable[[Array], Array],
    key: Array,
    num_mc_samples: int,
) -> List[Array]:
    """Static forward-pass ansatz for ``C^{phi,ell}``, ``ell = 1, ..., depth``.

    At initialisation the pre-activations satisfy ``C^{h,1} = Kx`` and
    ``C^{h,ell} = C^{phi,ell-1}``, so the kernels are obtained by iterating
    ``C^phi = <phi(h) phi(h)>`` over Gaussian ``h``. Each ``(P, P)`` kernel is
    then replicated across all ``(k, t)`` pairs.
    """
    kernel = jnp.asarray(Kx)
    sample_kernels = []
    for _ in range(depth):
        key, subkey = random.split(key)
        h = draw_gaussian_samples(kernel, subkey, num_mc_samples)
        activations = phi(h)
        kernel = activations.T @ activations / num_mc_samples
        sample_kernels.append(kernel)

    return [
        make_input_covariance(k, num_inference_steps, num_training_steps)
        for k in sample_kernels
    ]


def make_single_site_solver(
    beta_h: float,
    phi: Callable[[Array], Array],
    dphi: Callable[[Array], Array],
    num_inference_steps: int,
    num_training_steps: int,
    num_samples: int,
) -> Callable[[Array, Array, Array, Array], Tuple[Array, Array, Array, Array]]:
    """Build the exact causal sweep solving one hidden layer's single site.

    The returned callable maps ``(u_chi, u_xi, A_off, B_op)`` to
    ``(h, Delta, z, phi(h))``, where the source arrays have shape ``(..., n)``
    (any leading batch dimensions are allowed) and

        A_off = R^{phi,ell-1} + P,    B_op = R^{Delta,ell+1} + Q

    with ``P`` and ``Q`` the endpoint memory operators. ``A_off`` is strictly
    causal in ``(t, k)`` and ``B_op`` is causal including the current ``(t, k)``,
    so sweeping ``t`` outermost and ``k`` innermost visits every quantity only
    after everything it depends on has been computed. Contracting against the
    partially filled ``Delta`` / ``phi`` buffers is exact because the operators
    vanish on the not-yet-written entries.
    """
    K1 = num_inference_steps + 1
    T = num_training_steps
    P = num_samples

    def solve(u_chi, u_xi, A_off, B_op):
        H = jnp.zeros_like(u_chi)
        Delta = jnp.zeros_like(u_chi)
        Z = jnp.zeros_like(u_chi)
        Phi = jnp.zeros_like(u_chi)

        h_prev = d_prev = g_prev = None
        for t in range(T):
            for k in range(K1):
                base = k * T * P + t * P
                block = slice(base, base + P)
                a_row = A_off[block]
                if k == 0:
                    # Delta_0 = 0 fixes h_0 from the forward pass.
                    h_k = u_chi[..., block] + Delta @ a_row.T
                    d_k = jnp.zeros_like(h_k)
                else:
                    h_k = h_prev + beta_h * (g_prev - d_prev)
                    d_k = h_k - u_chi[..., block] - Delta @ a_row.T
                    Delta = Delta.at[..., block].set(d_k)

                H = H.at[..., block].set(h_k)
                phi_k = phi(h_k)
                Phi = Phi.at[..., block].set(phi_k)

                z_k = u_xi[..., block] + Phi @ B_op[block].T
                Z = Z.at[..., block].set(z_k)

                h_prev, d_prev = h_k, d_k
                g_prev = dphi(h_k) * z_k

        return H, Delta, Z, Phi

    return solve


def single_site_residuals(
    H: Array,
    Delta: Array,
    Z: Array,
    Phi: Array,
    u_chi: Array,
    u_xi: Array,
    A_off: Array,
    B_op: Array,
    beta_h: float,
    dphi: Callable[[Array], Array],
    num_inference_steps: int,
    num_training_steps: int,
    num_samples: int,
) -> Array:
    """Relative residual of the three single-site equations (sanity check)."""
    K1 = num_inference_steps + 1
    T = num_training_steps
    P = num_samples
    shape = (-1, K1, T, P)

    error_res = Delta + Delta @ A_off.T - (H - u_chi)
    backward_res = Z - u_xi - Phi @ B_op.T

    h = H.reshape(shape)
    delta = Delta.reshape(shape)
    g = (dphi(H) * Z).reshape(shape)
    step_res = h[:, 1:] - h[:, :-1] - beta_h * (g[:, :-1] - delta[:, :-1])

    scale = jnp.maximum(1.0, jnp.sqrt(jnp.mean(H**2)))
    return (
        jnp.sqrt(
            jnp.mean(error_res**2) + jnp.mean(backward_res**2) + jnp.mean(step_res**2)
        )
        / scale
    )


def make_hidden_pc_nonlin_layer_solver(
    beta_h: float,
    phi: Callable[[Array], Array],
    dphi: Callable[[Array], Array],
    num_inference_steps: int,
    num_training_steps: int,
    num_samples: int,
    Rphi_mask: Array,
    Rdelta_mask: Array,
    delta_projector: Array,
) -> Callable[..., Dict[str, Array]]:
    """Build a reusable estimator of one hidden layer's kernels.

    The returned callable maps ``(u_chi, u_xi, A_off, B_op)`` to the layer's
    correlation and response functions. The correlations are Monte-Carlo
    averages over all sampled sources; the responses are averages of the
    per-sample Jacobians

        R^phi   = d phi(h) / d u_xi,   R^Delta = d Delta / d u_chi,

    obtained by forward-mode differentiation of the causal sweep and evaluated
    on the first ``num_jacobian_samples`` samples. The Jacobians cost
    ``O(S n^3)`` and dominate the runtime, hence the separate sample budget.

    The compiled pieces are cached in the closure, so the sweep is traced once
    and reused across layers and fixed-point iterations.
    """
    solve = make_single_site_solver(
        beta_h, phi, dphi, num_inference_steps, num_training_steps, num_samples
    )

    @jax.jit
    def correlations(u_chi, u_xi, A_off, B_op):
        H, Delta, Z, Phi = solve(u_chi, u_xi, A_off, B_op)
        total = u_chi.shape[0]
        G = dphi(H) * Z
        residual = single_site_residuals(
            H=H,
            Delta=Delta,
            Z=Z,
            Phi=Phi,
            u_chi=u_chi,
            u_xi=u_xi,
            A_off=A_off,
            B_op=B_op,
            beta_h=beta_h,
            dphi=dphi,
            num_inference_steps=num_inference_steps,
            num_training_steps=num_training_steps,
            num_samples=num_samples,
        )
        return {
            "Cphi": symmetrise(Phi.T @ Phi / total),
            "Cdelta": delta_projector
            @ symmetrise(Delta.T @ Delta / total)
            @ delta_projector,
            "Ch": symmetrise(H.T @ H / total),
            "Cg": symmetrise(G.T @ G / total),
            "equation_residual": residual,
            "max_abs_h": jnp.max(jnp.abs(H)),
        }

    @jax.jit
    def mean_jacobians(u_chi, u_xi, A_off, B_op):
        def phi_of(chi_vec, xi_vec):
            return solve(chi_vec, xi_vec, A_off, B_op)[3]

        def delta_of(chi_vec, xi_vec):
            return solve(chi_vec, xi_vec, A_off, B_op)[1]

        Rphi = jax.vmap(jax.jacfwd(phi_of, argnums=1))(u_chi, u_xi)
        Rdelta = jax.vmap(jax.jacfwd(delta_of, argnums=0))(u_chi, u_xi)
        return jnp.mean(Rphi, axis=0), jnp.mean(Rdelta, axis=0)

    def layer_solver(
        u_chi: Array,
        u_xi: Array,
        A_off: Array,
        B_op: Array,
        num_jacobian_samples: int,
        jacobian_batch_size: int,
    ) -> Dict[str, Array]:
        results = correlations(u_chi, u_xi, A_off, B_op)

        n = u_chi.shape[1]
        num_jacobian_samples = min(num_jacobian_samples, u_chi.shape[0])
        batch = min(jacobian_batch_size, num_jacobian_samples)
        num_batches = num_jacobian_samples // batch

        Rphi = jnp.zeros((n, n), dtype=u_chi.dtype)
        Rdelta = jnp.zeros((n, n), dtype=u_chi.dtype)
        for b in range(num_batches):
            block = slice(b * batch, (b + 1) * batch)
            phi_jac, delta_jac = mean_jacobians(u_chi[block], u_xi[block], A_off, B_op)
            Rphi += phi_jac / num_batches
            Rdelta += delta_jac / num_batches

        results["Rphi"] = Rphi * Rphi_mask
        results["Rdelta"] = delta_projector @ (Rdelta * Rdelta_mask)
        return results

    return layer_solver


def solve_pc_kernels_nonlin(
    Kx: Array,
    y: Array,
    depth: int,
    eta: float,
    gamma: float,
    beta_h: float,
    num_training_steps: int = 20,
    num_inference_steps: int = 10,
    num_fixed_point_steps: int = 15,
    num_mc_samples: int = 1000,
    num_jacobian_samples: Optional[int] = None,
    jacobian_batch_size: int = 25,
    damping: float = 0.5,
    nonlinearity: str = "tanh",
    beta: float = 1.0,
    tolerance: Optional[float] = None,
    cdelta_init_eps: float = 1e-2,
    resample_fields: bool = False,
    seed: int = 0,
) -> Tuple[
    List[Array], List[Array], List[Array], List[Array], Array, Array, Array, dict
]:
    """Solve the non-linear PC DMFT by sampled fixed-point iteration.

    Mirrors ``theory_pc_utils.solve_pc_kernels`` and returns the same tuple, so
    it is a drop-in replacement for plotting and for ``train.py``. With
    ``nonlinearity="linear"`` and enough samples it reproduces the linear
    solver up to Monte-Carlo error.

    **Main arguments:**

    - `Kx`: input Gram matrix ``x_mu . x_nu / D``, shape ``(P, P)``.
    - `y`: targets, shape ``(P,)`` or ``(P, output_dim)``.
    - `depth`: number of hidden layers.
    - `eta`: weight learning rate.
    - `gamma`: muP richness scale; enters only through
      ``C^{Delta,L} = gamma^2 C^{Delta,top}``.
    - `beta_h`: inference (activity) learning rate.

    **Other arguments:**

    - `num_mc_samples`: Monte-Carlo samples for the correlation functions.
    - `num_jacobian_samples`: samples used for the responses (defaults to
      ``min(num_mc_samples, 200)``); these dominate the cost.
    - `jacobian_batch_size`: samples differentiated at once, trading memory
      (``O(batch * n^2)``) for speed.
    - `damping`: fraction of each new estimate mixed into the running kernels.
    - `nonlinearity` / `beta`: activation, see ``get_nonlinearity``.
    - `cdelta_init_eps`: the error kernels start at ``eps I`` (projected onto
      ``Delta_0 = 0``); starting from a replicated output Gram makes the first
      memory operator spuriously large and destabilises the explicit-Euler
      inference sweep.
    - `resample_fields`: if ``True``, redraw the iid normals every iteration
      (stochastic approximation). The default reuses common random numbers,
      which makes the iteration a deterministic map and lets ``tolerance``
      detect convergence.

    **Returns:**

    ``(all_Cphi, all_Cdelta, all_Rphi, all_Rdelta, C_delta_top, loss,
    mean_delta, diagnostics)``.
    """
    if not (0.0 < damping <= 1.0):
        raise ValueError("damping must lie in (0,1].")
    if Kx.ndim != 2 or Kx.shape[0] != Kx.shape[1]:
        raise ValueError("Kx must be a square Gram matrix.")
    if depth < 1:
        raise ValueError("depth must be at least one.")
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be at least one.")
    if cdelta_init_eps < 0.0:
        raise ValueError("cdelta_init_eps must be non-negative.")
    if jacobian_batch_size < 1:
        raise ValueError("jacobian_batch_size must be at least one.")

    P = Kx.shape[0]
    T = num_training_steps
    K = num_inference_steps
    K1 = K + 1
    n = K1 * T * P
    dtype = Kx.dtype

    if num_jacobian_samples is None:
        num_jacobian_samples = min(num_mc_samples, 200)
    num_jacobian_samples = min(num_jacobian_samples, num_mc_samples)
    if num_jacobian_samples < jacobian_batch_size:
        jacobian_batch_size = num_jacobian_samples

    phi, dphi = get_nonlinearity(nonlinearity, beta)

    Rphi_mask, Rdelta_mask = make_response_causality_masks(K, T, P, dtype=dtype)
    delta_projector = make_delta0_projector(K, T, P, dtype=dtype)

    layer_solver = make_hidden_pc_nonlin_layer_solver(
        beta_h=beta_h,
        phi=phi,
        dphi=dphi,
        num_inference_steps=K,
        num_training_steps=T,
        num_samples=P,
        Rphi_mask=Rphi_mask,
        Rdelta_mask=Rdelta_mask,
        delta_projector=delta_projector,
    )

    key = random.PRNGKey(seed)
    key, init_key = random.split(key)

    Cphi0 = make_input_covariance(Kx, K, T)
    zero_op = jnp.zeros((n, n), dtype=dtype)

    all_Cphi = initialise_phi_kernels(Kx, depth, K, T, phi, init_key, num_mc_samples)
    eps_eye = cdelta_init_eps * jnp.eye(n, dtype=dtype)
    all_Cdelta = [delta_projector @ eps_eye @ delta_projector for _ in range(depth)]
    all_Rphi = [zero_op for _ in range(depth)]
    all_Rdelta = [zero_op for _ in range(depth)]

    def draw_normals(rng):
        chi = []
        xi = []
        for _ in range(depth):
            rng, chi_key, xi_key = random.split(rng, 3)
            chi.append(random.normal(chi_key, (n, num_mc_samples), dtype=dtype))
            xi.append(random.normal(xi_key, (n, num_mc_samples), dtype=dtype))
        return rng, chi, xi

    key, z_chi, z_xi = draw_normals(key)

    eta_p = eta / P
    residual_history = []
    equation_history = []
    final_layers: List[Dict[str, Array]] = []
    iteration = 0

    for iteration in range(num_fixed_point_steps):
        old_Cphi, old_Cdelta = all_Cphi, all_Cdelta
        old_Rphi, old_Rdelta = all_Rphi, all_Rdelta

        if resample_fields and iteration > 0:
            key, z_chi, z_xi = draw_normals(key)

        top_result = solve_pc_output_boundary(
            Ch_last=old_Cphi[-1],
            Rh_last=old_Rphi[-1],
            y=y,
            eta=eta,
            num_inference_steps=K,
            num_training_steps=T,
            num_samples=P,
        )
        C_delta_top = top_result["C_delta_top"]

        raw_layers: List[Dict[str, Array]] = []
        for l in range(depth):
            Cphi_minus = Cphi0 if l == 0 else old_Cphi[l - 1]
            Rphi_minus = zero_op if l == 0 else old_Rphi[l - 1]

            if l == depth - 1:
                # Delta^L = gamma (y - f); the top residual response is O(1/N).
                Cdelta_plus = gamma**2 * C_delta_top
                Rdelta_plus = zero_op
            else:
                Cdelta_plus = old_Cdelta[l + 1]
                Rdelta_plus = old_Rdelta[l + 1]

            P_op = make_endpoint_memory_operator(Cphi_minus, K, T, P, eta_p)
            Q_op = make_endpoint_memory_operator(Cdelta_plus, K, T, P, eta_p)

            A_off = Rphi_minus + P_op
            B_op = Rdelta_plus + Q_op

            u_chi = colour_normal_samples(Cphi_minus, z_chi[l])
            u_xi = colour_normal_samples(Cdelta_plus, z_xi[l])

            layer = layer_solver(
                u_chi=u_chi,
                u_xi=u_xi,
                A_off=A_off,
                B_op=B_op,
                num_jacobian_samples=num_jacobian_samples,
                jacobian_batch_size=jacobian_batch_size,
            )
            layer.update({"A_off": A_off, "B_op": B_op, "P": P_op, "Q": Q_op})
            raw_layers.append(layer)

        all_Cphi = [
            symmetrise(damp(old_Cphi[l], raw_layers[l]["Cphi"], damping))
            for l in range(depth)
        ]
        all_Cdelta = [
            delta_projector
            @ symmetrise(damp(old_Cdelta[l], raw_layers[l]["Cdelta"], damping))
            @ delta_projector
            for l in range(depth)
        ]
        all_Rphi = [
            damp(old_Rphi[l], raw_layers[l]["Rphi"], damping) * Rphi_mask
            for l in range(depth)
        ]
        all_Rdelta = [
            delta_projector
            @ (damp(old_Rdelta[l], raw_layers[l]["Rdelta"], damping) * Rdelta_mask)
            for l in range(depth)
        ]

        changes = []
        for old_list, new_list in (
            (old_Cphi, all_Cphi),
            (old_Cdelta, all_Cdelta),
            (old_Rphi, all_Rphi),
            (old_Rdelta, all_Rdelta),
        ):
            changes.extend(relative_change(o, u) for o, u in zip(old_list, new_list))
        fp_residual = jnp.max(jnp.stack(changes))
        residual_history.append(fp_residual)

        equation_residual = jnp.max(
            jnp.stack([layer["equation_residual"] for layer in raw_layers])
        )
        equation_history.append(equation_residual)
        final_layers = raw_layers

        if tolerance is not None and float(fp_residual) < tolerance:
            break

    top_result = solve_pc_output_boundary(
        Ch_last=all_Cphi[-1],
        Rh_last=all_Rphi[-1],
        y=y,
        eta=eta,
        num_inference_steps=K,
        num_training_steps=T,
        num_samples=P,
    )

    diagnostics = {
        "iterations": iteration + 1,
        "fixed_point_residual": residual_history[-1],
        "equation_residual": equation_history[-1],
        "fixed_point_history": jnp.asarray(residual_history),
        "equation_history": jnp.asarray(equation_history),
        "beta_h": beta_h,
        "nonlinearity": nonlinearity,
        "beta": beta,
        "num_mc_samples": num_mc_samples,
        "num_jacobian_samples": num_jacobian_samples,
        "cdelta_init_eps": cdelta_init_eps,
        "Rphi_causality_mask": Rphi_mask,
        "Rdelta_causality_mask": Rdelta_mask,
        "delta0_projector": delta_projector,
        "all_Ch": [layer["Ch"] for layer in final_layers],
        "all_Cg": [layer["Cg"] for layer in final_layers],
        "max_abs_h": jnp.max(jnp.stack([l_["max_abs_h"] for l_ in final_layers])),
        "R_delta_top": top_result["R_delta_top"] * Rdelta_mask,
        "layers": final_layers,
    }

    return (
        all_Cphi,
        all_Cdelta,
        all_Rphi,
        all_Rdelta,
        top_result["C_delta_top"],
        top_result["loss"],
        top_result["mean_delta"],
        diagnostics,
    )
