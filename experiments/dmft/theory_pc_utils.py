"""Utilities for deterministic linear predictive-coding DMFT.

This revision keeps the complete inference trajectory k=0,...,K and
implements the forward-pass boundary condition through

    Delta_0^ell(t) = 0.

For each hidden layer the linear single-site equations are solved as the
square block system

    [-I,                         A] [h    ]   [-u_chi]
    [D_raw - beta_h S B,  beta_h S] [Delta] = [beta_h S u_xi]
    [ 0,                        E0]           [   0   ]

where D_raw is the rectangular forward-difference operator on h_0,...,h_K
(entries +/-1), S selects k=0,...,K-1, and E0 selects the k=0 error
components. Multiplying the inference block by beta_h is equivalent to the
textbook form with F = D_raw / beta_h, but avoids mixing O(1/beta_h) and O(1)
scales in the same matrix.

Flattening convention throughout:
    compound index = (k, t, mu)
with k the slowest block index.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

Array = jax.Array


def _state_size(K: int, T: int, P: int) -> int:
    return (K + 1) * T * P


def _update_size(K: int, T: int, P: int) -> int:
    return K * T * P


def make_pc_operators(
    num_inference_steps: int,
    num_training_steps: int,
    num_samples: int,
    dtype=jnp.float64,
) -> Tuple[Array, Array, Array]:
    """Construct unscaled D_raw, S and E0 for the full trajectory h_0,...,h_K.

    D_raw implements the forward difference
        (D_raw h)_k = h_{k+1} - h_k,
        k=0,...,K-1.

    The inference learning rate beta_h is applied when assembling the block
    system in solve_hidden_pc_layer, as the equivalent row-scaled form

        (D_raw - beta_h S B) h + beta_h S Delta = beta_h S u_xi,

    which is algebraically identical to (F - S B) h + S Delta = S u_xi with
    D = D_raw / beta_h, but keeps matrix entries O(1) instead of mixing
    O(1/beta_h) difference terms with O(1) feedback.

    S selects k=0,...,K-1 from a full state vector.
    E0 selects k=0 from a full error vector.

    Shapes
    ------
    D_raw : (K*T*P, (K+1)*T*P)
    S     : (K*T*P, (K+1)*T*P)
    E0    : (T*P,   (K+1)*T*P)
    """
    K = num_inference_steps
    T = num_training_steps
    P = num_samples
    block = T * P

    Dk = jnp.zeros((K, K + 1), dtype=dtype)
    rows = jnp.arange(K)
    Dk = Dk.at[rows, rows].set(-1.0)
    Dk = Dk.at[rows, rows + 1].set(1.0)

    Sk = jnp.concatenate(
        [jnp.eye(K, dtype=dtype), jnp.zeros((K, 1), dtype=dtype)], axis=1
    )

    E0k = jnp.zeros((1, K + 1), dtype=dtype)
    E0k = E0k.at[0, 0].set(1.0)

    I_block = jnp.eye(block, dtype=dtype)
    return (
        jnp.kron(Dk, I_block),
        jnp.kron(Sk, I_block),
        jnp.kron(E0k, I_block),
    )


def make_input_covariance(
    Kx: Array,
    num_inference_steps: int,
    num_training_steps: int,
) -> Array:
    """Lift Kx[mu,nu] to the full k=0,...,K state space."""
    P = Kx.shape[0]
    K1 = num_inference_steps + 1
    T = num_training_steps
    C0 = jnp.broadcast_to(
        Kx[None, None, :, None, None, :],
        (K1, T, P, K1, T, P),
    )
    return C0.reshape(K1 * T * P, K1 * T * P)


def lift_targets(y: Array, num_inference_steps: int, num_training_steps: int) -> Array:
    """Lift y[mu,c] to y[k,t,mu,c] for k=0,...,K."""
    y = jnp.asarray(y)
    if y.ndim == 1:
        y = y[:, None]
    P, output_dim = y.shape
    K1 = num_inference_steps + 1
    T = num_training_steps
    y_lifted = jnp.broadcast_to(y[None, None, :, :], (K1, T, P, output_dim))
    return y_lifted.reshape(K1 * T * P, output_dim)


def make_endpoint_memory_operator(
    covariance: Array,
    num_inference_steps: int,
    num_training_steps: int,
    num_samples: int,
    eta_gamma: float,
) -> Array:
    r"""Construct the strictly training-time-causal endpoint operator.

    [T[X] v]_{k,t,mu}
      = eta_gamma * sum_{s<t,nu} X_{k,K;mu,nu}(t,s) v_{K,s,nu}.

    Both input and output live on the full k=0,...,K state space.
    """
    K = num_inference_steps
    K1 = K + 1
    T = num_training_steps
    P = num_samples

    X = jnp.asarray(covariance).reshape(K1, T, P, K1, T, P)
    endpoint = X[:, :, :, K, :, :]
    causal_t = jnp.tril(jnp.ones((T, T), dtype=X.dtype), k=-1)
    endpoint = endpoint * causal_t[None, :, None, :, None]

    op = jnp.zeros((K1, T, P, K1, T, P), dtype=X.dtype)
    op = op.at[:, :, :, K, :, :].set(eta_gamma * endpoint)
    n = K1 * T * P
    return op.reshape(n, n)


def make_response_causality_masks(
    num_inference_steps: int,
    num_training_steps: int,
    num_samples: int,
    dtype=jnp.float64,
) -> Tuple[Array, Array]:
    """Return masks for R^h and R^Delta.

    Causal ordering is training time first, then inference time:

      R^h_{k,t ; k',t'} may be nonzero when
          t' < t, or t'=t and k' < k.

      R^Delta_{k,t ; k',t'} may be nonzero when
          t' < t, or t'=t and k' <= k.

    Sample indices are unrestricted by causality.
    """
    K1 = num_inference_steps + 1
    T = num_training_steps
    P = num_samples

    k = jnp.arange(K1)[:, None, None, None]
    t = jnp.arange(T)[None, :, None, None]
    kp = jnp.arange(K1)[None, None, :, None]
    tp = jnp.arange(T)[None, None, None, :]

    past_time = tp < t
    same_time = tp == t
    mask_h_kt = past_time | (same_time & (kp < k))
    mask_d_kt = past_time | (same_time & (kp <= k))

    # Expand sample pairs and reorder to (k,t,mu,k',t',nu).
    mask_h = jnp.broadcast_to(
        mask_h_kt[:, :, None, :, :, None],
        (K1, T, P, K1, T, P),
    )
    mask_d = jnp.broadcast_to(
        mask_d_kt[:, :, None, :, :, None],
        (K1, T, P, K1, T, P),
    )

    n = K1 * T * P
    return mask_h.reshape(n, n).astype(dtype), mask_d.reshape(n, n).astype(dtype)


def make_delta0_projector(
    num_inference_steps: int,
    num_training_steps: int,
    num_samples: int,
    dtype=jnp.float64,
) -> Array:
    """Diagonal projector onto the allowed Delta subspace k>=1."""
    K1 = num_inference_steps + 1
    T = num_training_steps
    P = num_samples
    mask = jnp.ones((K1, T, P), dtype=dtype)
    mask = mask.at[0].set(0.0)
    return jnp.diag(mask.reshape(-1))


def symmetrise(matrix: Array) -> Array:
    return 0.5 * (matrix + matrix.T)


def damp(old: Array, candidate: Array, damping: float) -> Array:
    return (1.0 - damping) * old + damping * candidate


def relative_change(old: Array, new: Array) -> Array:
    return jnp.linalg.norm(new - old) / jnp.maximum(1.0, jnp.linalg.norm(old))


def solve_hidden_pc_layer(
    A: Array,
    B: Array,
    Ch_minus: Array,
    Cdelta_plus: Array,
    D_raw: Array,
    S: Array,
    E0: Array,
    beta_h: float,
    Rh_mask: Array,
    Rdelta_mask: Array,
    delta_projector: Array,
) -> Dict[str, Array]:
    """Solve one hidden-layer linear PC saddle point exactly.

    Implements the explicit-Euler PC step

        (h_{k+1} - h_k) = beta_h (g_k - Delta_k),
        g = u_xi + B h,

    together with A Delta = h - u_chi and Delta_0 = 0, using the
    beta_h-row-scaled block system (equivalent to D = D_raw / beta_h):

        [-I,                         A] [h    ]   [-u_chi]
        [D_raw - beta_h S B,  beta_h S] [Delta] = [beta_h S u_xi]
        [0,                         E0]           [0]

    u_chi and u_xi are independent with covariances Ch_minus and
    Cdelta_plus respectively.
    """
    n = A.shape[0]
    m = D_raw.shape[0]
    b = E0.shape[0]
    dtype = A.dtype
    beta = jnp.asarray(beta_h, dtype=dtype)

    I_n = jnp.eye(n, dtype=dtype)

    system = jnp.block(
        [
            [-I_n, A],
            [D_raw - beta * (S @ B), beta * S],
            [jnp.zeros((b, n), dtype=dtype), E0],
        ]
    )

    # Source injection matrices for full u_chi and u_xi vectors.
    J_chi = jnp.concatenate(
        [-I_n, jnp.zeros((m, n), dtype=dtype), jnp.zeros((b, n), dtype=dtype)],
        axis=0,
    )
    J_xi = jnp.concatenate(
        [
            jnp.zeros((n, n), dtype=dtype),
            beta * S,
            jnp.zeros((b, n), dtype=dtype),
        ],
        axis=0,
    )

    rhs = jnp.concatenate([J_chi, J_xi], axis=1)
    transfer = jnp.linalg.solve(system, rhs)

    T_chi = transfer[:, :n]
    T_xi = transfer[:, n:]

    T_h_chi = T_chi[:n]
    T_delta_chi = T_chi[n:]
    T_h_xi = T_xi[:n]
    T_delta_xi = T_xi[n:]

    Ch = (
        T_h_chi @ Ch_minus @ T_h_chi.T
        + T_h_xi @ Cdelta_plus @ T_h_xi.T
    )
    Cdelta = (
        T_delta_chi @ Ch_minus @ T_delta_chi.T
        + T_delta_xi @ Cdelta_plus @ T_delta_xi.T
    )

    Rh = T_h_xi * Rh_mask
    Rdelta = T_delta_chi * Rdelta_mask

    Ch = symmetrise(Ch)
    Cdelta = delta_projector @ symmetrise(Cdelta) @ delta_projector
    Rdelta = delta_projector @ Rdelta

    return {
        "Ch": Ch,
        "Cdelta": Cdelta,
        "Rh": Rh,
        "Rdelta": Rdelta,
        "system": system,
        "J_chi": J_chi,
        "J_xi": J_xi,
        "T_h_chi": T_h_chi,
        "T_h_xi": T_h_xi,
        "T_delta_chi": T_delta_chi,
        "T_delta_xi": T_delta_xi,
    }


def solve_pc_output_boundary(
    Ch_last: Array,
    Rh_last: Array,
    y: Array,
    eta: float,
    gamma: float,
    num_inference_steps: int,
    num_training_steps: int,
    num_samples: int,
    source_covariance: Optional[Array] = None,
    normalise_outputs: bool = False,
) -> Dict[str, Array]:
    """Solve the top residual process on the full k=0,...,K space.

    This retains the original output-boundary convention
        (I + Rh_last + P_top) Delta_top = y - u_chi.
    The hidden-layer forward-pass constraint Delta_0=0 is not imposed on
    the output residual unless the model's output boundary requires it.

    The response of this boundary residual to its own bottom-up drive,
    R^{Delta,top} = d Delta_top / d u_chi = -A_top^{-1}, is returned so the
    caller can feed it back self-consistently as R^{Delta,ell+1} for the
    last hidden layer (see solve_pc_kernels).

    The reported loss is evaluated at k=0, i.e. on the initial forward-pass
    prediction error, before any inference-step correction.
    """
    K1 = num_inference_steps + 1
    T = num_training_steps
    P = num_samples
    n = K1 * T * P

    I = jnp.eye(n, dtype=Ch_last.dtype)
    P_top = make_endpoint_memory_operator(
        Ch_last,
        num_inference_steps,
        num_training_steps,
        num_samples,
        eta / P,
    )
    A_top = I + Rh_last + P_top

    y_flat = lift_targets(y, num_inference_steps, num_training_steps)
    output_dim = y_flat.shape[1]

    T_top = jnp.linalg.solve(A_top, I)
    mean_delta_flat = T_top @ y_flat
    R_delta_top = -T_top

    C_delta_top = mean_delta_flat @ mean_delta_flat.T
    if normalise_outputs:
        C_delta_top = C_delta_top / output_dim

    mean_delta = mean_delta_flat.reshape(K1, T, P, output_dim)

    k0_mean = mean_delta[0]

    mean_squared_error = jnp.sum(k0_mean**2, axis=(1, 2))
    loss = 0.5 * (mean_squared_error) / P

    return {
        "mean_delta": mean_delta,
        "mean_delta_flat": mean_delta_flat,
        "C_delta_top": symmetrise(C_delta_top),
        "R_delta_top": R_delta_top,
        "loss": loss,
        "A_top": A_top,
        "P_top": P_top,
    }


# def solve_pc_output_boundary(
#     Ch_last: Array,
#     Rh_last: Array,
#     y: Array,
#     eta: float,
#     gamma: float,
#     num_inference_steps: int,
#     num_training_steps: int,
#     num_samples: int,
#     source_covariance: Optional[Array] = None,
#     normalise_outputs: bool = False,
# ) -> Dict[str, Array]:
#     """Solve the top residual process on the full k=0,...,K space.
#     NOTE: Old version that assumes 1/sqrt(N) at the output

#     This retains the original output-boundary convention
#         (I + Rh_last + P_top) Delta_top = y - u_chi.
#     The hidden-layer forward-pass constraint Delta_0=0 is not imposed on
#     the output residual unless the model's output boundary requires it.

#     The response of this boundary residual to its own bottom-up drive,
#     R^{Delta,top} = d Delta_top / d u_chi = -A_top^{-1}, is returned so the
#     caller can feed it back self-consistently as R^{Delta,ell+1} for the
#     last hidden layer (see solve_pc_kernels).

#     The reported loss is evaluated at k=0, i.e. on the initial forward-pass
#     prediction error, before any inference-step correction.
#     """
#     K1 = num_inference_steps + 1
#     T = num_training_steps
#     P = num_samples
#     n = K1 * T * P

#     if source_covariance is None:
#         source_covariance = Ch_last

#     I = jnp.eye(n, dtype=Ch_last.dtype)
#     P_top = make_endpoint_memory_operator(
#         Ch_last,
#         num_inference_steps,
#         num_training_steps,
#         num_samples,
#         eta * gamma / P,
#     )
#     A_top = I + Rh_last + P_top

#     y_flat = lift_targets(y, num_inference_steps, num_training_steps)
#     output_dim = y_flat.shape[1]

#     T_top = jnp.linalg.solve(A_top, I)
#     mean_delta_flat = T_top @ y_flat
#     R_delta_top = -T_top
#     centred_covariance = symmetrise(T_top @ source_covariance @ T_top.T)

#     mean_second_moment = mean_delta_flat @ mean_delta_flat.T
#     covariance_multiplier = output_dim
#     if normalise_outputs:
#         mean_second_moment = mean_second_moment / output_dim
#         covariance_multiplier = 1.0

#     C_delta_top = mean_second_moment + covariance_multiplier * centred_covariance
#     mean_delta = mean_delta_flat.reshape(K1, T, P, output_dim)

#     k0_mean = mean_delta[0]
#     sigma = centred_covariance.reshape(K1, T, P, K1, T, P)
#     k0_variance = jnp.einsum("tmtm->tm", sigma[0, :, :, 0, :, :])

#     mean_squared_error = jnp.sum(k0_mean**2, axis=(1, 2))
#     variance_error = covariance_multiplier * jnp.sum(k0_variance, axis=1)
#     # loss = 0.5 * (mean_squared_error + variance_error) / P
#     loss = 0.5 * (mean_squared_error) / P

#     return {
#         "mean_delta": mean_delta,
#         "mean_delta_flat": mean_delta_flat,
#         "centred_covariance": centred_covariance,
#         "C_delta_top": symmetrise(C_delta_top),
#         "R_delta_top": R_delta_top,
#         "loss": loss,
#         "A_top": A_top,
#         "P_top": P_top,
#     }


def solve_pc_kernels(
    Kx: Array,
    y: Array,
    depth: int,
    eta: float,
    gamma: float,
    beta_h: float,
    num_training_steps: int = 100,
    num_inference_steps: int = 10,
    num_fixed_point_steps: int = 25,
    damping: float = 0.1,
    sigma: float = 1.0,
    tolerance: Optional[float] = None,
    cdelta_init_eps: float = 1e-2,
) -> Tuple[List[Array], List[Array], List[Array], List[Array], Array, Array, Array, dict]:
    """Solve the boundary-conditioned linear PC DMFT by fixed-point iteration.

    Hidden layers use the exact block equations with Delta_0=0. Responses
    are causally masked after each raw update and before they are used in
    the next fixed-point iteration.

    The output boundary's own response R^{Delta,top} = -A_top^{-1} is
    solved for self-consistently every iteration (via
    solve_pc_output_boundary) and fed back as R^{Delta,ell+1} for the last
    hidden layer, rather than being supplied externally.

    Error kernels are initialised to eps * I (projected to Delta_0=0), matching
    the algorithm text. Initialising from a replicated output Gram matrix
    makes the first-iterate memory operator Q spuriously large and drives
    the explicit-Euler block resolvent unstable at moderate beta_h / K.
    """
    if not (0.0 < damping <= 1.0):
        raise ValueError("damping must lie in (0,1].")
    if Kx.ndim != 2 or Kx.shape[0] != Kx.shape[1]:
        raise ValueError("Kx must be a square Gram matrix.")
    if depth < 1:
        raise ValueError("depth must be at least one.")
    if cdelta_init_eps < 0.0:
        raise ValueError("cdelta_init_eps must be non-negative.")

    P = Kx.shape[0]
    T = num_training_steps
    K = num_inference_steps
    n = _state_size(K, T, P)
    dtype = Kx.dtype

    D_raw, S, E0 = make_pc_operators(K, T, P, dtype=dtype)
    Rh_mask, Rdelta_mask = make_response_causality_masks(K, T, P, dtype=dtype)
    delta_projector = make_delta0_projector(K, T, P, dtype=dtype)

    Ch0 = make_input_covariance(Kx, K, T)
    Rh0 = jnp.zeros((n, n), dtype=dtype)

    all_Ch = [sigma ** (2 * (l + 1)) * Ch0 for l in range(depth)]
    # Section 12: C^Delta_(0) = eps I (not a replicated output Gram).
    eps_eye = cdelta_init_eps * jnp.eye(n, dtype=dtype)
    all_Cdelta = [
        delta_projector @ eps_eye @ delta_projector for _ in range(depth)
    ]

    all_Rh = [jnp.zeros((n, n), dtype=dtype) for _ in range(depth)]
    all_Rdelta = [jnp.zeros((n, n), dtype=dtype) for _ in range(depth)]

    eta_p = eta / P
    residual_history = []
    equation_history = []
    final_layers: List[Dict[str, Array]] = []

    for iteration in range(num_fixed_point_steps):
        old_Ch = all_Ch
        old_Cdelta = all_Cdelta
        old_Rh = all_Rh
        old_Rdelta = all_Rdelta

        top_result = solve_pc_output_boundary(
            Ch_last=old_Ch[-1],
            Rh_last=old_Rh[-1],
            y=y,
            eta=eta,
            gamma=gamma,
            num_inference_steps=K,
            num_training_steps=T,
            num_samples=P,
        )
        C_delta_top = top_result["C_delta_top"]
        R_delta_top = top_result["R_delta_top"] * Rdelta_mask

        raw_layers: List[Dict[str, Array]] = []
        for l in range(depth):
            Ch_minus = Ch0 if l == 0 else old_Ch[l - 1]
            Rh_minus = Rh0 if l == 0 else old_Rh[l - 1]

            if l == depth - 1:
                Cdelta_plus = (gamma ** 2) * C_delta_top
                Rdelta_plus = jnp.zeros_like(R_delta_top)
            else:
                Cdelta_plus = old_Cdelta[l + 1]
                Rdelta_plus = old_Rdelta[l + 1]

            P_op = make_endpoint_memory_operator(
                Ch_minus, K, T, P, eta_p
            )
            Q_op = make_endpoint_memory_operator(
                Cdelta_plus, K, T, P, eta_p
            )

            A = jnp.eye(n, dtype=dtype) + Rh_minus + P_op
            B = Rdelta_plus + Q_op

            layer = solve_hidden_pc_layer(
                A=A,
                B=B,
                Ch_minus=Ch_minus,
                Cdelta_plus=Cdelta_plus,
                D_raw=D_raw,
                S=S,
                E0=E0,
                beta_h=beta_h,
                Rh_mask=Rh_mask,
                Rdelta_mask=Rdelta_mask,
                delta_projector=delta_projector,
            )
            layer.update({"A": A, "B": B, "P": P_op, "Q": Q_op})
            raw_layers.append(layer)

        all_Ch = [
            symmetrise(damp(old_Ch[l], raw_layers[l]["Ch"], damping))
            for l in range(depth)
        ]
        all_Cdelta = [
            delta_projector
            @ symmetrise(damp(old_Cdelta[l], raw_layers[l]["Cdelta"], damping))
            @ delta_projector
            for l in range(depth)
        ]
        all_Rh = [
            damp(old_Rh[l], raw_layers[l]["Rh"], damping) * Rh_mask
            for l in range(depth)
        ]
        all_Rdelta = [
            delta_projector
            @ (damp(old_Rdelta[l], raw_layers[l]["Rdelta"], damping) * Rdelta_mask)
            for l in range(depth)
        ]

        changes = []
        for old_list, new_list in (
            (old_Ch, all_Ch),
            (old_Cdelta, all_Cdelta),
            (old_Rh, all_Rh),
            (old_Rdelta, all_Rdelta),
        ):
            changes.extend(relative_change(o, n_) for o, n_ in zip(old_list, new_list))
        fp_residual = jnp.max(jnp.stack(changes))
        residual_history.append(fp_residual)

        eq_residuals = []
        for layer in raw_layers:
            system = layer["system"]
            J_chi = layer["J_chi"]
            J_xi = layer["J_xi"]
            Tchi = jnp.concatenate([layer["T_h_chi"], layer["T_delta_chi"]], axis=0)
            Txi = jnp.concatenate([layer["T_h_xi"], layer["T_delta_xi"]], axis=0)
            eq_residuals.append(
                jnp.linalg.norm(system @ Tchi - J_chi)
                / jnp.maximum(1.0, jnp.linalg.norm(J_chi))
            )
            eq_residuals.append(
                jnp.linalg.norm(system @ Txi - J_xi)
                / jnp.maximum(1.0, jnp.linalg.norm(J_xi))
            )
        equation_residual = jnp.max(jnp.stack(eq_residuals))
        equation_history.append(equation_residual)
        final_layers = raw_layers

        if tolerance is not None and float(fp_residual) < tolerance:
            break

    top_result = solve_pc_output_boundary(
        Ch_last=all_Ch[-1],
        Rh_last=all_Rh[-1],
        y=y,
        eta=eta,
        gamma=gamma,
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
        "D_raw": D_raw,
        "S": S,
        "E0": E0,
        "beta_h": beta_h,
        "cdelta_init_eps": cdelta_init_eps,
        "Rh_causality_mask": Rh_mask,
        "Rdelta_causality_mask": Rdelta_mask,
        "delta0_projector": delta_projector,
        "R_delta_top": top_result["R_delta_top"] * Rdelta_mask,
        "layers": final_layers,
    }

    return (
        all_Ch,
        all_Cdelta,
        all_Rh,
        all_Rdelta,
        top_result["C_delta_top"],
        top_result["loss"],
        top_result["mean_delta"],
        diagnostics,
    )
