"""Utilities for deterministic linear predictive-coding DMFT.

Uses the complete inference trajectory k=0,...,K with the forward-pass
boundary condition

    Delta_0^ell(t) = 0.

Two mathematically equivalent solvers are available via
``solve_pc_kernels(..., backend=...)``:

* ``"reference"`` — assembles and factors the full square block system

      [-I,                         A] [h    ]   [-u_chi]
      [D_raw - beta_h S B,  beta_h S] [Delta] = [beta_h S u_xi]
      [ 0,                        E0]           [   0   ]

  where D_raw is the rectangular forward-difference operator on h_0,...,h_K
  (entries +/-1), S selects k=0,...,K-1, and E0 selects the k=0 error
  components. Multiplying the inference block by beta_h is equivalent to the
  textbook form with D = D_raw / beta_h, but avoids mixing O(1/beta_h) and O(1)
  scales in the same matrix.

* ``"optimised"`` (default) — same fixed point, damping and eight-element
  return, but reorganised so the dominant dense linear algebra is much
  cheaper (see below).

Where the optimised speedup comes from
--------------------------------------
1. Static block elimination of h and of Delta_0.
   The textbook form is the square 2n x 2n system above, with
   n = (K+1)TP. Its first row block gives h = u_chi + A Delta exactly
   and its last row block gives Delta_0 = 0 exactly, so both can be
   substituted out before any factorisation. What remains is a system of
   size m = K T P in the unknowns Delta_1,...,Delta_K,

       G Delta_{1:K} = beta_h S u_xi - M u_chi,
       M = D_raw - beta_h S B,   G = (M A + beta_h S)[:, k>=1],

   from which the transfer matrices follow as

       T_delta_chi = G^{-1} (-M),       T_delta_xi = beta_h [G^{-1} | 0],
       T_h_chi     = I + A T_delta_chi, T_h_xi     = A T_delta_xi,

   padded with the structural zeros they are known to have. The LU drops
   from size 2n to size m = K/(K+1) n, i.e. (2(K+1)/K)^3 ~ 14x fewer
   factorisation flops at K=5, and the triangular-solve stage by
   (2(K+1)/K)^2 ~ 6x. Peak memory drops roughly threefold because the
   2n x 2n system, its right-hand side and the stacked transfer matrix
   are never formed.

2. No separate solve for the u_xi source.
   The u_xi source is exactly beta_h [I_m | 0], so T_delta_xi is beta_h
   G^{-1} with a zero k=K column block appended. One triangular-solve
   stage against [-M | beta_h I_m] therefore yields both transfer
   matrices, and T_h follows from the single matmul A[:, k>=1] @ X.

3. Structural operators are applied, never multiplied.
     S X         -> X[:m]                       (was an m x n by n x n matmul)
     D_raw X     -> X[1:] - X[:-1] on block rows (was a matmul)
     E0 X        -> X[:block]
     Proj X Proj -> drop the k=0 block row/column
   In the reference backend, ``S @ B`` plus the four
   ``delta_projector @ ... @ delta_projector`` sandwiches cost
   ~12 n^3 flops per layer per iteration; here they are free.

4. Structurally zero blocks are dropped from the covariance updates.
   C^Delta has a zero k=0 row/column by construction, so only its m x m
   block is propagated; T_*_xi has a zero k=K column block, so the u_xi
   contractions run over m rather than n.

5. The output boundary no longer inverts A_top inside the loop.
   Only C^{Delta,top} = (A_top^{-1} y)(A_top^{-1} y)^T is needed per
   iteration, i.e. a solve with ``output_dim`` right-hand sides instead
   of n. The full inverse is formed once, after the loop, for the
   R^{Delta,top} diagnostic.

6. The equation residual is measured on the reduced system, so it costs
   ~2.5 n^3 instead of the 16 n^3 needed to multiply the 2n x 2n system
   by its transfer blocks. ``check_equations=False`` skips it entirely.

7. One XLA program per fixed-point iteration.
   The whole sweep (output boundary, every layer, damping, masking,
   convergence diagnostics) is a single ``jax.jit`` region, so the many
   O(n^2) elementwise passes fuse into a handful of kernels instead of
   dozens of separate dispatches, and there is no host synchronisation
   unless ``tolerance`` is set. With ``batch_layers=True`` the ``depth``
   layers — independent given the previous iterate, since the sweep is
   Jacobi-style — are evaluated by one ``vmap``, turning ``depth`` small
   factorisations into a single batched one.

Together these cut the flop count by ~3x, shrink the factorisation by
~14x, lower peak memory by ~3x, and collapse the kernel-launch count.

Flattening convention throughout:
    compound index = (k, t, mu)
with k the slowest block index.
"""

from __future__ import annotations

from functools import partial
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

Array = jax.Array


def _state_size(K: int, T: int, P: int) -> int:
    return (K + 1) * T * P


# ---------------------------------------------------------------------------
# Shared structural operators and helpers.
# ---------------------------------------------------------------------------


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
    eta: float,
) -> Array:
    r"""Construct the strictly training-time-causal endpoint operator.

    [T[X] v]_{k,t,mu}
      = eta * sum_{s<t,nu} X_{k,K;mu,nu}(t,s) v_{K,s,nu}.

    Both input and output live on the full k=0,...,K state space.
    Only the k'=K column block is nonzero, so the operator is assembled by
    concatenation instead of scattering into a zero buffer.
    """
    K = num_inference_steps
    K1 = K + 1
    T = num_training_steps
    P = num_samples
    n = K1 * T * P

    X = jnp.asarray(covariance).reshape(K1, T, P, K1, T, P)
    causal_t = jnp.tril(jnp.ones((T, T), dtype=X.dtype), k=-1)
    endpoint = X[:, :, :, K, :, :] * causal_t[None, :, None, :, None]

    op = jnp.concatenate(
        [
            jnp.zeros((K1, T, P, K, T, P), dtype=X.dtype),
            (eta * endpoint)[:, :, :, None, :, :],
        ],
        axis=3,
    )
    return op.reshape(n, n)


def _causal_masks_kt(
    num_inference_steps: int, num_training_steps: int
) -> Tuple[Array, Array]:
    """Boolean (K1,T,K1,T) causality masks, before the sample broadcast.

    Keeping the sample axes implicit lets XLA fuse the mask into the
    select instead of materialising a dense (n,n) mask per use.
    """
    K1 = num_inference_steps + 1
    T = num_training_steps
    k = jnp.arange(K1)[:, None, None, None]
    t = jnp.arange(T)[None, :, None, None]
    kp = jnp.arange(K1)[None, None, :, None]
    tp = jnp.arange(T)[None, None, None, :]
    past_time = tp < t
    same_time = tp == t
    return past_time | (same_time & (kp < k)), past_time | (same_time & (kp <= k))


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
    mask_h_kt, mask_d_kt = _causal_masks_kt(num_inference_steps, num_training_steps)

    n = K1 * T * P
    mask_h = jnp.broadcast_to(
        mask_h_kt[:, :, None, :, :, None],
        (K1, T, P, K1, T, P),
    )
    mask_d = jnp.broadcast_to(
        mask_d_kt[:, :, None, :, :, None],
        (K1, T, P, K1, T, P),
    )
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


# ---------------------------------------------------------------------------
# Reference backend: full 2n x 2n block system.
# ---------------------------------------------------------------------------


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
    num_inference_steps: int,
    num_training_steps: int,
    num_samples: int,
    normalise_outputs: bool = False,
    compute_response: bool = True,
) -> Dict[str, Array]:
    """Solve the top residual process on the full k=0,...,K space.

    This retains the original output-boundary convention
        (I + Rh_last + P_top) Delta_top = y - u_chi.
    The hidden-layer forward-pass constraint Delta_0=0 is not imposed on
    the output residual unless the model's output boundary requires it.

    Also returns the boundary residual response
    R^{Delta,top} = d Delta_top / d u_chi = -A_top^{-1}
    when ``compute_response=True``. With ``compute_response=False`` only the
    ``output_dim`` columns needed for C^{Delta,top} are solved, and
    ``R_delta_top`` is omitted.

    The reported loss is evaluated at k=0, i.e. on the initial forward-pass
    prediction error, before any inference-step correction.
    """
    K1 = num_inference_steps + 1
    T = num_training_steps
    P = num_samples
    n = K1 * T * P
    dtype = Ch_last.dtype

    P_top = make_endpoint_memory_operator(
        Ch_last,
        num_inference_steps,
        num_training_steps,
        num_samples,
        eta / P,
    )
    A_top = jnp.eye(n, dtype=dtype) + Rh_last + P_top

    y_flat = lift_targets(y, num_inference_steps, num_training_steps)
    output_dim = y_flat.shape[1]

    if compute_response:
        T_top = jnp.linalg.solve(A_top, jnp.eye(n, dtype=dtype))
        mean_delta_flat = T_top @ y_flat
    else:
        mean_delta_flat = jnp.linalg.solve(A_top, y_flat)

    C_delta_top = mean_delta_flat @ mean_delta_flat.T
    if normalise_outputs:
        C_delta_top = C_delta_top / output_dim

    mean_delta = mean_delta_flat.reshape(K1, T, P, output_dim)
    mean_squared_error = jnp.sum(mean_delta[0] ** 2, axis=(1, 2))
    loss = 0.5 * mean_squared_error / P

    result = {
        "mean_delta": mean_delta,
        "mean_delta_flat": mean_delta_flat,
        "C_delta_top": symmetrise(C_delta_top),
        "loss": loss,
        "A_top": A_top,
        "P_top": P_top,
    }
    if compute_response:
        result["R_delta_top"] = -T_top
    return result


def _solve_pc_kernels_reference(
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
    """Reference fixed-point solver using the full 2n x 2n block system."""
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
            num_inference_steps=K,
            num_training_steps=T,
            num_samples=P,
        )
        C_delta_top = top_result["C_delta_top"]

        raw_layers: List[Dict[str, Array]] = []
        for l in range(depth):
            Ch_minus = Ch0 if l == 0 else old_Ch[l - 1]
            Rh_minus = Rh0 if l == 0 else old_Rh[l - 1]

            if l == depth - 1:
                Cdelta_plus = gamma**2 * C_delta_top
                Rdelta_plus = jnp.zeros((n, n), dtype=dtype)
            else:
                Cdelta_plus = old_Cdelta[l + 1]
                Rdelta_plus = old_Rdelta[l + 1]

            P_op = make_endpoint_memory_operator(Ch_minus, K, T, P, eta_p)
            Q_op = make_endpoint_memory_operator(Cdelta_plus, K, T, P, eta_p)

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
        "backend": "reference",
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


# ---------------------------------------------------------------------------
# Optimised backend: reduced m x m Delta system + jitted Jacobi sweep.
# ---------------------------------------------------------------------------


def _apply_causal_mask(X: Array, mask_kt: Array, K1: int, T: int, P: int) -> Array:
    """Apply a (K1,T,K1,T) causality mask to a flattened (n,n) matrix."""
    n = K1 * T * P
    return jnp.where(
        mask_kt[:, :, None, :, :, None], X.reshape(K1, T, P, K1, T, P), 0.0
    ).reshape(n, n)


def _project_delta0(X: Array, block: int, columns: bool = True) -> Array:
    """Zero the k=0 block row (and optionally block column) of X."""
    X = X.at[:block].set(0.0)
    if columns:
        X = X.at[:, :block].set(0.0)
    return X


def _block_diff_rows(X: Array, K1: int, block: int) -> Array:
    """D_raw @ X: forward differences of the block rows of X."""
    Xb = X.reshape(K1, block, -1)
    return (Xb[1:] - Xb[:-1]).reshape((K1 - 1) * block, -1)


def _solve_hidden_layer_reduced(
    A: Array,
    B: Array,
    Ch_minus: Array,
    Cdelta_plus: Array,
    beta: Array,
    K: int,
    block: int,
    check_equations: bool = True,
) -> Dict[str, Array]:
    """Exact hidden-layer saddle point via the reduced m x m Delta system.

    Solves the same equations as ``solve_hidden_pc_layer`` after eliminating
    h = u_chi + A Delta and Delta_0 = 0 analytically. Returns unmasked
    transfer matrices; causal masking is the caller's job.
    """
    K1 = K + 1
    n = K1 * block
    m = K * block
    dtype = A.dtype
    eye_block = jnp.eye(block, dtype=dtype)
    kk = jnp.arange(K)[:, None]

    # Delta only couples to columns k>=1 of A, and only rows k<=K-1 of B
    # enter the inference equation (that is what S B selects).
    A_cols = A[:, block:]
    B_top = B[:m]

    # G = (D_raw A - beta (S B) A + beta S)[:, k>=1]. Restricted to columns
    # k>=1, beta S becomes beta times the subdiagonal identity blocks.
    sub = (kk == jnp.arange(K)[None, :] + 1).astype(dtype)
    G = _block_diff_rows(A_cols, K1, block) - beta * (B_top @ A_cols)
    G = (
        G.reshape(K, block, K, block)
        + beta * sub[:, None, :, None] * eye_block[None, :, None, :]
    ).reshape(m, m)

    # u_chi source: -M = -D_raw + beta S B.
    kp = jnp.arange(K1)[None, :]
    neg_D = (kk == kp).astype(dtype) - (kk + 1 == kp).astype(dtype)
    src_chi = (
        beta * B_top.reshape(K, block, K1, block)
        + neg_D[:, None, :, None] * eye_block[None, :, None, :]
    ).reshape(m, n)

    # The u_xi source is beta [I_m | 0]; the zero column block is appended
    # after the solve rather than carried through it.
    rhs = jnp.concatenate([src_chi, beta * jnp.eye(m, dtype=dtype)], axis=1)
    X = jnp.linalg.solve(G, rhs)
    Td_chi = X[:, :n]
    Td_xi = X[:, n:]

    # T_h = I + A T_delta, sharing one matmul between both sources.
    TH = A_cols @ X
    T_h_chi = TH[:, :n] + jnp.eye(n, dtype=dtype)
    T_h_xi_top = TH[:, n:]

    # T_*_xi has a zero k'=K column block, so contract u_xi over m only.
    Cd_plus = Cdelta_plus[:m, :m]
    Ch = T_h_chi @ Ch_minus @ T_h_chi.T + T_h_xi_top @ Cd_plus @ T_h_xi_top.T
    Cdelta_low = (Td_chi @ Ch_minus) @ Td_chi.T + (Td_xi @ Cd_plus) @ Td_xi.T

    out = {
        "Ch": symmetrise(Ch),
        # The k=0 row/column block is structurally zero, so the Delta_0
        # projector is applied by padding rather than by two matmuls.
        "Cdelta": jnp.pad(symmetrise(Cdelta_low), ((block, 0), (block, 0))),
        "T_h_chi": T_h_chi,
        "T_h_xi": jnp.pad(T_h_xi_top, ((0, 0), (0, block))),
        "T_delta_chi": jnp.pad(Td_chi, ((block, 0), (0, 0))),
        "T_delta_xi": jnp.pad(Td_xi, ((block, 0), (0, block))),
        "system_reduced": G,
        "J_chi_reduced": src_chi,
    }

    if check_equations:
        rhs_xi = beta * jnp.eye(m, dtype=dtype)
        res_chi = jnp.linalg.norm(G @ Td_chi - src_chi) / jnp.maximum(
            1.0, jnp.linalg.norm(src_chi)
        )
        res_xi = jnp.linalg.norm(G @ Td_xi - rhs_xi) / jnp.maximum(
            1.0, jnp.linalg.norm(rhs_xi)
        )
        out["equation_residual"] = jnp.maximum(res_chi, res_xi)
    else:
        out["equation_residual"] = jnp.zeros((), dtype=dtype)

    return out


@partial(
    jax.jit,
    static_argnames=(
        "depth",
        "K",
        "T",
        "P",
        "batch_layers",
        "check_equations",
        "return_layers",
    ),
)
def _fixed_point_iteration(
    Ch: Array,
    Cdelta: Array,
    Rh: Array,
    Rdelta: Array,
    Ch0: Array,
    y_flat: Array,
    beta: Array,
    gamma: Array,
    eta_p: Array,
    damping: Array,
    *,
    depth: int,
    K: int,
    T: int,
    P: int,
    batch_layers: bool,
    check_equations: bool,
    return_layers: bool,
) -> Tuple[Array, Array, Array, Array, Array, Array, Dict[str, Array]]:
    """One damped Jacobi sweep over all layers, as a single XLA program.

    Kernel state is stacked over layers with leading axis ``depth``.
    """
    K1 = K + 1
    block = T * P
    n = K1 * block
    dtype = Ch.dtype
    zero = jnp.zeros((1, n, n), dtype=dtype)
    I_n = jnp.eye(n, dtype=dtype)

    # Output boundary: only C^{Delta,top} is needed here, so solve against
    # y (output_dim columns) instead of forming A_top^{-1}.
    A_top = I_n + Rh[-1] + make_endpoint_memory_operator(Ch[-1], K, T, P, eta_p)
    mean_delta_top = jnp.linalg.solve(A_top, y_flat)
    C_delta_top = symmetrise(mean_delta_top @ mean_delta_top.T)

    # Every layer reads only the previous iterate, so the layers are
    # mutually independent and may be batched.
    Ch_minus = jnp.concatenate([Ch0[None], Ch[:-1]], axis=0)
    Rh_minus = jnp.concatenate([zero, Rh[:-1]], axis=0)
    Cdelta_plus = jnp.concatenate([Cdelta[1:], (gamma**2 * C_delta_top)[None]], axis=0)
    Rdelta_plus = jnp.concatenate([Rdelta[1:], zero], axis=0)

    mask_h_kt, mask_d_kt = _causal_masks_kt(K, T)

    def layer(Ch_m, Rh_m, Cd_p, Rd_p):
        P_op = make_endpoint_memory_operator(Ch_m, K, T, P, eta_p)
        Q_op = make_endpoint_memory_operator(Cd_p, K, T, P, eta_p)
        A = I_n + Rh_m + P_op
        B = Rd_p + Q_op
        out = _solve_hidden_layer_reduced(
            A, B, Ch_m, Cd_p, beta, K, block, check_equations
        )
        out["Rh"] = _apply_causal_mask(out["T_h_xi"], mask_h_kt, K1, T, P)
        out["Rdelta"] = _apply_causal_mask(out["T_delta_chi"], mask_d_kt, K1, T, P)
        if return_layers:
            out.update({"A": A, "B": B, "P": P_op, "Q": Q_op})
        else:
            for key in (
                "T_h_chi",
                "T_h_xi",
                "T_delta_chi",
                "T_delta_xi",
                "system_reduced",
                "J_chi_reduced",
            ):
                del out[key]
        return out

    if batch_layers:
        raw = jax.vmap(layer)(Ch_minus, Rh_minus, Cdelta_plus, Rdelta_plus)
    else:
        per_layer = [
            layer(Ch_minus[l], Rh_minus[l], Cdelta_plus[l], Rdelta_plus[l])
            for l in range(depth)
        ]
        raw = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *per_layer)

    # Damping, symmetrisation, causal masking and the Delta_0 projection
    # are all elementwise here, so XLA fuses them into single passes.
    new_Ch = jax.vmap(symmetrise)(damp(Ch, raw["Ch"], damping))
    new_Cdelta = jax.vmap(lambda X: _project_delta0(X, block))(
        jax.vmap(symmetrise)(damp(Cdelta, raw["Cdelta"], damping))
    )
    new_Rh = jax.vmap(_apply_causal_mask, in_axes=(0, None, None, None, None))(
        damp(Rh, raw["Rh"], damping), mask_h_kt, K1, T, P
    )
    new_Rdelta = jax.vmap(lambda X: _project_delta0(X, block, columns=False))(
        jax.vmap(_apply_causal_mask, in_axes=(0, None, None, None, None))(
            damp(Rdelta, raw["Rdelta"], damping), mask_d_kt, K1, T, P
        )
    )

    def rel(old, new):
        num = jnp.linalg.norm((new - old).reshape(depth, -1), axis=1)
        den = jnp.maximum(1.0, jnp.linalg.norm(old.reshape(depth, -1), axis=1))
        return num / den

    fp_residual = jnp.max(
        jnp.concatenate(
            [
                rel(Ch, new_Ch),
                rel(Cdelta, new_Cdelta),
                rel(Rh, new_Rh),
                rel(Rdelta, new_Rdelta),
            ]
        )
    )
    equation_residual = jnp.max(raw["equation_residual"])

    return new_Ch, new_Cdelta, new_Rh, new_Rdelta, fp_residual, equation_residual, raw


def _solve_pc_kernels_optimised(
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
    *,
    batch_layers: Optional[bool] = None,
    check_equations: bool = True,
    return_layer_diagnostics: bool = True,
    return_operators: bool = True,
) -> Tuple[List[Array], List[Array], List[Array], List[Array], Array, Array, Array, dict]:
    """Optimised fixed-point solver (reduced Delta system + jitted sweep).

    Drop-in replacement for the reference backend: same equations, same
    damping, same eight-element return. See the module docstring section
    "Where the optimised speedup comes from" for the full write-up
    (block elimination of h / Delta_0, structural slicing, dropped zero
    blocks, cheap output boundary, reduced residual, single jitted
    Jacobi sweep with optional layer ``vmap``).

    Extra keyword-only options
    --------------------------
    batch_layers
        Evaluate all ``depth`` layers with one ``vmap``, i.e. a single
        batched factorisation instead of ``depth`` small ones. Layers
        read only the previous iterate, so this is exact. Defaults to
        True when the extra depth-fold working set is small
        (``depth * n^2 * itemsize < 2 GiB``), else False.
    check_equations
        Measure the relative residual of the reduced linear system every
        iteration (~10% overhead). If False, ``equation_residual`` is 0.
    return_layer_diagnostics
        Keep the final iteration's per-layer matrices in
        ``diagnostics["layers"]``. False saves ~10 n^2 per layer.
    return_operators
        Materialise D_raw, S, E0, the causality masks and the Delta_0
        projector in the diagnostics, as the reference backend does.
        False avoids ~5 n^2 of arrays the solver never needs.

    Notes
    -----
    ``diagnostics["equation_residual"]`` measures the residual of the
    reduced m x m Delta system rather than the 2n x 2n block system, so
    its value is not numerically comparable with the reference residual
    (it is a residual for a better-conditioned system, hence typically
    smaller).
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
    block = T * P
    n = _state_size(K, T, P)
    dtype = Kx.dtype

    if batch_layers is None:
        itemsize = jnp.zeros((), dtype=dtype).itemsize
        batch_layers = depth > 1 and depth * n * n * itemsize < 2 * 1024**3

    Ch0 = make_input_covariance(Kx, K, T)
    y_flat = lift_targets(y, K, T)

    eps_diag = jnp.concatenate(
        [
            jnp.zeros(block, dtype=dtype),
            jnp.full(K * block, cdelta_init_eps, dtype=dtype),
        ]
    )
    Ch = jnp.stack([sigma ** (2 * (l + 1)) * Ch0 for l in range(depth)])
    Cdelta = jnp.broadcast_to(jnp.diag(eps_diag), (depth, n, n))
    Rh = jnp.zeros((depth, n, n), dtype=dtype)
    Rdelta = jnp.zeros((depth, n, n), dtype=dtype)

    beta = jnp.asarray(beta_h, dtype=dtype)
    gamma_a = jnp.asarray(gamma, dtype=dtype)
    eta_p = jnp.asarray(eta / P, dtype=dtype)
    damping_a = jnp.asarray(damping, dtype=dtype)

    residual_history: List[Array] = []
    equation_history: List[Array] = []
    raw_layers: Dict[str, Array] = {}
    iteration = -1

    for iteration in range(num_fixed_point_steps):
        (
            Ch,
            Cdelta,
            Rh,
            Rdelta,
            fp_residual,
            equation_residual,
            raw_layers,
        ) = _fixed_point_iteration(
            Ch,
            Cdelta,
            Rh,
            Rdelta,
            Ch0,
            y_flat,
            beta,
            gamma_a,
            eta_p,
            damping_a,
            depth=depth,
            K=K,
            T=T,
            P=P,
            batch_layers=batch_layers,
            check_equations=check_equations,
            return_layers=return_layer_diagnostics,
        )
        residual_history.append(fp_residual)
        equation_history.append(equation_residual)

        # The only host synchronisation in the loop, and only if asked for.
        if tolerance is not None and float(fp_residual) < tolerance:
            break

    top_result = solve_pc_output_boundary(
        Ch_last=Ch[-1],
        Rh_last=Rh[-1],
        y=y,
        eta=eta,
        num_inference_steps=K,
        num_training_steps=T,
        num_samples=P,
        compute_response=True,
    )

    _, mask_d_kt = _causal_masks_kt(K, T)
    diagnostics = {
        "iterations": iteration + 1,
        "fixed_point_residual": residual_history[-1],
        "equation_residual": equation_history[-1],
        "fixed_point_history": jnp.asarray(residual_history),
        "equation_history": jnp.asarray(equation_history),
        "beta_h": beta_h,
        "cdelta_init_eps": cdelta_init_eps,
        "R_delta_top": _apply_causal_mask(
            top_result["R_delta_top"], mask_d_kt, K + 1, T, P
        ),
        "batch_layers": batch_layers,
        "layers": [],
        "backend": "optimised",
    }

    if return_operators:
        D_raw, S, E0 = make_pc_operators(K, T, P, dtype=dtype)
        Rh_mask, Rdelta_mask = make_response_causality_masks(K, T, P, dtype=dtype)
        diagnostics.update(
            {
                "D_raw": D_raw,
                "S": S,
                "E0": E0,
                "Rh_causality_mask": Rh_mask,
                "Rdelta_causality_mask": Rdelta_mask,
                "delta0_projector": make_delta0_projector(K, T, P, dtype=dtype),
            }
        )

    if return_layer_diagnostics and raw_layers:
        diagnostics["layers"] = [
            jax.tree_util.tree_map(lambda x: x[l], raw_layers) for l in range(depth)
        ]

    return (
        list(Ch),
        list(Cdelta),
        list(Rh),
        list(Rdelta),
        top_result["C_delta_top"],
        top_result["loss"],
        top_result["mean_delta"],
        diagnostics,
    )


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


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
    *,
    backend: str = "optimised",
    batch_layers: Optional[bool] = None,
    check_equations: bool = True,
    return_layer_diagnostics: bool = True,
    return_operators: bool = True,
) -> Tuple[List[Array], List[Array], List[Array], List[Array], Array, Array, Array, dict]:
    """Solve the boundary-conditioned linear PC DMFT by fixed-point iteration.

    Hidden layers use the exact block equations with Delta_0=0. Responses
    are causally masked after each raw update and before they are used in
    the next fixed-point iteration.

    The output residual covariance C^{Delta,top} is refreshed every iteration
    via solve_pc_output_boundary and enters the last hidden layer as
    C^{Delta,ell+1} = gamma^2 C^{Delta,top}. The top residual response is
    not fed back: R^{Delta,ell+1} = 0 for the last hidden layer.

    Error kernels are initialised to eps * I (projected to Delta_0=0), matching
    the algorithm text. Initialising from a replicated output Gram matrix
    makes the first-iterate memory operator Q spuriously large and drives
    the explicit-Euler block resolvent unstable at moderate beta_h / K.

    Parameters
    ----------
    backend
        ``"optimised"`` (default) uses the reduced Delta system and a jitted
        Jacobi sweep. ``"reference"`` assembles the full 2n x 2n block
        system each layer (slower; useful for debugging / residual checks).
    batch_layers, check_equations, return_layer_diagnostics, return_operators
        Optimised-backend-only options; ignored when ``backend="reference"``.
    """
    common = dict(
        Kx=Kx,
        y=y,
        depth=depth,
        eta=eta,
        gamma=gamma,
        beta_h=beta_h,
        num_training_steps=num_training_steps,
        num_inference_steps=num_inference_steps,
        num_fixed_point_steps=num_fixed_point_steps,
        damping=damping,
        sigma=sigma,
        tolerance=tolerance,
        cdelta_init_eps=cdelta_init_eps,
    )
    if backend == "reference":
        return _solve_pc_kernels_reference(**common)
    if backend == "optimised":
        return _solve_pc_kernels_optimised(
            **common,
            batch_layers=batch_layers,
            check_equations=check_equations,
            return_layer_diagnostics=return_layer_diagnostics,
            return_operators=return_operators,
        )
    raise ValueError(
        f"unknown backend {backend!r}; expected 'optimised' or 'reference'"
    )
