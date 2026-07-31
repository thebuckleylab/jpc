"""
Utilities to compute PC DMFT quantities. 
"""

from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp

Array = jax.Array


def make_inference_difference(
    num_inference_steps: int,
    beta_h: float,
    num_training_steps: int,
    num_samples: int,
    dtype=jnp.float64,
) -> Array:
    """
    Construct D after eliminating the fixed initial state h_0.

    Flattening convention:
        compound index = (k, t, mu)

    with free inference states
        k  = 1, ..., K
        t  = 0, ..., T-1
        mu = 0, ..., P-1

    Discrete inference (explicit SGD on activities) is

        h_{k} = h_{k-1} - β_h ∇_h E(h_{k-1}) ,

    which rearranges to the backward difference

        (D h)_k := (h_k - h_{k-1}) / β_h = - ∇_h E(h_{k-1}) .

    On the free block (h_1, ..., h_K) this is the lower-bidiagonal

        D_k = (1/β_h) * [[ 1, 0, ...],
                         [-1, 1, ...],
                         [ 0,-1, ...]],

    with the known h_0 term moved into the source (hence eliminated
    from D itself). D is Kronecker-extended as identity over (t, μ).

    Relation to the schematic forward form
    ``(h_{k+1} - h_k)/β_h`` in the PC DMFT writeup: that is the same
    Euler step written one index later. For the free vector
    (h_1,...,h_K), the backward matrix above is the natural square
    reduction after fixing h_0; a naive forward matrix on the same
    free vector is upper-bidiagonal and leaves the last row without an
    h_{K+1}. With J = B - D, using backward D is consistent with
    writing the stationarity condition as D h + (force from B) = 0
    on the updated states. If your derivation defines D as forward
    on (h_0,...,h_{K-1}) instead, the matrix here would need to change
    in tandem with how J enters the saddle equations.

    Returns
    -------
    D : (K*T*P, K*T*P)
    """
    K = num_inference_steps
    T = num_training_steps
    P = num_samples

    Dk = jnp.eye(K, dtype=dtype)

    if K > 1:
        Dk = Dk - jnp.eye(K, k=-1, dtype=dtype)

    Dk = Dk / beta_h

    # Flattening is (k, t, mu), so each k block has size T*P.
    I_tp = jnp.eye(T * P, dtype=dtype)

    return jnp.kron(Dk, I_tp)


def make_input_covariance(
    Kx: Array,
    num_inference_steps: int,
    num_training_steps: int,
) -> Array:
    """
    Lift the input Gram matrix Kx[mu, nu] to a covariance over
    compound indices (k, t, mu).

    The default assumes that the input field is unchanged across
    inference steps and training times:

        C0[k,t,mu,k',t',nu] = Kx[mu,nu].

    Returns
    -------
    C0 : (K*T*P, K*T*P)
    """
    P = Kx.shape[0]
    K = num_inference_steps
    T = num_training_steps

    C0 = jnp.broadcast_to(
        Kx[None, None, :, None, None, :],
        (K, T, P, K, T, P),
    )

    return C0.reshape(K * T * P, K * T * P)


def lift_targets(y, num_inference_steps, num_training_steps):
    """
    Lift y[mu, output] to y[k, t, mu, output].

    Parameters
    ----------
    y
        Shape (P,) or (P, output_dim).

    Returns
    -------
    y_flat
        Shape (K*T*P, output_dim).
    """
    y = jnp.asarray(y)

    if y.ndim == 1:
        y = y[:, None]

    P, output_dim = y.shape
    K = num_inference_steps
    T = num_training_steps

    y_lifted = jnp.broadcast_to(
        y[None, None, :, :],
        (K, T, P, output_dim),
    )

    return y_lifted.reshape(K * T * P, output_dim)


def make_endpoint_memory_operator(
    covariance,
    num_inference_steps,
    num_training_steps,
    num_samples,
    eta_gamma,
):
    r"""
    Construct the endpoint-memory operator

        T[X]_{(k,t,mu),(k',s,nu)}
          = 1_{s<t} X_{k,K;mu,nu}(t,s) delta_{k',K}.

    Parameters
    ----------
    covariance
        Shape:
            (K*T*P, K*T*P)
        or:
            (K, T, P, K, T, P).

    num_inference_steps
        Number of represented inference states, K.

    num_training_steps
        Number of physical training times, T.

    num_samples
        Number of samples, P.

    eta_gamma
        Prefactor for the memory operator. Callers should pass
        ``eta * gamma / P`` to match the backpropagation DMFT.

    Returns
    -------
    operator
        Shape (K*T*P, K*T*P).

    Flattening convention
    ---------------------
        index = ((k * T) + t) * P + mu
    """
    K = num_inference_steps
    T = num_training_steps
    P = num_samples

    X = jnp.asarray(covariance).reshape(K, T, P, K, T, P)

    # Select only the endpoint on the second inference index:
    #
    # endpoint[k,t,mu,s,nu] = X[k,t,mu,K-1,s,nu].
    endpoint = X[:, :, :, -1, :, :]

    # Strict physical-time causality: include only s < t.
    causal_mask = jnp.tril(
        jnp.ones((T, T), dtype=X.dtype),
        k=-1,
    )

    endpoint = endpoint * causal_mask[None, :, None, :, None]

    # Construct:
    #
    # operator[k,t,mu,k_prime,s,nu]
    #
    # with nonzero entries only at k_prime = K-1.
    operator = jnp.zeros(
        (K, T, P, K, T, P),
        dtype=X.dtype,
    )

    operator = operator.at[:, :, :, -1, :, :].set(
        eta_gamma * endpoint
    )

    return operator.reshape(K * T * P, K * T * P)



def two_sided_solve(operator: Array, source: Array) -> Array:
    """
    Return operator^{-1} source operator^{-T}
    without explicitly constructing an inverse.
    """
    left_solved = jnp.linalg.solve(operator, source)

    result = jnp.linalg.solve(
        operator,
        left_solved.T,
    ).T

    return result


def symmetrise(matrix: Array) -> Array:
    """
    Symmetrise a matrix.
    """
    return 0.5 * (matrix + matrix.T)


def damp(old: Array, candidate: Array, damping: float) -> Array:
    """
    damping = 1 gives the raw fixed-point update.
    damping < 1 gives under-relaxation.
    """
    return (1.0 - damping) * old + damping * candidate


def solve_pc_output_boundary(
    Ch_last,
    Rh_last,
    y,
    eta,
    gamma,
    num_inference_steps,
    num_training_steps,
    num_samples,
    source_covariance=None,
    normalise_outputs=False,
):
    r"""
    Solve the top-layer single-site equation

        (I + Rh_last + P_top) Delta_top = y - u_chi

    and construct the uncentred second moment C_delta_top.

    Parameters
    ----------
    Ch_last
        C^{h,L}, shape (K*T*P, K*T*P).

    Rh_last
        R^{h,L}, shape (K*T*P, K*T*P).

    y
        Targets, shape (P,) or (P, output_dim).

    eta, gamma
        DMFT parameters.

    num_inference_steps
        Number of represented inference states.

    num_training_steps
        Number of physical training times.

    num_samples
        Number of samples.

    source_covariance
        C^{u_chi,L+1}. If None, use Ch_last.

    normalise_outputs
        If True, average rather than sum over output coordinates.

    Returns
    -------
    result
        Dictionary containing:
          - mean_delta
          - centred_covariance
          - C_delta_top
          - loss
          - A_top
          - P_top
    """
    K = num_inference_steps
    T = num_training_steps
    P = num_samples
    n = K * T * P

    if source_covariance is None:
        source_covariance = Ch_last

    identity = jnp.eye(n, dtype=Ch_last.dtype)

    # P^{L+1} = (η γ / P) T[C^{h,L}], matching the BP DMFT 1/P factor.
    P_top = make_endpoint_memory_operator(
        covariance=Ch_last,
        num_inference_steps=K,
        num_training_steps=T,
        num_samples=P,
        eta_gamma=eta * gamma / P,
    )

    A_top = identity + Rh_last + P_top

    # y[k,t,mu,c].
    y_flat = lift_targets(
        y=y,
        num_inference_steps=K,
        num_training_steps=T,
    )

    output_dim = y_flat.shape[1]

    # Mean residual:
    #
    #     m_delta = A_top^{-1} y.
    mean_delta_flat = jnp.linalg.solve(
        A_top,
        y_flat,
    )

    # Centred covariance for one output coordinate:
    #
    #     Sigma_delta =
    #       A_top^{-1} C_u A_top^{-T}.
    centred_covariance = two_sided_solve(
        A_top,
        source_covariance,
    )

    centred_covariance = 0.5 * (
        centred_covariance + centred_covariance.T
    )

    # Mean contribution summed over output coordinates:
    #
    # mean_second_moment[a,b]
    #     = sum_c m[a,c] m[b,c].
    mean_second_moment = (
        mean_delta_flat @ mean_delta_flat.T
    )

    # If each output coordinate has an independent effective source
    # with the same covariance, covariance contributions add.
    covariance_multiplier = output_dim

    if normalise_outputs:
        mean_second_moment = mean_second_moment / output_dim
        covariance_multiplier = 1.0

    C_delta_top = (
        mean_second_moment
        + covariance_multiplier * centred_covariance
    )

    # Reshape the mean to (K, T, P, output_dim).
    mean_delta = mean_delta_flat.reshape(
        K, T, P, output_dim
    )

    # Extract endpoint mean.
    endpoint_mean = mean_delta[-1]  # (T, P, output_dim)

    # Extract endpoint variance:
    #
    # Sigma[K,t,mu,K,t,mu].
    sigma_tensor = centred_covariance.reshape(
        K, T, P, K, T, P
    )

    endpoint_variance = jnp.einsum(
        "tmtm->tm",
        sigma_tensor[-1, :, :, -1, :, :],
    )

    mean_squared_error = jnp.sum(
        endpoint_mean**2,
        axis=(1, 2),
    )

    variance_error = (
        covariance_multiplier
        * jnp.sum(endpoint_variance, axis=1)
    )

    loss = (
        0.5
        * (mean_squared_error) #  + variance_error)
        / P
    )

    return {
        "mean_delta": mean_delta,
        "mean_delta_flat": mean_delta_flat,
        "centred_covariance": centred_covariance,
        "C_delta_top": C_delta_top,
        "loss": loss,
        "A_top": A_top,
        "P_top": P_top,
    }


# def loss_from_top_correlation(
#     C_delta_top,
#     num_inference_steps,
#     num_training_steps,
#     num_samples,
# ):
#     """
#     Extract

#         L(t) = (1 / 2P)
#                sum_mu C_delta_top[K,t,mu,K,t,mu].
#     """
#     K = num_inference_steps
#     T = num_training_steps
#     P = num_samples

#     C = C_delta_top.reshape(
#         K, T, P, K, T, P
#     )

#     endpoint_block = C[-1, :, :, -1, :, :]

#     endpoint_diagonal = jnp.diagonal(
#         endpoint_block.reshape(T * P, T * P)
#     ).reshape(T, P)

#     return 0.5 * jnp.sum(endpoint_diagonal, axis=1) / P

# # Optionally Ch_delta_top can be computed in a separate step.
# top = solve_pc_output_boundary(...)

# loss_1 = top["loss"]

# loss_2 = loss_from_top_correlation(
#     C_delta_top=top["C_delta_top"],
#     num_inference_steps=K,
#     num_training_steps=T,
#     num_samples=P,
# )


def solve_pc_kernels(
    Kx: Array,
    y: Array,
    depth: int,
    eta: float,
    gamma: float,
    beta_h: float,
    R_delta_top: Optional[Array] = None,
    num_training_steps: int = 100,
    num_inference_steps: int = 10,
    num_fixed_point_steps: int = 10,
    damping: float = 1.0,
    sigma: float = 1.0,
    tolerance: Optional[float] = None,
) -> Tuple[
    List[Array],
    List[Array],
    List[Array],
    List[Array],
    Array,
    Array,
    Array,
    dict,
]:
    r"""
    Solve the linear predictive-coding DMFT fixed-point equations

        A_l = I + R_h,l-1 + P_l
        B_l = R_Delta,l+1 + Q_l
        J_l = B_l - D

        M_l = I - A_l J_l
        N_l = I - J_l A_l

        C_h,l =
            M_l^{-1}
            [C_h,l-1 + A_l C_Delta,l+1 A_l^T]
            M_l^{-T}

        C_Delta,l =
            N_l^{-1}
            [C_Delta,l+1 + J_l C_h,l-1 J_l^T]
            N_l^{-T}

        R_h,l     = M_l^{-1} A_l
        R_Delta,l = N_l^{-1} J_l.

    The fixed inference initial state h_0 is eliminated from D.

    Memory operators use (η γ / P), matching the 1/P factor in the
    backpropagation DMFT solver (``theory_utils.solve_kernels``).

    Parameters
    ----------
    Kx
        Input Gram matrix with shape (P, P).

    depth
        Number of hidden PC layers represented by the DMFT recursion.

    eta, gamma
        Learning-rate and feature-scale parameters.

    beta_h
        Inference-step size.

    R_delta_top
        Top boundary response R^{Delta,L+1}. If None, it is set to zero.

    num_training_steps
        Number of training-time points T.

    num_inference_steps
        Number of unknown inference states after eliminating h_0.
        These correspond to h_1, ..., h_K.

    num_fixed_point_steps
        Maximum number of outer self-consistency iterations.

    damping
        Fixed-point damping coefficient in (0,1].

    sigma
        Scale used only to initialise the layer covariances.

    tolerance
        Optional relative convergence tolerance. Setting this to None
        runs exactly num_fixed_point_steps iterations.

    Returns
    -------
    all_Ch
        List [C^{h,1}, ..., C^{h,L}].

    all_Cdelta
        List [C^{Delta,1}, ..., C^{Delta,L}].

    all_Rh
        List [R^{h,1}, ..., R^{h,L}].

    all_Rdelta
        List [R^{Delta,1}, ..., R^{Delta,L}].

    C_delta_top
        Top-boundary error correlation C^{Delta,L+1}.

    pc_training_loss
        DMFT training loss of shape (T,).

    mean_delta_top
        Mean top residual of shape (K, T, P, output_dim).

    diagnostics
        Fixed-point residual, equation residual, and derived operators.
    """
    if not (0.0 < damping <= 1.0):
        raise ValueError("damping must lie in (0, 1].")

    if Kx.ndim != 2 or Kx.shape[0] != Kx.shape[1]:
        raise ValueError("Kx must be a square (P, P) Gram matrix.")

    P = Kx.shape[0]
    T = num_training_steps
    K = num_inference_steps
    n = K * T * P

    dtype = Kx.dtype
    I = jnp.eye(n, dtype=dtype)

    # Match BP DMFT: learning-rate factors carry the empirical 1/P.
    eta_gamma = eta * gamma / P
    final_residual = jnp.inf
    final_equation_residual = jnp.inf

    # Saved only for diagnostics.
    final_A = [None] * depth
    final_B = [None] * depth
    final_J = [None] * depth
    final_M = [None] * depth
    final_N = [None] * depth

    # ------------------------------------------------------------
    # Boundary operators
    # ------------------------------------------------------------
    D = make_inference_difference(
        num_inference_steps=K,
        beta_h=beta_h,
        num_training_steps=T,
        num_samples=P,
        dtype=dtype,
    )

    Ch0 = make_input_covariance(
        Kx=Kx,
        num_inference_steps=K,
        num_training_steps=T,
    )

    # Initial guess for the top correlation (overwritten each outer
    # iteration by the boundary solve, analogous to get_Delta in BP).
    y_flat = lift_targets(y, K, T)
    C_delta_top = y_flat @ y_flat.T

    # Input boundary response.
    Rh0 = jnp.zeros((n, n), dtype=dtype)

    if R_delta_top is None:
        R_delta_top = jnp.zeros((n, n), dtype=dtype)
    else:
        R_delta_top = jnp.asarray(
            R_delta_top,
            dtype=dtype,
        ).reshape(n, n)


    # ------------------------------------------------------------
    # Initial kernel guesses
    # ------------------------------------------------------------

    all_Ch = [
        sigma ** (2 * (l + 1)) * Ch0
        for l in range(depth)
    ]

    all_Cdelta = [
        sigma ** (2 * (depth - l)) * C_delta_top
        for l in range(depth)
    ]

    all_Rh = [
        jnp.zeros((n, n), dtype=dtype)
        for _ in range(depth)
    ]

    all_Rdelta = [
        jnp.zeros((n, n), dtype=dtype)
        for _ in range(depth)
    ]

    # ------------------------------------------------------------
    # Outer self-consistency iteration
    # ------------------------------------------------------------

    for iteration in range(num_fixed_point_steps):
        old_Ch = all_Ch
        old_Cdelta = all_Cdelta
        old_Rh = all_Rh
        old_Rdelta = all_Rdelta

        candidate_Ch = []
        candidate_Cdelta = []
        candidate_Rh = []
        candidate_Rdelta = []

        candidate_A = []
        candidate_B = []
        candidate_J = []
        candidate_M = []
        candidate_N = []
        neighbour_Ch_minus = []
        neighbour_Cdelta_plus = []

        # Derived top boundary from current kernels — same role as
        # Delta = get_Delta(all_H, all_G, ...) in the BP solver.
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
        C_delta_top = top_result["C_delta_top"]

        # Jacobi in the layer kernels: neighbours come from the
        # previous outer iteration. The freshly computed C_delta_top
        # is used immediately (as BP uses the new Delta).
        for l in range(depth):
            Ch_minus = Ch0 if l == 0 else old_Ch[l - 1]
            Rh_minus = Rh0 if l == 0 else old_Rh[l - 1]

            if l == depth - 1:
                Cdelta_plus = C_delta_top
                Rdelta_plus = R_delta_top
            else:
                Cdelta_plus = old_Cdelta[l + 1]
                Rdelta_plus = old_Rdelta[l + 1]

            neighbour_Ch_minus.append(Ch_minus)
            neighbour_Cdelta_plus.append(Cdelta_plus)

            # P^ℓ = (η γ / P) ∑_{t'<t} C^{h,ℓ-1}_{k,K}(t,t')  (linear: φ = h)
            # Q^ℓ = (η γ / P) ∑_{t'<t} C^{Δ,ℓ+1}_{k,K}(t,t')
            P_op = make_endpoint_memory_operator(
                covariance=Ch_minus,
                num_inference_steps=K,
                num_training_steps=T,
                num_samples=P,
                eta_gamma=eta_gamma,
            )

            Q_op = make_endpoint_memory_operator(
                covariance=Cdelta_plus,
                num_inference_steps=K,
                num_training_steps=T,
                num_samples=P,
                eta_gamma=eta_gamma,
            )

            A = I + Rh_minus + P_op
            B = Rdelta_plus + Q_op
            J = B - D

            M = I - A @ J
            N = I - J @ A

            # Response updates.
            Rh_raw = jnp.linalg.solve(M, A)
            Rdelta_raw = jnp.linalg.solve(N, J)

            # Covariance-source terms.
            source_h = (
                Ch_minus
                + A @ Cdelta_plus @ A.T
            )

            source_delta = (
                Cdelta_plus
                + J @ Ch_minus @ J.T
            )

            Ch_raw = two_sided_solve(M, source_h)
            Cdelta_raw = two_sided_solve(N, source_delta)

            # Covariances should be symmetric.
            Ch_raw = symmetrise(Ch_raw)
            Cdelta_raw = symmetrise(Cdelta_raw)

            candidate_Ch.append(Ch_raw)
            candidate_Cdelta.append(Cdelta_raw)
            candidate_Rh.append(Rh_raw)
            candidate_Rdelta.append(Rdelta_raw)

            candidate_A.append(A)
            candidate_B.append(B)
            candidate_J.append(J)
            candidate_M.append(M)
            candidate_N.append(N)

        # --------------------------------------------------------
        # Damped simultaneous update of layer kernels
        # --------------------------------------------------------
        all_Ch = [
            damp(old_Ch[l], candidate_Ch[l], damping)
            for l in range(depth)
        ]

        all_Cdelta = [
            damp(
                old_Cdelta[l],
                candidate_Cdelta[l],
                damping,
            )
            for l in range(depth)
        ]

        all_Rh = [
            damp(old_Rh[l], candidate_Rh[l], damping)
            for l in range(depth)
        ]

        all_Rdelta = [
            damp(
                old_Rdelta[l],
                candidate_Rdelta[l],
                damping,
            )
            for l in range(depth)
        ]

        # --------------------------------------------------------
        # Relative fixed-point residual
        # --------------------------------------------------------
        residuals = []

        for old_list, new_list in (
            (old_Ch, all_Ch),
            (old_Cdelta, all_Cdelta),
            (old_Rh, all_Rh),
            (old_Rdelta, all_Rdelta),
        ):
            for old, new in zip(old_list, new_list):
                denominator = jnp.maximum(
                    1.0,
                    jnp.linalg.norm(old),
                )

                residuals.append(
                    jnp.linalg.norm(new - old) / denominator
                )

        final_residual = jnp.max(jnp.stack(residuals))

        # Algebraic residual of the accepted (damped) kernels against
        # the linear layer equations from this Jacobi step.
        eq_residuals = []
        for l in range(depth):
            layer_eq = equation_residuals(
                Ch=all_Ch[l],
                Cdelta=all_Cdelta[l],
                Rh=all_Rh[l],
                Rdelta=all_Rdelta[l],
                Ch_minus=neighbour_Ch_minus[l],
                Cdelta_plus=neighbour_Cdelta_plus[l],
                A=candidate_A[l],
                J=candidate_J[l],
                M=candidate_M[l],
                N=candidate_N[l],
            )
            eq_residuals.extend(layer_eq.values())

        final_equation_residual = jnp.max(
            jnp.stack(eq_residuals)
        )

        final_A = candidate_A
        final_B = candidate_B
        final_J = candidate_J
        final_M = candidate_M
        final_N = candidate_N

        # Python-side stopping is convenient for an exploratory solver.
        # Remove this branch if the complete function will be jitted.
        if tolerance is not None:
            if float(final_residual) < tolerance:
                break

    diagnostics = {
        "iterations": iteration + 1,
        "fixed_point_residual": final_residual,
        "equation_residual": final_equation_residual,
        "D": D,
        "A": final_A,
        "B": final_B,
        "J": final_J,
        "M": final_M,
        "N": final_N,
    }

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

    C_delta_top = top_result["C_delta_top"]
    pc_training_loss = top_result["loss"]
    mean_delta_top = top_result["mean_delta"]

    return (
        all_Ch,
        all_Cdelta,
        all_Rh,
        all_Rdelta,
        C_delta_top,
        pc_training_loss,
        mean_delta_top,
        diagnostics,
    )


def equation_residuals(
    Ch: Array,
    Cdelta: Array,
    Rh: Array,
    Rdelta: Array,
    Ch_minus: Array,
    Cdelta_plus: Array,
    A: Array,
    J: Array,
    M: Array,
    N: Array,
) -> dict:
    source_h = Ch_minus + A @ Cdelta_plus @ A.T
    source_delta = Cdelta_plus + J @ Ch_minus @ J.T

    return {
        "Rh": (
            jnp.linalg.norm(M @ Rh - A)
            / jnp.maximum(1.0, jnp.linalg.norm(A))
        ),
        "Rdelta": (
            jnp.linalg.norm(N @ Rdelta - J)
            / jnp.maximum(1.0, jnp.linalg.norm(J))
        ),
        "Ch": (
            jnp.linalg.norm(M @ Ch @ M.T - source_h)
            / jnp.maximum(1.0, jnp.linalg.norm(source_h))
        ),
        "Cdelta": (
            jnp.linalg.norm(
                N @ Cdelta @ N.T - source_delta
            )
            / jnp.maximum(
                1.0,
                jnp.linalg.norm(source_delta),
            )
        ),
    }


# min_singular_M = jnp.linalg.svd(M, compute_uv=False)[-1]
# min_singular_N = jnp.linalg.svd(N, compute_uv=False)[-1]