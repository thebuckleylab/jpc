"""Theoretical tools for PC networks."""

from jax import vmap
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import equinox.nn as nn
from jaxtyping import PyTree, ArrayLike, Array


def linear_equilib_energy_single(
        network: PyTree[nn.Linear],
        x: ArrayLike,
        y: ArrayLike
) -> Array:
    """See docs of `dln_equilib_energy_batch`."""
    Ws = [
        layer.weight for seq in network for layer in seq if hasattr(layer, "weight")
    ]
    L = len(Ws)

    # Compute product of weight matrices
    WLto1 = jnp.eye(Ws[-1].shape[0])
    for i in range(L - 1, -1, -1):
        WLto1 = WLto1 @ Ws[i]

    # Compute rescaling
    S = jnp.eye(Ws[-1].shape[0])
    cumulative_prod = jnp.eye(Ws[-1].shape[0])
    for i in range(L - 1, 0, -1):
        cumulative_prod = cumulative_prod @ Ws[i]
        S += cumulative_prod @ cumulative_prod.T

    # Compute full expression
    r = y - WLto1 @ x
    return r.T @ jnp.linalg.inv(S) @ r


@eqx.filter_jit
def linear_equilib_energy_batch(
        network: PyTree[nn.Linear],
        x: ArrayLike,
        y: ArrayLike
) -> Array:
    """Computes the theoretical equilibrated PC energy for a deep linear network (DLN).

    $$
    \mathcal{F}^* = 1/N \sum_i^N (\mathbf{y}_i - W_{L:1}\mathbf{x}_i)^T S^{-1}(\mathbf{y}_i - W_{L:1}\mathbf{x}_i)
    $$

    where the rescaling is $S = I_{d_y} + \sum_{\ell=2}^L (W_{L:\ell})(W_{L:\ell})^T$,
    and we use the shorthand $W_{k:\ell} = W_k \dots W_\ell$ for $\ell, k \in 1,\dots, L$.
    See the reference below for more details.

    !!! note

        This expression assumes no biases.

    ??? cite "Reference"

        ```bibtex
        @article{innocenti2024only,
          title={Only Strict Saddles in the Energy Landscape of Predictive Coding Networks?},
          author={Innocenti, Francesco and Achour, El Mehdi and Singh, Ryan and Buckley, Christopher L},
          journal={arXiv preprint arXiv:2408.11979},
          year={2024}
        }
        ```

    **Main arguments:**

    - `network`: Linear network defined as a list of Equinox Linear layers.
    - `x`: Network input.
    - `y`: Network output.

    **Returns:**

    Mean total analytical energy over a batch or dataset.

    """
    return vmap(lambda x, y: linear_equilib_energy_single(
        network,
        x,
        y
    ))(x, y).mean()


@eqx.filter_jit
def linear_activities_hessian(
        Ws: PyTree[Array],
        n_skip: int = 0,
        param_type: str = "SP",
        activity_decay: bool = False,
        diag: bool = True,
        off_diag: bool = True
) -> Array:
    """See docs of `linear_activities_solution_batch`."""
    L = len(Ws)
    N = Ws[0].shape[0]

    # Get layer dimensions
    non_unit_width = 0 if len(Ws[0].shape) == 1 else 1
    dims = [Ws[0].shape[non_unit_width]] + [W.shape[0] for W in Ws]

    # Dimension of A (excluding input and output)
    n_activities = np.sum(dims[1:-1])

    # Preallocate A as a zero matrix
    A = jnp.zeros((n_activities, n_activities))

    # Fill in the blocks of A
    start_idx = 0
    for i in range(1, L):
        end_idx = start_idx + dims[i]

        if diag:
            # Diagonal block
            I = jnp.eye(dims[i])
            WT_W = Ws[i].T @ Ws[i]

            if param_type == "SP":
                a_l = 1
            elif param_type == "NTP":
                a_l = 1 / np.sqrt(N) if n_skip == 0 else 1 / np.sqrt(N*L)
            elif param_type == "μP":
                if i+1 == L:
                    a_l = 1 / N
                else:
                    a_l = 1 / np.sqrt(N) if n_skip == 0 else 1 / np.sqrt(N*L)

            if n_skip == 0:
                if activity_decay:
                    diagonal_block = 2*I + a_l**2 * WT_W
                else:
                    diagonal_block = I + a_l**2 * WT_W

            elif n_skip == 1:
                W = Ws[i]
                if i+1 == L:
                    if activity_decay:
                        diagonal_block = 2*I + a_l**2 * WT_W
                    else:
                        diagonal_block = I + a_l**2 * WT_W
                else:
                    if activity_decay:
                        diagonal_block = 3*I + a_l**2 * WT_W + 2*a_l * (W.T + W)
                    else:
                        diagonal_block = 2*I + a_l**2 * WT_W + a_l * (W.T + W)

            A = A.at[start_idx:end_idx, start_idx:end_idx].set(diagonal_block)

        if off_diag:
            # Off-diagonal block
            if i < L - 1:
                if n_skip == 0:
                    a_l = 1 if param_type == "SP" else 1 / np.sqrt(N)
                    off_diagonal_block = - a_l * Ws[i].T

                if n_skip == 1:
                    I = jnp.eye(dims[i])
                    a_l = 1 if param_type == "SP" else 1 / np.sqrt(N*L)
                    off_diagonal_block = - a_l * Ws[i].T - I

                A = A.at[start_idx:end_idx, end_idx:end_idx + dims[i + 1]].set(
                    off_diagonal_block
                )
                A = A.at[end_idx:end_idx + dims[i + 1], start_idx:end_idx].set(
                    off_diagonal_block.T
                )

        start_idx = end_idx

    return A


def linear_activities_solution_single(
        network: PyTree[nn.Linear],
        x: ArrayLike,
        y: ArrayLike,
        n_skip: int = 0,
        param_type: str = "SP",
        activity_decay: bool = False
) -> PyTree[Array]:
    """See docs of `linear_activities_solution_batch`."""
    # Extract all weight matrices from the network
    Ws = [
        layer.weight for seq in network for layer in seq if hasattr(layer, "weight")
    ]
    D = Ws[0].shape[1]
    N = Ws[0].shape[0]
    H = len(Ws)-1

    # Construct matrix of the linear system
    A = linear_activities_hessian(
        Ws=Ws,
        param_type=param_type,
        n_skip=n_skip,
        activity_decay=activity_decay
    )

    if param_type == "SP":
        a_1, a_L = 1, 1
    if param_type == "NTP":
        a_1, a_L = 1/np.sqrt(D), 1/np.sqrt(N)
    elif param_type == "μP":
        a_1, a_L = 1/np.sqrt(D), 1/N

    # Get layer dimensions
    is_scalar = N == 1
    nonunit_width = 0 if is_scalar else 1
    dims = [Ws[0].shape[nonunit_width]] + [W.shape[0] for W in Ws]

    # Compute activities solution
    b = jnp.zeros(len(A))
    if H == 1:
        b = a_1 * Ws[0] @ x + a_L * Ws[-1].T @ y
    else:
        b = b.at[:dims[1]].set(a_1 * Ws[0] @ x)
        b = b.at[-dims[-2]:].set(a_L * Ws[-1].T @ y)
    
    z_star = jnp.linalg.inv(A) @ b

    # Reshape result as a list of activities for each layer
    activities_solution = []
    start_idx = 0
    for dim in dims[1:-1]:
        activities_solution.append(z_star[start_idx:start_idx + dim])
        start_idx += dim

    # Add target prediction for energy computation
    activities_solution.append(a_L * Ws[-1] @ activities_solution[-1])

    return activities_solution


@eqx.filter_jit
def linear_activities_solution_batch(
        network: PyTree[nn.Linear],
        x: ArrayLike,
        y: ArrayLike,
        n_skip: int = 0,
        param_type: str = "SP",
        activity_decay: bool = False
) -> PyTree[Array]:
    """Computes the theoretical solution for the PC activities of a deep linear network (DLN).

    $$
    \mathbf{z}^* = A^{-1} \mathbf{b}
    $$

    where $A$ is a sparse block diagonal matrix depending only on the weights,
    and $\mathbf{b} = [W_1 \mathbf{x}, \mathbf{0}, \dots, W_L^T \mathbf{y}]^T$.
    In particular, $A_{\ell,k} = I + W_\ell^T W_\ell$ if $\ell = k$,
    $A_{\ell,k} = -W_\ell$ if $\ell = k+1$,
    $A_{\ell,k} = -W_\ell^T$ if $\ell = k-1$, and $\mathbf{0}$ otherwise,
    for $\ell, k \in [2, \dots, L]$.

    !!! note

        This expression assumes no biases.

    **Main arguments:**

    - `network`: Linear network defined as a list of Equinox Linear layers.
    - `x`: Network input.
    - `y`: Network output.

    **Returns:**

    List of theoretical activities for each layer.

    """
    return vmap(lambda x, y: linear_activities_solution_single(
        network,
        x,
        y,
        n_skip,
        param_type,
        activity_decay
    ))(x, y)
