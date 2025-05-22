"""Theoretical tools to study PC networks."""

from jax import vmap
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import equinox.nn as nn
from jaxtyping import PyTree, ArrayLike, Array


def compute_linear_equilib_energy(
        network: PyTree[nn.Linear],
        x: ArrayLike,
        y: ArrayLike
) -> Array:
    """Computes the theoretical equilibrated PC energy for a deep linear network.

    $$
    \mathcal{F}^* = 1/N \sum_i^N (\mathbf{y}_i - W_{L:1}\mathbf{x}_i)^T S^{-1}(\mathbf{y}_i - W_{L:1}\mathbf{x}_i)
    $$

    where the rescaling is $S = I_{d_y} + \sum_{\ell=2}^L (W_{L:\ell})(W_{L:\ell})^T$,
    and we use the shorthand $W_{k:\ell} = W_k \dots W_\ell$ for $\ell, k \in 1,\dots, L$.
    See the reference below for more details.

    !!! note

        This expression assumes no biases. It could also be generalised to 
        other network architectures (e.g. ResNets) and parameterisations
        (see https://arxiv.org/abs/2505.13124).

    ??? cite "Reference"

        ```bibtex
        @article{innocenti2025only,
            title={Only Strict Saddles in the Energy Landscape of Predictive Coding Networks?},
            author={Innocenti, Francesco and Achour, El Mehdi and Singh, Ryan and Buckley, Christopher L},
            journal={Advances in Neural Information Processing Systems},
            volume={37},
            pages={53649--53683},
            year={2025}
        }
        ```

    **Main arguments:**

    - `network`: Linear network defined as a list of Equinox Linear layers.
    - `x`: Network input.
    - `y`: Network output.

    **Returns:**

    Mean total theoretical energy over a data batch.

    """
    Ws = [
        layer.weight for seq in network for layer in seq if hasattr(layer, "weight")
    ]
    L = len(Ws)

    # compute product of weight matrices
    WLto1 = jnp.eye(Ws[-1].shape[0])
    for i in range(L - 1, -1, -1):
        WLto1 = WLto1 @ Ws[i]

    # compute rescaling
    S = jnp.eye(Ws[-1].shape[0])
    cumulative_prod = jnp.eye(Ws[-1].shape[0])
    for i in range(L - 1, 0, -1):
        cumulative_prod = cumulative_prod @ Ws[i]
        S += cumulative_prod @ cumulative_prod.T

    # compute final expression
    S_inv = jnp.linalg.inv(S)
    return vmap(
        lambda x, y: 0.5 * (y - WLto1 @ x).T @ S_inv @ (y - WLto1 @ x)
    )(x, y).mean()


@eqx.filter_jit
def compute_linear_activity_hessian(
        Ws: PyTree[Array],
        *,
        n_skip: int = 0,
        param_type: str = "sp",
        activity_decay: bool = False,
        diag: bool = True,
        off_diag: bool = True
) -> Array:
    """Computes the theoretical Hessian matrix of the PC energy with respect to 
    the activities for a linear network, 
    $\partial^2 \mathcal{F}/\partial \mathbf{z}_\ell \partial \mathbf{z}_k$.
    
    See [this paper](https://arxiv.org/abs/2505.13124) for more details.

    !!! warning

        This was highly hard-coded for quick experimental iteration with 
        different models and parameterisations. The computation of the blocks
        could be implemented much more elegantly by fetching the 
        transformation for each layer (see the paper below).

    !!! info

        This function can be used to study the inference landscape of linear
        PC networks and compute the solution with 
        `compute_linear_activity_solution()`.

    ??? cite "Reference"

        ```bibtex
        TODO
        ```

    **Main arguments:**

    - `Ws`: List of all the network weight matrices.

    **Other arguments:**

    - `n_skip`: Number of layers to skip for the skip connections.
    - `param_type`: Determines the parameterisation. Options are `sp`, `mup`, or `ntp`.
    - `activity_decay`: $\ell^2$ regulariser for the activities.
    - `diag`: Whether to compute the diagonal blocks of the Hessian.
    - `off-diag`: Whether to compute the off-diagonal blocks of the Hessian.

    **Returns:**

    The activity Hessian matrix of size NHxNH where N is the width and H is the
    number of hidden layers.

    """
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

            if param_type == "sp":
                a_l = 1
            elif param_type == "ntp":
                a_l = 1 / np.sqrt(N) if n_skip == 0 else 1 / np.sqrt(N*L)
            elif param_type == "mup":
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
                    a_l = 1 if param_type == "sp" else 1 / np.sqrt(N)
                    off_diagonal_block = - a_l * Ws[i].T

                if n_skip == 1:
                    I = jnp.eye(dims[i])
                    a_l = 1 if param_type == "sp" else 1 / np.sqrt(N*L)
                    off_diagonal_block = - a_l * Ws[i].T - I

                A = A.at[start_idx:end_idx, end_idx:end_idx + dims[i + 1]].set(
                    off_diagonal_block
                )
                A = A.at[end_idx:end_idx + dims[i + 1], start_idx:end_idx].set(
                    off_diagonal_block.T
                )

        start_idx = end_idx

    return A


def compute_linear_activity_solution(
        network: PyTree[nn.Linear],
        x: ArrayLike,
        y: ArrayLike,
        *,
        n_skip: int = 0,
        param_type: str = "sp",
        activity_decay: bool = False
) -> PyTree[Array]:
    """Computes the theoretical solution for the PC activities of a linear network.

    $$
    \mathbf{z}^* = A^{-1} \mathbf{b}
    $$

    where $A$ is the Hessian of the PC energy with respect to all the activities,
    and $\mathbf{b} = [W_1 \mathbf{x}, \mathbf{0}, \dots, W_L^T \mathbf{y}]^T$.
    In particular, $A_{\ell,k} = I + W_\ell^T W_\ell$ if $\ell = k$,
    $A_{\ell,k} = -W_\ell$ if $\ell = k+1$,
    $A_{\ell,k} = -W_\ell^T$ if $\ell = k-1$, and $\mathbf{0}$ otherwise,
    for $\ell, k \in [2, \dots, L]$. See [this paper](https://arxiv.org/abs/2505.13124) 
    for more details.

    !!! info

        This uses `compute_linear_activity_hessian()` to compute $A$. 
    
    ??? cite "Reference"

        ```bibtex
        TODO
        ```

    **Main arguments:**

    - `network`: Linear network defined as a list of Equinox Linear layers.
    - `x`: Network input.
    - `y`: Network output.

    **Returns:**

    List of theoretical activities for each layer.

    """
    # extract all weight matrices from the network
    Ws = [
        layer.weight for seq in network for layer in seq if hasattr(layer, "weight")
    ]
    D = Ws[0].shape[1]
    N = Ws[0].shape[0]
    H = len(Ws)-1

    # construct matrix of the linear system
    A = compute_linear_activity_hessian(
        Ws=Ws,
        param_type=param_type,
        n_skip=n_skip,
        activity_decay=activity_decay
    )

    if param_type == "sp":
        a_1, a_L = 1, 1
    if param_type == "ntp":
        a_1, a_L = 1/np.sqrt(D), 1/np.sqrt(N)
    elif param_type == "mup":
        a_1, a_L = 1/np.sqrt(D), 1/N

    # get layer dimensions
    is_scalar = N == 1
    nonunit_width = 0 if is_scalar else 1
    dims = [Ws[0].shape[nonunit_width]] + [W.shape[0] for W in Ws]

    # compute activities solution
    A_inv = jnp.linalg.inv(A)
    def compute_linear_activity_solution_single(x, y, A_inv, Ws, dims, H, a_1, a_L):
        b = jnp.zeros(len(A))
        if H == 1:
            b = a_1 * Ws[0] @ x + a_L * Ws[-1].T @ y
        else:
            b = b.at[:dims[1]].set(a_1 * Ws[0] @ x)
            b = b.at[-dims[-2]:].set(a_L * Ws[-1].T @ y)
        
        return A_inv @ b
    
    batched_fn = vmap(
        lambda x, y: compute_linear_activity_solution_single(
            x, y, A_inv, Ws, dims, H, a_1, a_L
        )
    )
    z_star = batched_fn(x, y)

    # reshape result as a list of activities for each layer
    activities_solution = []
    start_idx = 0
    for dim in dims[1:-1]:
        activities_solution.append(z_star[start_idx:start_idx + dim])
        start_idx += dim

    # add dummy target prediction for optional later energy computation
    activities_solution.append(a_L * Ws[-1] @ activities_solution[-1])

    return activities_solution
