"""Theoretical tools to study PC networks."""

from jax import vmap
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import equinox.nn as nn
from jaxtyping import PyTree, ArrayLike, Array
from ._errors import _check_param_type


def compute_linear_equilib_energy(
        network: PyTree[nn.Linear],
        x: ArrayLike,
        y: ArrayLike
) -> Array:
    """Computes the theoretical [PC energy](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc.pc_energy_fn) 
    at the solution of the activities for a deep linear network ([Innocenti et al. 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6075fc6540b9a3cb951752099efd86ef-Abstract-Conference.html)):

    $$
    \mathcal{F}(\mathbf{z}^*) = 1/2N \sum_i^N (\mathbf{y}_i - \mathbf{W}_{L:1}\mathbf{x}_i)^T \mathbf{S}^{-1}(\mathbf{y}_i - \mathbf{W}_{L:1}\mathbf{x}_i)
    $$

    where $\mathbf{S} = \mathbf{I}_{d_y} + \sum_{\ell=2}^L (\mathbf{W}_{L:\ell})(\mathbf{W}_{L:\ell})^T$
    and $\mathbf{W}_{k:\ell} = \mathbf{W}_k \dots \mathbf{W}_\ell$ for $\ell, k \in 1,\dots, L$.

    !!! note

        This expression assumes no biases. It could also be generalised to 
        other network architectures (e.g. ResNets) and parameterisations
        (see [Innocenti et al. 2025](https://arxiv.org/abs/2505.13124)). 
        However, note that the equilibrated energy for ResNets and other
        parameterisations can still be computed by getting the activity solution
        with [`jpc.compute_linear_activity_solution()`](https://thebuckleylab.github.io/jpc/api/Theoretical%20tools/#jpc.compute_linear_activity_solution) 
        and then plugging this into the standard PC energy 
        [jpc.pc_energy_fn()](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc.pc_energy_fn).

    !!! example

        In practice, this means that if you run, at any point in training, the 
        inference dynamics of any PC linear network to equilibrium, then 
        [`jpc.pc_energy_fn()`](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc.pc_energy_fn) 
        will return the same energy value as this function. For a demonstration, see
        [this example notebook](https://thebuckleylab.github.io/jpc/examples/linear_net_theoretical_energy/).
        
        ![](linear_net_theoretical_energy.png)

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
        use_skips: bool = False,
        param_type: str = "sp",
        activity_decay: bool = False,
        diag: bool = True,
        off_diag: bool = True
) -> Array:
    """Computes the theoretical Hessian matrix of the [PC energy](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc.pc_energy_fn) 
    with respect to the activities for a linear network, 
    $(\mathbf{H}_{\mathbf{z}})_{\ell k} := \partial^2 \mathcal{F} / \partial \mathbf{z}_\ell \partial \mathbf{z}_k \in \mathbb{R}^{(NH)×(NH)}$ 
    where $N$ and $H$ are the width and number of hidden layers, respectively ([Innocenti et al., 2025](https://arxiv.org/abs/2505.13124)).

    !!! info

        This function can be used (i) to study the inference landscape of linear
        PC networks and (ii) to compute the analytical solution with 
        [`jpc.compute_linear_activity_solution()`](https://thebuckleylab.github.io/jpc/api/Theoretical%20tools/#jpc.compute_linear_activity_solution).
    
    !!! warning

        This was highly hard-coded for quick experimental iteration with 
        different models and parameterisations. The computation of the blocks
        could be implemented much more elegantly by fetching the 
        transformation for each layer.

    ??? cite "Reference"

        ```bibtex
        @article{innocenti2025mu,
            title={$$\backslash$mu $ PC: Scaling Predictive Coding to 100+ Layer Networks},
            author={Innocenti, Francesco and Achour, El Mehdi and Buckley, Christopher L},
            journal={arXiv preprint arXiv:2505.13124},
            year={2025}
        }
        ```

    **Main arguments:**

    - `Ws`: List of all the network weight matrices.

    **Other arguments:**

    - `use_skips`: Whether to assume one-layer skip connections at every layer 
        except from the input and to the output.
    - `param_type`: Determines the parameterisation. Options are `sp` (standard
        parameterisation), `mupc` ([μPC](https://arxiv.org/abs/2505.13124)), or 
        `ntp` (neural tangent parameterisation). See [`jpc.get_param_scalings()`](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc._get_param_scalings) 
        for the specific scalings of these different parameterisations.
    - `activity_decay`: $\ell^2$ regulariser for the activities.
    - `diag`: Whether to compute the diagonal blocks of the Hessian.
    - `off-diag`: Whether to compute the off-diagonal blocks of the Hessian.

    **Returns:**

    The activity Hessian matrix of size $NH×NH$ where $N$ is the width and $H$ 
    is the number of hidden layers.

    """
    _check_param_type(param_type)

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
                a_l = 1 / np.sqrt(N) if not use_skips else 1 / np.sqrt(N*L)
            elif param_type == "mupc":
                if i+1 == L:
                    a_l = 1 / N
                else:
                    a_l = 1 / np.sqrt(N) if not use_skips else 1 / np.sqrt(N*L)

            if not use_skips:
                if activity_decay:
                    diagonal_block = 2*I + a_l**2 * WT_W
                else:
                    diagonal_block = I + a_l**2 * WT_W

            elif use_skips:
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
                if not use_skips:
                    a_l = 1 if param_type == "sp" else 1 / np.sqrt(N)
                    off_diagonal_block = - a_l * Ws[i].T

                if use_skips:
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
        use_skips: bool = False,
        param_type: str = "sp",
        activity_decay: bool = False
) -> PyTree[Array]:
    """Computes the theoretical solution for the PC activities of a linear network ([Innocenti et al., 2025](https://arxiv.org/abs/2505.13124)).

    $$
    \mathbf{z}^* = \mathbf{H}_{\mathbf{z}}^{-1}\mathbf{b}
    $$

    where $(\mathbf{H}_{\mathbf{z}})_{\ell k} := \partial^2 \mathcal{F} / \partial \mathbf{z}_\ell \partial \mathbf{z}_k \in \mathbb{R}^{(NH)×(NH)}$ 
    is the Hessian of the energy with respect to the activities, and 
    $\mathbf{b} \in \mathbb{R}^{NH}$ is a sparse vector depending only on the 
    data and associated weights. The activity Hessian is computed analytically 
    using [`jpc.compute_linear_activity_hessian()`](https://thebuckleylab.github.io/jpc/api/Theoretical%20tools/#jpc.compute_linear_activity_hessian).    
    
    !!! info

        This can be used to study how linear PC networks learn when they perform 
        perfect inference. An example notebook demonstration is in the works!
    
    ??? cite "Reference"

        ```bibtex
        @article{innocenti2025mu,
            title={$$\backslash$mu $ PC: Scaling Predictive Coding to 100+ Layer Networks},
            author={Innocenti, Francesco and Achour, El Mehdi and Buckley, Christopher L},
            journal={arXiv preprint arXiv:2505.13124},
            year={2025}
        }
        ```

    **Main arguments:**

    - `network`: Linear network defined as a list of Equinox Linear layers.
    - `x`: Network input.
    - `y`: Network output.

    **Other arguments:**

    - `use_skips`: Whether to assume one-layer skip connections at every layer 
        except from the input and to the output.
    - `param_type`: Determines the parameterisation. Options are `sp` (standard
        parameterisation), `mupc` ([μPC](https://arxiv.org/abs/2505.13124)), or 
        `ntp` (neural tangent parameterisation). See [`jpc.get_param_scalings()`](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc._get_param_scalings) 
        for the specific scalings of these different parameterisations.
    - `activity_decay`: $\ell^2$ regulariser for the activities.

    **Returns:**

    List of theoretical activities for each layer.

    """
    _check_param_type(param_type)

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
        use_skips=use_skips,
        activity_decay=activity_decay
    )

    if param_type == "sp":
        a_1, a_L = 1, 1
    if param_type == "ntp":
        a_1, a_L = 1/np.sqrt(D), 1/np.sqrt(N)
    elif param_type == "mupc":
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
        
        z_star = A_inv @ b
    
        # reshape result as a list of activities for each layer
        activities_solution = []
        start_idx = 0
        for dim in dims[1:-1]:
            activities_solution.append(z_star[start_idx:start_idx + dim])
            start_idx += dim

        # add dummy target prediction for optional later energy computation
        activities_solution.append(a_L * Ws[-1] @ activities_solution[-1])

        return activities_solution

    return vmap(
        lambda x, y: compute_linear_activity_solution_single(
            x, y, A_inv, Ws, dims, H, a_1, a_L
        )
    )(x, y)
