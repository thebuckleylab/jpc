"""Theoretical tools for predictive coding networks."""

from jax import vmap
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
from jaxtyping import PyTree, ArrayLike, Array


def linear_equilib_energy_single(
        network: PyTree[nn.Linear],
        x: ArrayLike,
        y: ArrayLike
) -> Array:
    """See docs of `dln_equilib_energy_batch`."""
    Ws = [l.weight for l in network]
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
    and we use the shorthand $W_{L:\ell} = W_L W_{L-1} \dots W_\ell$.
    See reference below.

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

    Mean total analytical energy across data batch.

    """
    return vmap(lambda x, y: linear_equilib_energy_single(
        network,
        x,
        y
    ))(x, y).mean()
