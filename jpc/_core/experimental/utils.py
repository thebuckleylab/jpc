from jax import vmap
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PyTree, ArrayLike, Array, Scalar
from typing import Optional, Tuple, Callable


def new_linear_equilib_energy(
    params: Tuple[PyTree[Callable], Optional[PyTree[Callable]]],
    x: ArrayLike,
    y: ArrayLike,
    *,
    gamma: Scalar,
) -> Array:
    """Computes the closed-form equilibrated energy for a one-hidden-layer linear
    network with scalar output:

    $$
    \mathcal{E}^* = \frac{1}{2}\frac{\left(y - \frac{\mathbf{w}_2 \mathbf{W}_1 \mathbf{x}}
    {\gamma N \sqrt{D}}\right)^2}
    {1 + \frac{\mathbf{w}_2 \mathbf{w}_2^\top}{\gamma^2 N^2}}
    $$

    where $\mathbf{w}_2 \in \mathbb{R}^{1 \times N}$, $\mathbf{W}_1 \in \mathbb{R}^{N \times D}$,
    $\mathbf{x} \in \mathbb{R}^{D}$, $y \in \mathbb{R}$, and $N$ and $D$ are the hidden
    and input dimensions.

    **Main arguments:**

    - `params`: Tuple with callable network layers and optional skip connections.
    - `x`: Network input.
    - `y`: Scalar network target.

    **Other arguments:**

    - `gamma`: Output-layer scaling factor.

    **Returns:**

    Mean equilibrated energy over a data batch.

    """
    model, skip_model = params

    Ws = [
        layer.weight for seq in model for layer in seq if hasattr(layer, "weight")
    ]
    if len(Ws) != 2:
        raise ValueError(
            "new_linear_equilib_energy requires a one-hidden-layer linear network "
            f"with 2 weight matrices, got {len(Ws)}."
        )
    if Ws[-1].shape[0] != 1:
        raise ValueError(
            "new_linear_equilib_energy requires scalar output "
            f"(output dimension 1), got {Ws[-1].shape[0]}."
        )
    if skip_model is not None and any(s is not None for s in skip_model):
        raise ValueError("new_linear_equilib_energy does not support skip connections.")

    W1, w2 = Ws[0], Ws[1]
    N = W1.shape[0]
    D = W1.shape[1]
    denom = gamma * N * jnp.sqrt(D)
    S = 1.0 + jnp.squeeze(w2 @ w2.T) / N

    def _single_energy(x_i, y_i):
        pred = (w2 @ W1 @ x_i) / denom
        residual = y_i - jnp.squeeze(pred)
        return 0.5 * residual ** 2 / S

    if x.ndim == 1:
        return _single_energy(x, y)
    return vmap(_single_energy)(x, y).mean()


def compute_new_linear_equilib_energy_grads(
    params: Tuple[PyTree[Callable], Optional[PyTree[Callable]]],
    x: ArrayLike,
    y: ArrayLike,
    *,
    gamma: Scalar,
) -> PyTree[Array]:
    """Computes the gradient of [`new_linear_equilib_energy()`]
    with respect to model parameters $∇_θ \mathcal{E}^*$.

    **Main arguments:**

    - `params`: Tuple with callable model layers and optional skip connections.
    - `x`: Network input.
    - `y`: Scalar network target.

    **Other arguments:**

    - `gamma`: Output-layer scaling factor.

    **Returns:**

    Parameter gradients for the network.

    """
    return eqx.filter_grad(new_linear_equilib_energy)(
        params,
        x,
        y,
        gamma=gamma,
    )
