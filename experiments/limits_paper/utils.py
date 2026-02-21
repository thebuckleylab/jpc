import os
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
from jax import vmap

import jpc
import optax
import equinox as eqx
import equinox.nn as nn


def configure_param_optim(optim_id, param_type, use_skips, param_lr, gamma_0, width, depth):
    if param_type == "sp":
        return optax.sgd(param_lr) if optim_id == "gd" else optax.adam(param_lr)
    else:
        if optim_id == "gd":
            scaled_lr = param_lr * (gamma_0**2) * width
            return optax.sgd(scaled_lr)
        elif optim_id == "adam":
            if use_skips:
                scaled_lr = param_lr / ( np.sqrt(width) * np.sqrt(depth) )
            else:
                scaled_lr = param_lr / np.sqrt(width)
            return optax.adam(scaled_lr)
        else:
            raise ValueError(f"Invalid optimiser: {optim_id}")



def setup_pc_experiment(
        results_dir,
        input_dim,
        n_samples,
        n_hidden,
        use_skips,
        act_fn,
        param_type,
        param_optim_id,
        param_lr,
        gamma_0,
        n_train_iters,
        infer_mode,
        n_infer_iters,
        activity_lr,
        width,
        seed
):
    return os.path.join(
        results_dir,
        f"{input_dim}_input_dim",
        f"{n_samples}_n_samples",
        f"{n_hidden}_n_hidden",
        f"{use_skips}_use_skips",
        f"{act_fn}_act_fn",
        f"{param_type}_param_type",
        f"{param_optim_id}_param_optim",
        f"{param_lr}_param_lr",
        f"{gamma_0}_gamma_0",
        f"{n_train_iters}_n_train_iters",
        f"{infer_mode}_infer_mode",
        f"{n_infer_iters}_n_infer_iters",
        f"{activity_lr}_activity_lr",
        f"{width}_width",
        str(seed)
    )


def setup_bp_experiment(
        results_dir,
        input_dim,
        n_samples,
        n_hidden,
        use_skips,
        act_fn,
        param_type,
        optim_id,
        param_lr,
        gamma_0,
        n_train_iters,
        width,
        seed
):
    return os.path.join(
        results_dir,
        f"{input_dim}_input_dim",
        f"{n_samples}_n_samples",
        f"{n_hidden}_n_hidden",
        f"{use_skips}_use_skips",
        f"{act_fn}_act_fn",
        f"{param_type}_param_type",
        f"{optim_id}_optim_id",
        f"{param_lr}_param_lr",
        f"{gamma_0}_gamma_0",
        f"{n_train_iters}_n_train_iters",
        f"{width}_width",
        str(seed)
    )


def create_toy_dataset(key, D, P):
    X = jr.normal(key, (D, P))
    y = jnp.where(jnp.arange(P) < P//2, 1.0, -1.0)
    return X, y


class MLP(eqx.Module):
    D: int
    N: int
    L: int
    param_type: str
    use_skips: bool
    layers: list
    gamma: float
    
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
        self.D = d_in
        self.N = N
        self.L = L
        self.param_type = param_type
        self.use_skips = use_skips
        self.gamma = gamma

        keys = jr.split(key, L)
        self.layers = []
        for i in range(L):
            act_fn_l = nn.Identity() if i == 0 else jpc.get_act_fn(act_fn)
            _in = d_in if i == 0 else N
            _out = d_out if (i + 1) == L else N
            layer = nn.Sequential(
                [
                    nn.Lambda(act_fn_l),
                    nn.Linear(
                        _in,
                        _out,
                        use_bias=use_bias,
                        key=keys[i]
                    )
                ]
            )
            self.layers.append(layer)

    def __call__(self, x):
        if self.param_type == "mupc":
            for i, f in enumerate(self.layers):
                if (i + 1) == 1:
                    x = f(x) / jnp.sqrt(self.D)
                elif 1 < (i + 1) < self.L:
                    residual = x if self.use_skips else 0
                    rescaling = jnp.sqrt(
                        self.N * self.L
                    ) if self.use_skips else jnp.sqrt(self.N)
                    x = (f(x) / rescaling) + residual
                elif (i + 1) == self.L:
                    x = f(x) / self.N

        else:
            for i, f in enumerate(self.layers):
                residual = x if self.use_skips and (1 < (i + 1) < self.L) else 0

                x = f(x) + residual

        return x / self.gamma


def flatten_grads(grads):
        flat_grads, _ = ravel_pytree(eqx.filter(grads, eqx.is_array))
        return flat_grads


def compute_grad_cosine_similarities(pc_grads, bp_grads):
    """
    Compute cosine similarities between PC and BP gradients over multiple training steps.
    Memory-efficient implementation that processes gradients iteratively to avoid OOM.
    
    Args:
        pc_grads: List or array of PC gradients (shape: n_train_iters x grad_dim)
        bp_grads: List or array of BP gradients (shape: n_train_iters x grad_dim)
    
    Returns:
        Array of cosine similarities (shape: n_train_iters)
    """
    # Ensure same number of iterations
    n_iterations = min(len(pc_grads), len(bp_grads))
    
    # Pre-allocate output array
    cosine_similarities = np.zeros(n_iterations)
    
    # Process each iteration separately to avoid loading all gradients into memory
    for i in range(n_iterations):
        # Get gradients for this iteration (already numpy arrays from train functions)
        pc_grad = pc_grads[i]
        bp_grad = bp_grads[i]
        
        # Ensure they are 1D arrays
        if pc_grad.ndim > 1:
            pc_grad = pc_grad.flatten()
        if bp_grad.ndim > 1:
            bp_grad = bp_grad.flatten()
        
        # Ensure same gradient dimension
        min_dim = min(len(pc_grad), len(bp_grad))
        pc_grad = pc_grad[:min_dim]
        bp_grad = bp_grad[:min_dim]
        
        # Compute cosine similarity: cos_sim = (a · b) / (||a|| * ||b||)
        dot_product = np.dot(pc_grad, bp_grad)
        pc_norm = np.linalg.norm(pc_grad)
        bp_norm = np.linalg.norm(bp_grad)
        
        # Avoid division by zero
        norms_product = pc_norm * bp_norm
        if norms_product > 1e-10:
            cosine_similarities[i] = dot_product / norms_product
        else:
            cosine_similarities[i] = 0.0
        
        # Explicitly delete references to free memory (though Python GC should handle this)
        del pc_grad, bp_grad
    
    return cosine_similarities


def evaluate_pc(params, test_loader, param_type):
    model, skip_model = params
    avg_test_loss, avg_test_acc = 0, 0
    for _, (img_batch, label_batch) in enumerate(test_loader):
        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

        test_loss, test_acc = jpc.test_discriminative_pc(
            model=model,
            output=label_batch,
            input=img_batch,
            skip_model=skip_model,
            param_type=param_type
        )
        avg_test_loss += test_loss
        avg_test_acc += test_acc

    return avg_test_loss / len(test_loader), avg_test_acc / len(test_loader)


def evaluate_bp(model, test_loader):
    avg_test_loss, avg_test_acc = 0, 0
    for _, (img_batch, label_batch) in enumerate(test_loader):
        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()
        
        y_pred = vmap(model)(img_batch)
        avg_test_loss += 0.5 * jnp.mean(jnp.sum((label_batch - y_pred) ** 2, axis=1))
        avg_test_acc += jpc.compute_accuracy(label_batch, y_pred)
    
    return avg_test_loss / len(test_loader), avg_test_acc / len(test_loader)
