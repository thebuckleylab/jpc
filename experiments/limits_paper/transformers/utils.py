import os
import urllib.request
from typing import Union

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import equinox.nn as nn
from jax.flatten_util import ravel_pytree


def init_weights(
    module: Union[nn.Linear, nn.Embedding],
    key: jax.Array,
    *,
    std: float = 1.0,
) -> Union[nn.Linear, nn.Embedding]:
    w = jr.normal(key, module.weight.shape, dtype=module.weight.dtype) * std
    return eqx.tree_at(lambda m: m.weight, module, w)


def load_shakespeare(batch_size, seq_len, seed):
    """Load Shakespeare text and return (x, y) for autoregressive LM.

    x: (batch_size, seq_len) int token indices
    y: (batch_size, seq_len, vocab_size) one-hot next token at each position
    """
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    try:
        with urllib.request.urlopen(url, timeout=10) as f:
            text = f.read().decode("utf-8")
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch Shakespeare from {url}."
        ) from e
    return _text_to_sequences(text, batch_size, seq_len, seed)


def _text_to_sequences(text, batch_size, seq_len, seed):
    """Convert raw text to (x, y) for autoregressive LM. x: (B, T) int indices, y: (B, T, V) one-hot next token."""
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_ix = {c: i for i, c in enumerate(chars)}
    indices = np.array([char_to_ix[c] for c in text], dtype=np.int32)
    # Need seq_len+1 tokens per sample: x[b,t] = token at t, y[b,t] = one-hot of token at t+1
    n_samples = len(indices) - seq_len - 1
    if n_samples < batch_size:
        raise ValueError(
            f"Not enough text for batch_size={batch_size}, seq_len={seq_len}. "
            f"Need at least {batch_size + seq_len + 1} chars, got {len(indices)}."
        )
    rng = np.random.default_rng(seed)
    max_start = n_samples - batch_size
    start = rng.integers(0, max_start + 1) if max_start > 0 else 0
    starts = np.arange(start, start + batch_size)
    # x: (batch_size, seq_len) int indices
    # y_indices: (batch_size, seq_len) where y_indices[b,t] = x[b,t+1]
    x_indices = np.stack([indices[s : s + seq_len] for s in starts])
    y_indices = np.stack([indices[s + 1 : s + seq_len + 1] for s in starts])
    x = jnp.array(x_indices)
    y = jnp.array(jax.nn.one_hot(y_indices, vocab_size, dtype=jnp.float32))
    return x, y, vocab_size


def flatten_grads_per_layer_transformer(model, grads):
    """Flatten gradients per parameter layer for Transformer. One entry per layer (empty array if no params)."""
    if isinstance(grads, tuple):
        grads = grads[0]
    result = []
    for i in range(len(model.layers)):
        flat, _ = ravel_pytree(eqx.filter(grads.layers[i], eqx.is_array))
        result.append(np.array(flat))
    return result


def get_tracked_transformer_param_positions_and_names(model):
    """Track embed, first block (attn+out proj inside AttnConcat), last block (mlp), lm_head for grad cos-sim."""
    layers = model.layers
    # Structure: embed(0), [attn_concat(1), residual(2), mlp(3)] * n_blocks, lm_head(-1)
    n_blocks = (len(layers) - 2) // 3
    param_layer_indices = list(range(len(layers)))
    tracked_param_positions = [0, 1, 3 * n_blocks, len(layers) - 1]
    tracked_layer_names = ["embed", "block0_attn", "block_last_mlp", "lm_head"]
    return param_layer_indices, tracked_param_positions, tracked_layer_names


def setup_experiment(args):
    return os.path.join(
        args.results_dir,
        f"{args.seq_len}_seq_len",
        f"{args.d_model}_d_model",
        f"{args.n_blocks}_n_blocks",
        f"{args.vocab_size}_vocab",
        f"{args.batch_size}_batch_size",
        f"{args.n_steps}_n_steps",
        "adamw_param_optim",
        f"{args.param_type}_param_type",
        f"{args.param_lr}_param_lr",
        f"{args.beta1}_beta1",
        f"{args.beta2}_beta2",
        f"{args.weight_decay}_weight_decay",
        f"{args.adam_eps}_adam_eps",
        f"{args.activity_lr}_activity_lr",
        f"{args.n_infer_iters}_n_infer_iters",
        f"{args.loss_id}_loss_id",
        str(args.seed),
    )


def compute_cosine_similarity(a, b):
    """Compute cosine similarity between two 1D numpy arrays."""
    if a.ndim > 1:
        a = a.reshape(-1)
    if b.ndim > 1:
        b = b.reshape(-1)
    min_dim = min(len(a), len(b))
    if min_dim == 0:
        return 0.0
    a = a[:min_dim]
    b = b[:min_dim]
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na * nb > 1e-10:
        return dot / (na * nb)
    return 0.0
