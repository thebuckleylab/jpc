import math
from typing import Callable, List

import jax
import jax.numpy as jnp
import jax.random as jr

import jpc
import equinox as eqx
import equinox.nn as nn

from utils import init_weights


class LayerNorm(eqx.Module):
    """Token-wise LayerNorm for inputs shaped (T, N)."""
    ln: eqx.nn.LayerNorm

    def __init__(self, ndim: int, *, eps: float = 1e-5, use_bias: bool = False):
        self.ln = eqx.nn.LayerNorm(ndim, eps=eps, use_bias=use_bias)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # vmap over tokens to support (T, N).
        return jax.vmap(self.ln)(x)


class CausalSelfAttention(eqx.Module):
    W_qkv: nn.Linear
    W_out: nn.Linear
    n_embd: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    d_head: int = eqx.field(static=True)
    use_softmax: bool = eqx.field(static=True)
    attn_scaling: float = eqx.field(static=True)

    def __init__(
        self,
        n_embd: int,
        n_heads: int = 1,
        *,
        key: jax.Array,
        param_type: str = "sp",
        use_softmax: bool = True,
        use_bias: bool = False,
        n_blocks: int = 2,
        init_std: float = 0.02
    ):
        assert n_embd % n_heads == 0

        k_qkv, k_out = jr.split(key, 2)
        k_qkv_lin, k_qkv_mup = jr.split(k_qkv, 2)
        k_out_lin, k_out_mup = jr.split(k_out, 2)
        self.W_qkv = nn.Linear(n_embd, 3 * n_embd, key=k_qkv_lin, use_bias=use_bias)
        self.W_out = nn.Linear(n_embd, n_embd, key=k_out_lin, use_bias=use_bias)

        std = init_std / math.sqrt(n_embd) if param_type == "sp" else init_std
        self.W_qkv = init_weights(self.W_qkv, k_qkv_mup, std=std)

        w_out_std = std if param_type == "mupc" else std / math.sqrt(2 * n_blocks)
        self.W_out = init_weights(self.W_out, k_out_mup, std=w_out_std)

        self.n_embd = n_embd
        self.n_heads = n_heads
        self.d_head = n_embd // n_heads
        self.use_softmax = use_softmax

        if param_type == "mupc":
            self.attn_scaling = 1 / self.d_head
        else:
            self.attn_scaling = 1 / math.sqrt(self.d_head)

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        # Assume no batch dimension: x is (T, N)
        assert x.ndim == 2
        N = x.shape[-1]
        assert N == self.n_heads * self.d_head

        T = x.shape[0]
        qkv = jax.vmap(self.W_qkv)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # (T, N) -> (n_heads, T, d_head)
        q = q.reshape(T, self.n_heads, self.d_head).transpose(1, 0, 2)
        k = k.reshape(T, self.n_heads, self.d_head).transpose(1, 0, 2)
        v = v.reshape(T, self.n_heads, self.d_head).transpose(1, 0, 2)
        attn = jnp.einsum("h t d, h s d -> h t s", q, k) * self.attn_scaling
        causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        neg_inf = jnp.array(-jnp.inf, dtype=attn.dtype)
        attn = jnp.where(causal_mask[None, :, :], attn, neg_inf)
        if self.use_softmax:
            attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum("h t s, h s d -> h t d", attn, v)
        out = out.transpose(1, 0, 2).reshape(T, N)
        return jax.vmap(self.W_out)(out)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.forward(x)


class MLP(eqx.Module):
    W1: nn.Linear
    W2: nn.Linear
    act_fn: Callable = eqx.field(static=True)

    def __init__(
        self,
        n_embd: int,
        *,
        key: jax.Array,
        param_type: str = "sp",
        expand_factor: int = 4,
        use_bias: bool = False,
        act_fn: str = "linear",
        n_blocks: int = 2,
        init_std: float = 0.02
    ):
        k1, k2 = jr.split(key, 2)
        k1_lin, k1_mup = jr.split(k1, 2)
        k2_lin, k2_mup = jr.split(k2, 2)

        self.W1 = nn.Linear(
            n_embd, expand_factor * n_embd, key=k1_lin, use_bias=use_bias
        )
        self.act_fn = jpc.get_act_fn(act_fn)
        self.W2 = nn.Linear(
            expand_factor * n_embd, n_embd, key=k2_lin, use_bias=use_bias
        )

        std = init_std / math.sqrt(n_embd) if param_type == "sp" else init_std
        self.W1 = init_weights(self.W1, k1_mup, std=std)

        w2_std = std if param_type == "mupc" else std / math.sqrt(2 * n_blocks)
        self.W2 = init_weights(self.W2, k2_mup, std=w2_std)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Assume no batch dimension: x is (T, N)
        assert x.ndim == 2
        h = self.act_fn(jax.vmap(self.W1)(x))
        out = jax.vmap(self.W2)(h)
        return out  # removed x + out


class Block(eqx.Module):
    ln_1: eqx.Module
    attn: CausalSelfAttention
    ln_2: eqx.Module
    mlp: MLP
    depth_scaling: float = eqx.field(static=True)

    def __init__(
        self,
        n_embd: int,
        n_heads: int = 1,
        *,
        key: jax.Array,
        param_type: str = "sp",
        use_layer_norm: bool = True,
        use_softmax: bool = True,
        use_bias: bool = False,
        act_fn: str = "linear",
        n_blocks: int = 2,
        init_std: float = 0.02
    ):
        k_attn, k_mlp = jr.split(key, 2)
        self.ln_1 = (
            LayerNorm(n_embd, eps=1e-5, use_bias=use_bias)
            if use_layer_norm
            else eqx.nn.Identity()
        )
        self.attn = CausalSelfAttention(
            n_embd=n_embd, 
            n_heads=n_heads, 
            key=k_attn, 
            param_type=param_type, 
            use_softmax=use_softmax, 
            use_bias=use_bias,
            n_blocks=n_blocks,
            init_std=init_std
        )
        self.mlp = MLP(
            n_embd=n_embd, 
            key=k_mlp, 
            param_type=param_type, 
            use_bias=use_bias, 
            act_fn=act_fn,
            n_blocks=n_blocks,
            init_std=init_std
        )
        self.ln_2 = (
            LayerNorm(n_embd, eps=1e-5, use_bias=use_bias)
            if use_layer_norm
            else eqx.nn.Identity()
        )
        self.depth_scaling = 1 / n_blocks

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x + self.depth_scaling * self.attn(self.ln_1(x))
        x = x + self.depth_scaling * self.mlp(self.ln_2(x))
        return x


class TokenPositionEmbedding(eqx.Module):
    wte: nn.Embedding
    wpe: nn.Embedding

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        n_embd: int,
        *,
        key: jax.Array,
        init_std: float = 0.02
    ):
        kt, kp = jr.split(key, 2)
        kt_embed, kt_init = jr.split(kt, 2)
        kp_embed, kp_init = jr.split(kp, 2)
        self.wte = nn.Embedding(vocab_size, n_embd, key=kt_embed)
        self.wpe = nn.Embedding(seq_len, n_embd, key=kp_embed)
        self.wte = init_weights(self.wte, kt_init, std=init_std)
        self.wpe = init_weights(self.wpe, kp_init, std=init_std)

    def __call__(self, idx: jnp.ndarray) -> jnp.ndarray:
        # Assume no batch dimension: idx is (T,)
        assert idx.ndim == 1
        t = idx.shape[-1]
        pos = jnp.arange(t)

        tok_emb = jax.vmap(self.wte)(idx)
        pos_emb = jax.vmap(self.wpe)(pos)
        return tok_emb + pos_emb


class LMHead(eqx.Module):
    linear: nn.Linear
    scaling: float = eqx.field(static=True)

    def __init__(
        self,
        n_embd: int,
        vocab_size: int,
        *,
        key: jax.Array,
        param_type: str = "sp",
        use_bias: bool = False,
        init_std: float = 0.02
    ):
        k_lin, k_mup = jr.split(key, 2)
        self.linear = nn.Linear(n_embd, vocab_size, key=k_lin, use_bias=use_bias)
        self.linear = init_weights(self.linear, k_mup, std=init_std)

        self.scaling = 1 / n_embd if param_type == "mupc" else 1

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Assume no batch dimension: x is (T, N)
        assert x.ndim == 2
        y = jax.vmap(self.linear)(x)
        return self.scaling * y


class Transformer(eqx.Module):
    layers: List[eqx.Module]

    def __init__(
        self,
        *,
        key: jax.Array,
        vocab_size: int,
        seq_len: int,
        n_embd: int,
        n_blocks: int,
        n_heads: int = 1,
        param_type: str = "sp",
        use_layer_norm: bool = True,
        use_softmax: bool = True,
        use_bias: bool = False,
        act_fn: str = "linear",
        init_std: float = 0.02
    ):
        keys = jr.split(key, 2 + n_blocks)

        self.layers = []
        self.layers.append(
            TokenPositionEmbedding(
                vocab_size, 
                seq_len, 
                n_embd,
                key=keys[0],
                init_std=init_std
            )
        )
        for i in range(n_blocks):
            self.layers.append(
                Block(
                    n_embd,
                    n_heads,
                    key=keys[i + 1],
                    param_type=param_type,
                    use_softmax=use_softmax,
                    use_bias=use_bias,
                    act_fn=act_fn,
                    n_blocks=n_blocks,
                    use_layer_norm=use_layer_norm,
                    init_std=init_std
                )
            )
        self.layers.append(
            LayerNorm(n_embd, eps=1e-5, use_bias=use_bias)
            if use_layer_norm
            else eqx.nn.Identity()
        )
        self.layers.append(
            LMHead(
                n_embd,
                vocab_size,
                key=keys[-1],
                param_type=param_type,
                use_bias=use_bias,
                init_std=init_std
            )
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for f in self.layers:
            x = f(x)
        return x

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, idx: int) -> eqx.Module:
        return self.layers[idx]
