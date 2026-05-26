import math

import jax.random as jr
import jax.numpy as jnp

import jpc
import equinox as eqx
import equinox.nn as nn
from torch.nn.init import calculate_gain

from typing import Callable, List
from jaxtyping import PRNGKeyArray


class ScaledConv2d(eqx.Module):
    """2D convolution with optional muP scaling."""
    conv: nn.Conv2d
    scaling: float = eqx.field(static=True)
    
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        *,
        key,
        depth_scaling=1.,
        param_type="sp",
        use_bias=False,
        act_fn="linear",
        weight_std_scale: float = 1.0,
    ):
        keys = jr.split(key, 2)
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            use_bias=use_bias,
            key=keys[0]
        )
        if param_type == "mupc":
            W = jr.normal(
                keys[1], conv.weight.shape, dtype=conv.weight.dtype
            )
            conv = eqx.tree_at(lambda l: l.weight, conv, W * weight_std_scale)
            gain = calculate_gain(act_fn) if act_fn != "linear" else 1.0
            width_scaling = gain / math.sqrt(in_channels * kernel_size ** 2)
        else:
            width_scaling = 1.

        self.conv = conv
        self.scaling = width_scaling * depth_scaling

    def __call__(self, x):
        return self.scaling * self.conv(x)


class ResNetBlock(eqx.Module):
    """Residual block with non-linearity + convolution."""
    scaled_conv: ScaledConv2d
    act_fn: Callable = eqx.field(static=True)

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        *,
        key,
        depth_scaling=1.,
        param_type="sp",
        use_bias=False,
        act_fn="linear"
    ):
        self.act_fn = jpc.get_act_fn(act_fn)
        self.scaled_conv = ScaledConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            use_bias=use_bias,
            key=key,
            depth_scaling=depth_scaling,
            param_type=param_type,
            act_fn=act_fn
        )

    def __call__(self, x):
        res_path = x
        x = self.act_fn(x)
        return self.scaled_conv(x) + res_path


class Readout(eqx.Module):
    linear: nn.Linear
    scaling: float = eqx.field(static=True)

    def __init__(
        self,
        in_features,
        out_features,
        *,
        key,
        scaling=1.,
        param_type="sp",
        use_bias=False,
        weight_std_scale: float = 1.0,
    ):
        keys = jr.split(key, 2)
        linear = nn.Linear(
            in_features, 
            out_features, 
            use_bias=use_bias,
            key=keys[0]
        )
        if param_type == "mupc":
            W = jr.normal(
                keys[1], linear.weight.shape, dtype=linear.weight.dtype
            )
            linear = eqx.tree_at(lambda l: l.weight, linear, W * weight_std_scale)

        self.linear = linear
        self.scaling = scaling

    def __call__(self, x):
        x = jnp.ravel(x)
        return self.scaling * self.linear(x)


class ResNet(eqx.Module):
    """3-stage ResNet.
    
    The architecture and `mupc` parameterisation closely follow those of https://arxiv.org/abs/2309.16620.

    """
    layers: List[eqx.Module]

    def __init__(
        self,
        *,
        key: PRNGKeyArray,
        width: int,
        n_res_blocks: int = 3,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_bias: bool = False,
        param_type: str = "sp",
        in_channels: int = 3,
        input_size: int = 32,
        out_features: int = 10,
        act_fn: str = "linear",
        scale_non_res_layers: bool = False,
        additive_depth_factor: int = 0
    ):
        # 3 stages, each with n_blocks_per_stage residual blocks
        assert n_res_blocks % 3 == 0
        n_blocks_per_stage = n_res_blocks // 3
        
        depth = n_res_blocks + additive_depth_factor
        depth_scaling = 1 / math.sqrt(depth) if param_type == "mupc" else 1.0

        # 3 stages × (1 conv + n_blocks_per_stage blocks) + 1 readout
        tot_param_layers = 4 + 3 * n_blocks_per_stage
        keys = jr.split(key, tot_param_layers)
        key_idx = 0
        self.layers = []

        # scaling for non-residual layers
        non_res_depth_scaling = depth_scaling if (
            param_type == "mupc" and scale_non_res_layers
        ) else 1.0

        # Stage 1: conv in_channels→width + n_blocks_per_stage + avg pool
        self.layers.append(
            ScaledConv2d(
                in_channels=in_channels,
                out_channels=width,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                use_bias=use_bias,
                param_type=param_type,
                key=keys[key_idx],
                depth_scaling=non_res_depth_scaling,
                weight_std_scale=1/non_res_depth_scaling
            )
        )
        key_idx += 1
        for _ in range(n_blocks_per_stage):
            self.layers.append(
                ResNetBlock(
                    in_channels=width,
                    out_channels=width,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    use_bias=use_bias,
                    act_fn=act_fn,
                    param_type=param_type,
                    depth_scaling=depth_scaling,
                    key=keys[key_idx]
                )
            )
            key_idx += 1
        self.layers.append(nn.AvgPool2d(kernel_size=2, stride=2))

        # Stage 2: conv width→width + n_blocks_per_stage + avg pool
        self.layers.append(
            ScaledConv2d(
                in_channels=width,
                out_channels=width,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                use_bias=use_bias,
                param_type=param_type,
                key=keys[key_idx],
                depth_scaling=non_res_depth_scaling,
                weight_std_scale=1/non_res_depth_scaling
            )
        )
        key_idx += 1
        for _ in range(n_blocks_per_stage):
            self.layers.append(
                ResNetBlock(
                    in_channels=width,
                    out_channels=width,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    use_bias=use_bias,
                    act_fn=act_fn,
                    param_type=param_type,
                    depth_scaling=depth_scaling,
                    key=keys[key_idx]
                )
            )
            key_idx += 1
        self.layers.append(nn.AvgPool2d(kernel_size=2, stride=2))

        # Stage 3: conv width→width + n_blocks_per_stage + avg pool
        self.layers.append(
            ScaledConv2d(
                in_channels=width,
                out_channels=width,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                use_bias=use_bias,
                param_type=param_type,
                key=keys[key_idx],
                depth_scaling=non_res_depth_scaling,
                weight_std_scale=1/non_res_depth_scaling
            )
        )
        key_idx += 1
        for _ in range(n_blocks_per_stage):
            self.layers.append(
                ResNetBlock(
                    in_channels=width,
                    out_channels=width,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    use_bias=use_bias,
                    act_fn=act_fn,
                    param_type=param_type,
                    depth_scaling=depth_scaling,
                    key=keys[key_idx]
                )
            )
            key_idx += 1
        self.layers.append(nn.AvgPool2d(kernel_size=2, stride=2))

        # Readout
        num_pools = 3
        spatial_side = input_size // (2 ** num_pools)
        in_features = width * spatial_side * spatial_side
        self.layers.append(
            Readout(
                key=keys[key_idx],
                in_features=in_features,
                out_features=out_features,
                scaling=non_res_depth_scaling / (in_features) if param_type == "mupc" else 1.0,
                param_type=param_type,
                use_bias=use_bias,
                weight_std_scale=1/non_res_depth_scaling
            )
        )

    def __call__(self, x):
        for f in self.layers:
            x = f(x)
        return x

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, idx):
        return self.layers[idx]
