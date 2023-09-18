"""
Adapted from https://github.com/KAIR-BAIR/nerfacc/blob/master/examples/radiance_fields/mlp.py

Modifications:
    1. Enable hard-coded parameters to be user-defined through class
       initialization
    2. Normalize input position according to the user-defined AABB &
       contraction type
    3. Normalize input direction unit vector to be of length pi
    4. Added support for MLP weight normalization

Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import functools
import math
from typing import Callable, Optional, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nerfacc import ContractionType


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,  # The number of input tensor channels.
        output_dim: int = None,  # The number of output tensor channels.
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        hidden_init: Callable = nn.init.xavier_uniform_,
        hidden_activation: Callable = nn.ReLU(),
        output_enabled: bool = True,
        output_init: Optional[Callable] = nn.init.xavier_uniform_,
        output_activation: Optional[Callable] = nn.Identity(),
        bias_enabled: bool = True,
        bias_init: Callable = nn.init.zeros_,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net_depth = net_depth
        self.net_width = net_width
        self.skip_layer = skip_layer
        self.hidden_init = hidden_init
        self.hidden_activation = hidden_activation
        self.output_enabled = output_enabled
        self.output_init = output_init
        self.output_activation = output_activation
        self.bias_enabled = bias_enabled
        self.bias_init = bias_init

        self.hidden_layers = nn.ModuleList()
        in_features = self.input_dim
        for i in range(self.net_depth):
            self.hidden_layers.append(
                nn.Linear(in_features, self.net_width, bias=bias_enabled)
            )
            if (
                (self.skip_layer is not None)
                and (i % self.skip_layer == 0)
                and (i > 0)
            ):
                in_features = self.net_width + self.input_dim
            else:
                in_features = self.net_width
        if self.output_enabled:
            self.output_layer = nn.Linear(
                in_features, self.output_dim, bias=bias_enabled
            )
        else:
            self.output_dim = in_features

        self.initialize()

    def initialize(self):
        def init_func_hidden(m):
            if isinstance(m, nn.Linear):
                if self.hidden_init is not None:
                    self.hidden_init(m.weight)
                if self.bias_enabled and self.bias_init is not None:
                    self.bias_init(m.bias)

        self.hidden_layers.apply(init_func_hidden)
        if self.output_enabled:

            def init_func_output(m):
                if isinstance(m, nn.Linear):
                    if self.output_init is not None:
                        self.output_init(m.weight)
                    if self.bias_enabled and self.bias_init is not None:
                        self.bias_init(m.bias)

            self.output_layer.apply(init_func_output)

    def forward(self, x):
        inputs = x
        for i in range(self.net_depth):
            x = self.hidden_layers[i](x)
            x = self.hidden_activation(x)
            if (
                (self.skip_layer is not None)
                and (i % self.skip_layer == 0)
                and (i > 0)
            ):
                x = torch.cat([x, inputs], dim=-1)
        if self.output_enabled:
            x = self.output_layer(x)
            x = self.output_activation(x)
        return x


class DenseLayer(MLP):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            net_depth=0,  # no hidden layers
            **kwargs,
        )


class NerfMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,  # The number of input tensor channels.
        condition_dim: int,  # The number of condition tensor channels.
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        net_depth_condition: int = 1,  # The depth of the second part of MLP.
        net_width_condition: int = 128,  # The width of the second part of MLP.
        radiance_dim: int = 3,
        hidden_activation: Callable = nn.ReLU(),
        hidden_init: Optional[Callable] = nn.init.xavier_uniform_,
        output_init: Optional[Callable] = nn.init.xavier_uniform_,
        bias_init: Optional[Callable] = nn.init.zeros_
    ):
        super().__init__()
        self.base = MLP(
            input_dim=input_dim,
            net_depth=net_depth,
            net_width=net_width,
            skip_layer=skip_layer,
            output_enabled=False,
            hidden_activation=hidden_activation,
            hidden_init=hidden_init,
            output_init=output_init,
            bias_init=bias_init
        )
        hidden_features = self.base.output_dim
        self.sigma_layer = DenseLayer(
            input_dim=hidden_features,
            output_dim=1,
            output_init=output_init,
            bias_init=bias_init
        )

        if condition_dim > 0:
            self.bottleneck_layer = DenseLayer(
                input_dim=hidden_features,
                output_dim=net_width,
                output_init=output_init,
                bias_init=bias_init
            )
            self.rgb_layer = MLP(
                input_dim=net_width + condition_dim,
                output_dim=radiance_dim,
                net_depth=net_depth_condition,
                net_width=net_width_condition,
                skip_layer=None,
                hidden_activation=hidden_activation,
                hidden_init=hidden_init,
                output_init=output_init,
                bias_init=bias_init
            )
        else:
            self.rgb_layer = DenseLayer(
                input_dim=hidden_features,
                output_dim=radiance_dim,
                output_init=output_init,
                bias_init=bias_init
            )

    def query_density(self, x):
        x = self.base(x)
        raw_sigma = self.sigma_layer(x)
        return raw_sigma

    def forward(self, x, condition=None):
        x = self.base(x)
        raw_sigma = self.sigma_layer(x)
        if condition is not None:
            if condition.shape[:-1] != x.shape[:-1]:
                num_rays, n_dim = condition.shape
                condition = condition.view(
                    [num_rays] + [1] * (x.dim() - condition.dim()) + [n_dim]
                ).expand(list(x.shape[:-1]) + [n_dim])
            bottleneck = self.bottleneck_layer(x)
            x = torch.cat([bottleneck, condition], dim=-1)
        raw_rgb = self.rgb_layer(x)
        return raw_rgb, raw_sigma


class SinusoidalEncoder(nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(self, x_dim, min_deg, max_deg, use_identity: bool = True):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_identity = use_identity
        self.register_buffer(
            "scales", torch.tensor([2**i for i in range(min_deg, max_deg)])
        )

    @property
    def latent_dim(self) -> int:
        return (
            int(self.use_identity) + (self.max_deg - self.min_deg) * 2
        ) * self.x_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., x_dim]
        Returns:
            latent: [..., latent_dim]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[Ellipsis, None, :] * self.scales[:, None]),
            list(x.shape[:-1]) + [(self.max_deg - self.min_deg) * self.x_dim],
        )
        latent = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim=-1))
        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)
        return latent


class VanillaNeRFRadianceField(nn.Module):
    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        net_depth_condition: int = 1,  # The depth of the second part of MLP.
        net_width_condition: int = 128,  # The width of the second part of MLP.
        num_dim: int = 3,
        contraction_type: ContractionType = ContractionType.AABB,
        radiance_dim: int = 3,
        hidden_activation: Callable = nn.ReLU(),
        density_activation: Callable = nn.ReLU(),
        radiance_activation: Callable = nn.Sigmoid(),
        pos_encoder_max_deg: int = 10,
        view_encoder_max_deg: int = 4,
        weight_norm: bool = False
    ) -> None:
        super().__init__()
        assert isinstance(contraction_type, ContractionType)
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.num_dim = num_dim
        self.contraction_type = contraction_type
        self.density_activation = density_activation
        self.radiance_activation = radiance_activation

        self.posi_encoder = SinusoidalEncoder(
            x_dim=num_dim,
            min_deg=0,
            max_deg=pos_encoder_max_deg, 
            use_identity=True
        )
        self.view_encoder = SinusoidalEncoder(
            x_dim=num_dim,
            min_deg=0,
            max_deg=view_encoder_max_deg, 
            use_identity=True
        )
        self.mlp = NerfMLP(
            input_dim=self.posi_encoder.latent_dim,
            condition_dim=self.view_encoder.latent_dim,
            net_depth=net_depth,
            net_width=net_width,
            skip_layer=skip_layer,
            net_depth_condition=net_depth_condition,
            net_width_condition=net_width_condition,
            radiance_dim=radiance_dim,
            hidden_activation=hidden_activation,
            hidden_init=None,
            output_init=None,
            bias_init=None
        )

        # apply weight norm to `self.mlp`, if required
        """
        NOTE:
            If `weight_norm=True` & `trainer.accelerator=ddp_spawn`, PyTorch
            will raise the following RuntimeError:
                "Cowardly refusing to serialize non-leaf tensor which
                requires_grad, since autograd does not support crossing
                process boundaries.  If you just want to transfer the data,
                call detach() on the tensor before serializing (e.g., putting
                it on the queue)"
            since `<mlp_layer>.weight` are no longer leaf tensors, as they are
            then parameterized by `<mlp_layer>.weight_g/v`
        """
        if not weight_norm:
            return
        for module in self.mlp.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.utils.weight_norm(module)

    def contract_input_space(self, x):
        """Contract input space to [-pi, pi]"""
        if self.contraction_type == ContractionType.UN_BOUNDED_SPHERE:
            from .ngp import contract_to_unisphere
            x = contract_to_unisphere(x, self.aabb, num_dim=self.num_dim)
        elif self.contraction_type == ContractionType.UN_BOUNDED_TANH:
            from .ngp import contract_tanh
            x = contract_tanh(x, self.aabb, num_dim=self.num_dim)
        else:   # self.contraction_type == ContractionType.AABB:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)

        x = 2 * math.pi * (x - 0.5)
        return x, selector

    def query_opacity(self, x, step_size):
        density = self.query_density(x)
        # if the density is small enough those two are the same.
        # opacity = 1.0 - torch.exp(-density * step_size)
        opacity = density * step_size
        return opacity

    def query_density(self, x):
        x, selector = self.contract_input_space(x)
        x = self.posi_encoder(x)
        sigma = self.mlp.query_density(x)
        return self.density_activation(sigma) * selector[..., None]

    def forward(self, x, condition=None):
        x, selector = self.contract_input_space(x)
        x = self.posi_encoder(x)
        if condition is not None:
            condition = condition * math.pi     # [-1, 1] is at [-pi, pi]
            condition = self.view_encoder(condition)
        rgb, sigma = self.mlp(x, condition=condition)
        return self.radiance_activation(rgb), \
               self.density_activation(sigma) * selector[..., None]
