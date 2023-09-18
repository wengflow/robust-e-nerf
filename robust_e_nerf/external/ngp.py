"""
Adapted from https://github.com/KAIR-BAIR/nerfacc/blob/master/examples/radiance_fields/ngp.py

Modifications:
    1. Replace `tcnn.NetworkWithInputEncoding` of `self.mlp_base` with
       `tcnn.Encoding` + `MLP` [Ref. 1]
    2. Replace `tcnn.Network` of `self.mlp_head` with `MLP` [Ref. 1]
    3. Enable hard-coded parameters to be user-defined through class
       initialization
    4. Added support of `ContractionType.UN_BOUNDED_TANH` space contraction
    5. Added support for MLP weight normalization
    6. Implemented `shifted_trunc_exp()`
    7. Replace in-place operations in `contract_to_unisphere()` to out-of-place
    8. Replace `tcnn.Encoding` direction encoding with `SHEncoder` [Ref. 2]
    9. Specified preferred precision of `torch.float32` for `tcnn.Encoding`

References:
    1. https://github.com/NVlabs/tiny-cuda-nn/issues/131
    2. https://github.com/NVlabs/tiny-cuda-nn/issues/232

Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from typing import Callable, List, Union

import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

from .mlp import MLP
from .sh_encoder import SHEncoder
from nerfacc import ContractionType

try:
    import tinycudann as tcnn
except ImportError as e:
    print(
        f"Error: {e}! "
        "Please install tinycudann by: "
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    )
    exit()


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


def shifted_trunc_exp(x, shift=1):
    return trunc_exp(x - shift)


def contract_to_unisphere(
    x: torch.Tensor,
    aabb: torch.Tensor,
    eps: float = 1e-6,
    derivative: bool = False,
    num_dim: int = 3
):
    aabb_min, aabb_max = torch.split(aabb, num_dim, dim=-1)
    x = (x - aabb_min) / (aabb_max - aabb_min)
    x = x * 2 - 1                   # aabb is at [-1, 1]
    mag = x.norm(dim=-1, keepdim=True)
    mask = mag > 1

    if derivative:
        dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
            1 / mag**3 - (2 * mag - 1) / mag**4
        )
        dev = torch.where(
            mask, dev, torch.tensor(1, dtype=dev.dtype, device=dev.device)
        )
        dev = torch.clamp(dev, min=eps)
        return dev
    else:
        x = torch.where(mask, (2 - 1 / mag) * (x / mag), x)
        x = x / 4 + 0.5             # [-inf, inf] is at [0, 1]
        return x


def contract_tanh(
    x: torch.Tensor,
    aabb: torch.Tensor,
    num_dim: int = 3
):
    aabb_min, aabb_max = torch.split(aabb, num_dim, dim=-1)
    x = (x - aabb_min) / (aabb_max - aabb_min)
    x = x - 0.5                     # aabb is at [-0.5, 0.5]

    x = (torch.tanh(x) + 1) / 2     # [-inf, inf] is at [0, 1]
    return x


class NGPradianceField(torch.nn.Module):
    """Instance-NGP radiance Field"""

    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        num_dim: int = 3,
        use_viewdirs: bool = True,
        contraction_type: ContractionType = ContractionType.AABB,
        pos_encoding_config: dict = {
            "otype": "HashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": 16,
            "per_level_scale": 1.4472692012786865,
            "interpolation": "Linear"
        },
        dir_encoding_config: dict = {
            "degree": 4
        },
        mlp_base_config: dict = {
            "hidden_activation": torch.nn.ReLU(),
            "density_activation": shifted_trunc_exp,
            "n_neurons": 64,
            "n_hidden_layers": 1,
            "geo_feat_dim": 15,
            "weight_norm": False
        },
        mlp_head_config: dict = {
            "hidden_activation": torch.nn.ReLU(),
            "radiance_activation": torch.nn.Sigmoid(),
            "n_neurons": 64,
            "n_hidden_layers": 2,
            "output_dim": 3,
            "weight_norm": False
        }
    ) -> None:
        super().__init__()
        assert num_dim == 3 # hard contraint due to spherical harmonics dir enc
        assert isinstance(contraction_type, ContractionType)
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.num_dim = num_dim
        self.radiance_dim = mlp_head_config["output_dim"]
        self.use_viewdirs = use_viewdirs
        self.contraction_type = contraction_type
        self.density_activation = mlp_base_config["density_activation"]
        self.geo_feat_dim = mlp_base_config["geo_feat_dim"]

        if self.use_viewdirs:
            self.direction_encoding = SHEncoder(
                n_input_dims=num_dim,
                degree=dir_encoding_config["degree"]
            )

        position_encoding = tcnn.Encoding(
            n_input_dims=num_dim,
            encoding_config=pos_encoding_config,
            dtype=torch.float32
        )
        self.mlp_base = torch.nn.Sequential(
            position_encoding,
            MLP(
                input_dim=position_encoding.n_output_dims,
                output_dim=1 + mlp_base_config["geo_feat_dim"],
                net_depth=mlp_base_config["n_hidden_layers"],
                net_width=mlp_base_config["n_neurons"],
                skip_layer=None,
                hidden_init=None,
                hidden_activation=mlp_base_config["hidden_activation"],
                output_enabled=True,
                output_init=None,
                output_activation=torch.nn.Identity(),
                bias_enabled=True,
                bias_init=None,
            )
        )
        if self.geo_feat_dim > 0:
            self.mlp_head = MLP(
                input_dim=((self.direction_encoding.n_output_dims
                            if self.use_viewdirs
                            else 0)
                           + self.geo_feat_dim),
                output_dim=mlp_head_config["output_dim"],
                net_depth=mlp_head_config["n_hidden_layers"],
                net_width=mlp_head_config["n_neurons"],
                skip_layer=None,
                hidden_init=None,
                hidden_activation=mlp_head_config["hidden_activation"],
                output_enabled=True,
                output_init=None,
                output_activation=mlp_head_config["radiance_activation"],
                bias_enabled=True,
                bias_init=None,
            )

        # apply weight norm to `self.mlp_base/head`, if required
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
        for mlp, mlp_config in zip(
            [ self.mlp_base, self.mlp_head ],
            [ mlp_base_config, mlp_head_config ]
        ):
            if not mlp_config["weight_norm"]:
                continue
            for module in mlp.modules():
                if isinstance(module, torch.nn.Linear):
                    torch.nn.utils.weight_norm(module)

    def query_density(self, x, return_feat: bool = False):
        if self.contraction_type == ContractionType.UN_BOUNDED_SPHERE:
            x = contract_to_unisphere(x, self.aabb, num_dim=self.num_dim)
        elif self.contraction_type == ContractionType.UN_BOUNDED_TANH:
            x = contract_tanh(x, self.aabb, num_dim=self.num_dim)
        else:   # self.contraction_type == ContractionType.AABB:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)
        x = (
            self.mlp_base(x.view(-1, self.num_dim))
            .view(list(x.shape[:-1]) + [1 + self.geo_feat_dim])
            .to(x)
        )
        density_before_activation, base_mlp_out = torch.split(
            x, [1, self.geo_feat_dim], dim=-1
        )
        density = (
            self.density_activation(density_before_activation)
            * selector[..., None]
        )
        if return_feat:
            return density, base_mlp_out
        else:
            return density

    def _query_rgb(self, dir, embedding):
        if self.use_viewdirs:
            d = self.direction_encoding(dir.view(-1, dir.shape[-1]))
            h = torch.cat([d, embedding.view(-1, self.geo_feat_dim)], dim=-1)
        else:
            h = embedding.view(-1, self.geo_feat_dim)
        rgb = (
            self.mlp_head(h)
            .view(list(embedding.shape[:-1]) + [self.radiance_dim])
            .to(embedding)
        )
        return rgb

    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor = None,
    ):
        if self.use_viewdirs and (directions is not None):
            assert (
                positions.shape == directions.shape
            ), f"{positions.shape} v.s. {directions.shape}"
        density, embedding = self.query_density(positions, return_feat=True)
        rgb = self._query_rgb(directions, embedding=embedding)
        return rgb, density
