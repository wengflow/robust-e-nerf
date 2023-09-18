"""
Adapted from https://github.com/KAIR-BAIR/nerfacc/blob/master/nerfacc/vol_rendering.py#L15-L129

Modifications:
    1. Support monochrome radiance field

Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from typing import Callable, Optional, Tuple
import torch
from nerfacc import render_weight_from_density, render_weight_from_alpha, \
                    accumulate_along_rays


def rendering(
    # ray marching results
    t_starts: torch.Tensor,
    t_ends: torch.Tensor,
    ray_indices: torch.Tensor,
    n_rays: int,
    # radiance field
    rgb_sigma_fn: Optional[Callable] = None,
    rgb_alpha_fn: Optional[Callable] = None,
    # rendering options
    render_bkgd: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render the rays through the radience field defined by `rgb_sigma_fn`.

    This function is differentiable to the outputs of `rgb_sigma_fn` so it can 
    be used for gradient-based optimization.

    Note:
        Either `rgb_sigma_fn` or `rgb_alpha_fn` should be provided. 

    Warning:
        This function is not differentiable to `t_starts`, `t_ends` and `ray_indices`.

    Args:
        t_starts: Per-sample start distance. Tensor with shape (n_samples, 1).
        t_ends: Per-sample end distance. Tensor with shape (n_samples, 1).
        ray_indices: Ray index of each sample. IntTensor with shape (n_samples).
        n_rays: Total number of rays. This will decide the shape of the ouputs.
        rgb_sigma_fn: A function that takes in samples {t_starts (N, 1), t_ends (N, 1), \
            ray indices (N,)} and returns the post-activation rgb (N, 1/3) and density \
            values (N, 1). 
        rgb_alpha_fn: A function that takes in samples {t_starts (N, 1), t_ends (N, 1), \
            ray indices (N,)} and returns the post-activation rgb (N, 1/3) and opacity \
            values (N, 1).
        render_bkgd: Optional. Background color. Tensor with shape (1/3,).

    Returns:
        Ray colors (n_rays, 1/3), opacities (n_rays, 1) and depths (n_rays, 1).

    Examples:

    .. code-block:: python

        >>> rays_o = torch.rand((128, 3), device="cuda:0")
        >>> rays_d = torch.randn((128, 3), device="cuda:0")
        >>> rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
        >>> ray_indices, t_starts, t_ends = ray_marching(
        >>>     rays_o, rays_d, near_plane=0.1, far_plane=1.0, render_step_size=1e-3)
        >>> def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        >>>     # This is a dummy function that returns random values.
        >>>     rgbs = torch.rand((t_starts.shape[0], 3), device="cuda:0")
        >>>     sigmas = torch.rand((t_starts.shape[0], 1), device="cuda:0")
        >>>     return rgbs, sigmas
        >>> colors, opacities, depths = rendering(
        >>>     t_starts, t_ends, ray_indices, n_rays=128, rgb_sigma_fn=rgb_sigma_fn)
        >>> print(colors.shape, opacities.shape, depths.shape)
        torch.Size([128, 3]) torch.Size([128, 1]) torch.Size([128, 1])

    """
    if rgb_sigma_fn is None and rgb_alpha_fn is None:
        raise ValueError(
            "At least one of `rgb_sigma_fn` and `rgb_alpha_fn` should be specified."
        )

    # Query sigma/alpha and color with gradients
    if rgb_sigma_fn is not None:
        rgbs, sigmas = rgb_sigma_fn(t_starts, t_ends, ray_indices.long())
        assert rgbs.shape[-1] in (1, 3), \
               "rgbs must have 1 or 3 channels, got {}".format(rgbs.shape)
        assert (
            sigmas.shape == t_starts.shape
        ), "sigmas must have shape of (N, 1)! Got {}".format(sigmas.shape)
        # Rendering: compute weights.
        weights = render_weight_from_density(
            t_starts,
            t_ends,
            sigmas,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
    elif rgb_alpha_fn is not None:
        rgbs, alphas = rgb_alpha_fn(t_starts, t_ends, ray_indices.long())
        assert rgbs.shape[-1] in (1, 3), \
               "rgbs must have 1 or 3 channels, got {}".format(rgbs.shape)
        assert (
            alphas.shape == t_starts.shape
        ), "alphas must have shape of (N, 1)! Got {}".format(alphas.shape)
        # Rendering: compute weights.
        weights = render_weight_from_alpha(
            alphas,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )

    # Rendering: accumulate rgbs, opacities, and depths along the rays.
    colors = accumulate_along_rays(
        weights, ray_indices, values=rgbs, n_rays=n_rays
    )
    opacities = accumulate_along_rays(
        weights, ray_indices, values=None, n_rays=n_rays
    )
    depths = accumulate_along_rays(
        weights,
        ray_indices,
        values=(t_starts + t_ends) / 2.0,
        n_rays=n_rays,
    )

    # Background composition.
    if render_bkgd is not None:
        colors = colors + render_bkgd * (1.0 - opacities)

    return colors, opacities, depths
