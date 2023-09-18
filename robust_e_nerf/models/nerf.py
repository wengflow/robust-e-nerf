import easydict
import torch
import nerfacc
from ..utils import modules
from ..external import mlp, ngp, utils


def shifted_softplus(x, shift=1, beta=1, threshold=20):
    """
    Proposed by "mip-NeRF: A Multiscale Representation for Anti-Aliasing
    Neural Radiance Fields"
    """
    return torch.nn.functional.softplus(x - shift, beta, threshold)


class NeRF(torch.nn.Module):
    HIDDEN_ACTIVATION_NAME_TO_FN = {
        "softplus": torch.nn.Softplus(beta=100),
        "relu": torch.nn.ReLU()
    }
    DENSITY_ACTIVATION_NAME_TO_FN = {
        "shifted_trunc_exp": ngp.shifted_trunc_exp,
        "softplus": torch.nn.Softplus(beta=1),
        "shifted_softplus": shifted_softplus
    }
    RADIANCE_ACTIVATION_NAME_TO_FN= {
        "softplus": torch.nn.Softplus(beta=1),
        "sigmoid": torch.nn.Sigmoid(),
    }

    def __init__(
        self,
        aabb,
        contraction_type,
        occ_grid_config,
        near_plane,
        far_plane,
        render_step_size,
        render_bkgd,
        cone_angle,
        early_stop_eps,
        alpha_thre,
        test_chunk_size,
        arch,
        arch_config,
        num_dim,
        radiance_dim,
        opacity_eps=1e-10
    ):
        super().__init__()

        assert torch.all(torch.tensor(occ_grid_config.resolution) > 0)
        assert 0 <= occ_grid_config.occ_thre <= 1
        assert 0 <= occ_grid_config.ema_decay <= 1
        assert occ_grid_config.warmup_steps > 0
        assert occ_grid_config.n > 0

        if (not near_plane is None) and (not far_plane is None):
            assert 0 <= near_plane <= far_plane
        assert render_step_size > 0
        assert (render_bkgd is None) or (render_bkgd == "parameter") \
               or isinstance(render_bkgd, torch.Tensor)
        assert cone_angle >= 0
        assert 0 <= early_stop_eps <= 1
        assert 0 <= alpha_thre <= 1
        assert test_chunk_size > 0
        assert num_dim > 0
        assert radiance_dim > 0
        assert isinstance(opacity_eps, (int, float)) and opacity_eps > 0

        # save some arguments as attributes or buffers
        self.register_buffer("aabb", torch.tensor(aabb), persistent=False)      # (2 * num_dim)
        self.contraction_type = contraction_type
        self.occ_grid_config = occ_grid_config
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.register_buffer("render_step_size",
                             torch.tensor(render_step_size), persistent=False)  # ()
        if render_bkgd is None:
            self.render_bkgd = None
        elif render_bkgd == "parameter":
            softplus = modules.Softplus(beta=1)
            self.render_bkgd = torch.nn.parameter.Parameter(                    # (radiance_dim)
                torch.ones(radiance_dim)
            )
            torch.nn.utils.parametrize.register_parametrization(
                self, "render_bkgd", softplus
            )
        else:
            self.register_buffer("render_bkgd", render_bkgd, persistent=False)  # (radiance_dim)
        self.cone_angle = cone_angle
        self.early_stop_eps = early_stop_eps
        self.alpha_thre = alpha_thre
        self.test_chunk_size = test_chunk_size
        self.opacity_eps = opacity_eps

        # instantiate the occupancy grid
        self.occupancy_grid = nerfacc.OccupancyGrid(
            roi_aabb=aabb,
            resolution=occ_grid_config.resolution,
            contraction_type=contraction_type
        )

        # instantiate the specified neural radiance field model
        if arch == "ngp":
            # update `mlp_base` configurations
            mlp_base_config = easydict.EasyDict(arch_config.mlp_base)
            mlp_base_config.hidden_activation = (
                self.HIDDEN_ACTIVATION_NAME_TO_FN[
                    arch_config.mlp_base.hidden_activation
                ]
            )
            mlp_base_config.density_activation = (
                self.DENSITY_ACTIVATION_NAME_TO_FN[
                    arch_config.mlp_base.density_activation
                ]
            )

            # update `mlp_head` configurations
            mlp_head_config = easydict.EasyDict(arch_config.mlp_head)
            mlp_head_config.hidden_activation = (
                self.HIDDEN_ACTIVATION_NAME_TO_FN[
                    arch_config.mlp_head.hidden_activation
                ]
            )
            mlp_head_config.radiance_activation = (
                self.RADIANCE_ACTIVATION_NAME_TO_FN[
                    arch_config.mlp_head.radiance_activation
                ]
            )
            mlp_head_config.output_dim = radiance_dim

            self.radiance_field = ngp.NGPradianceField(
                aabb=aabb,
                num_dim=num_dim,
                use_viewdirs=True,
                contraction_type=contraction_type,
                pos_encoding_config=arch_config.pos_encoding,
                dir_encoding_config=arch_config.dir_encoding,
                mlp_base_config=mlp_base_config,
                mlp_head_config=mlp_head_config
            )
        elif arch == "mlp":
            self.radiance_field = mlp.VanillaNeRFRadianceField(
                aabb=aabb,
                net_depth=arch_config.net_depth,
                net_width=arch_config.net_width,
                skip_layer=arch_config.skip_layer,
                net_depth_condition=arch_config.net_depth_condition,
                net_width_condition=arch_config.net_width_condition,
                num_dim=num_dim,
                contraction_type=contraction_type,
                radiance_dim=radiance_dim,
                hidden_activation=self.HIDDEN_ACTIVATION_NAME_TO_FN[
                                    arch_config.hidden_activation
                                  ],
                density_activation=self.DENSITY_ACTIVATION_NAME_TO_FN[
                                    arch_config.density_activation
                                   ],
                radiance_activation=self.RADIANCE_ACTIVATION_NAME_TO_FN[
                                        arch_config.radiance_activation
                                    ],
                pos_encoder_max_deg=arch_config.pos_encoder_max_deg,
                view_encoder_max_deg=arch_config.view_encoder_max_deg,
                weight_norm=arch_config.weight_norm
            )
        else:
            raise NotImplementedError

    def update_occ_grid(self, step, T_wc_position):
        def occ_eval_fn(x):
            """
            Adapted from https://github.com/KAIR-BAIR/nerfacc/blob/master/examples/train_ngp_nerf.py#L190-L213
            """
            if self.cone_angle > 0.0:
                # randomly sample a camera for computing step size.
                camera_ids = torch.randint(
                    0, len(T_wc_position), (x.shape[0],),
                    device=T_wc_position.device
                )
                origins = T_wc_position[camera_ids, :]
                t = (origins - x).norm(dim=-1, keepdim=True)
                # compute actual step size used in marching, based on the distance to the camera.
                step_size = torch.clamp(
                    t * self.cone_angle, min=self.render_step_size
                )
                # filter out the points that are not in the near far plane.
                if (self.near_plane is not None) and (self.far_plane is not None):
                    step_size = torch.where(
                        (t > self.near_plane) & (t < self.far_plane),
                        step_size,
                        torch.zeros_like(step_size),
                    )
            else:
                step_size = self.render_step_size
            # compute occupancy
            density = self.radiance_field.query_density(x)
            return density * step_size

        self.occupancy_grid.every_n_step(
            step, occ_eval_fn,
            self.occ_grid_config.occ_thre, self.occ_grid_config.ema_decay,
            self.occ_grid_config.warmup_steps, self.occ_grid_config.n
        )

    @staticmethod
    def pixel_params_to_ray(
        intrinsics_inverse,                                                     # ([[M,] N,] 3, 3)
        pixel_position,                                                         # ([M,] N, 2)
        T_wc_position,                                                          # ([M,] N, 3)
        T_wc_orientation                                                        # ([M,] N, 3, 3)
    ):
        homogenous_pixel_position = torch.cat(                                  # ([M,] N, 3)
            ( pixel_position,                                                   # ([M,] N, 2)
              torch.ones_like(pixel_position[..., 0].unsqueeze(dim=-1)) ),      # ([M,] N, 1)
            dim=-1
        )
        homogenous_pixel_position = homogenous_pixel_position.unsqueeze(dim=-1) # ([M,] N, 3, 1)
        ray_direction = T_wc_orientation @ (                                    # ([M,] N, 3, 1)
            intrinsics_inverse @ homogenous_pixel_position
        )
        ray_direction = ray_direction.squeeze(dim=-1)                           # ([M,] N, 3)
        ray_direction = ray_direction / torch.linalg.vector_norm(               # ([M,] N, 3)
            ray_direction, dim=-1, keepdim=True
        )
        ray_origin = T_wc_position                                              # ([M,] N, 3)

        return ray_origin, ray_direction

    def forward(self, ray_origin, ray_direction):
        """
        Args:
            ray_origin (torch.Tensor): Ray origin positions of shape
                                       ([M,] N, num_dim)
            ray_direction (torch.Tensor): Ray direction unit vectors of shape
                                          ([M,] N, num_dim)
        Returns:
            radiance (torch.Tensor): Volume-rendered radiance values of shape
                                     ([M,] N [, radiance_dim])
            opacity (torch.Tensor): Volume rendering ray segment opacities of
                                    shape ([M,] N)
            depth (torch.Tensor): Expected ray termination depth of shape
                                  ([M,] N)
            mean_num_samples_per_ray (float): Mean number of samples used per
                                              ray for volume rendering
        """
        rays = utils.Rays(origins=ray_origin, viewdirs=ray_direction)
        if self.contraction_type == nerfacc.ContractionType.AABB:
            ray_marching_aabb = self.aabb
        else:
            ray_marching_aabb = None
        radiance, opacity, depth, num_samples_across_rays = utils.render_image( # ([M,] N, radiance_dim), ([M,] N, 1), ([M,] N, 1), int, dict
            self.radiance_field,
            self.occupancy_grid,
            rays,
            ray_marching_aabb,
            self.near_plane,
            self.far_plane,
            self.render_step_size,
            self.render_bkgd,
            self.cone_angle,
            self.early_stop_eps,
            self.alpha_thre,
            self.test_chunk_size
        )

        """
        NOTE:
            In practice, the volume rendering ray segment opacities will be
            less than 1. Thus, the rendering samples weights need to be
            normalized by the opacities for a more accurate expected ray
            termination depth estimation. This is not done for the rendered
            radiance since either a background radiance has been alpha over, or
            we assume the radiance on the ray beyond the boundaries are zero.

        Reference:
            https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/model_components/renderers.py#L253
        """
        radiance = radiance.squeeze(dim=-1)                                     # ([M,] N [, radiance_dim])
        opacity = opacity.squeeze(dim=-1)                                       # ([M,] N)
        depth = depth.squeeze(dim=-1)                                           # ([M,] N)
        depth = depth / (opacity + self.opacity_eps)
        num_rays = ray_origin.numel() // ray_origin.shape[-1]
        mean_num_samples_per_ray = num_samples_across_rays / num_rays

        return radiance, opacity, depth, mean_num_samples_per_ray
