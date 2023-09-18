import os
import math
import easydict
import torch
import pytorch_lightning as pl
import nerfacc
import numpy as np
import cv2

from . import event_generation_params, nerf, trajectories
from .. import loss_metric
from ..utils import autograd, modules, tensor_ops
from ..data import datasets


class RobustENeRF(pl.LightningModule):
    INTRINSICS_KEY = "intrinsics"
    NUM_DIM = 3
    MAX_NUM_SAMPLES_PER_RAY = 1024
    PREDICTIONS_FOLDER_NAME = "predictions"
    PREDICTION_FILE_EXTENSION = ".png"
    PREDICTION_BIT_DEPTH = 8
    MODEL_COMPONENTS = [ "contrast_threshold", "refractory_period", "nerf" ]

    def __init__(
        self,
        git_head_hash,
        eval_target,
        num_nodes,
        gpus,
        min_modeled_intensity,
        eval_save_pred_intensity_img,
        checkpoint_filepath,
        contrast_threshold,
        refractory_period,
        nerf,
        loss,
        metric,
        optimizer,
        lr_scheduler,
        dataset_directory,
        alpha_over_white_bg,
        train_eff_ray_sample_batch_size
    ):
        super().__init__()
        assert isinstance(min_modeled_intensity, (int, float)) \
               and min_modeled_intensity > 0
        for model_component_config in [
            contrast_threshold, refractory_period, nerf
        ]:
            assert isinstance(model_component_config.load_state_dict, bool)
            assert isinstance(model_component_config.freeze, bool)
        assert isinstance(train_eff_ray_sample_batch_size, int) \
               and train_eff_ray_sample_batch_size > 0

        # parameters & buffers of a model component can only be frozen, if
        # it is loaded from a checkpoint or appropriately / not randomly init.
        if nerf.freeze:
            assert nerf.load_state_dict

        # save some non-hyperparameter configs or its derivatives as attributes
        # or buffers
        num_gpus_per_node = len(gpus)
        num_gpus = num_nodes * num_gpus_per_node
        self.train_ray_sample_batch_size = train_eff_ray_sample_batch_size \
                                           // num_gpus
        self.eval_save_pred_intensity_img = eval_save_pred_intensity_img

        camera_calibration = datasets.Event.load_camera_calibration(
            dataset_directory
        )
        bayer_pattern = str(
            camera_calibration[datasets.Event.BAYER_PATTERN_KEY]
        )
        self.has_bayer_filter = (
            bayer_pattern != datasets.Event.NULL_BAYER_PATTERN
        )
        self.register_buffer(                                                   # (3, 3)
            "train_intrinsics_inv",
            torch.linalg.inv(
                torch.from_numpy(camera_calibration[self.INTRINSICS_KEY])
            ),
            persistent=False
        )

        if set(eval_target) == set([ "event_view" ]):
            val_posed_image_stage = "train"
        elif set(eval_target) == set([ "novel_view" ]):
            val_posed_image_stage = "val"
        else:
            raise NotImplementedError
        val_posed_imgs = datasets.PosedImage(
            dataset_directory, val_posed_image_stage, permutation_seed=None
        )

        val_img_height, val_img_width = (
            val_posed_imgs.posed_imgs.img.shape[-2:]
        )
        self.val_min_normalized_pixel_value = (
            val_posed_imgs.min_normalized_pixel_value
        )
        self.val_max_normalized_pixel_value = (
            val_posed_imgs.max_normalized_pixel_value
        )
        self.register_buffer(
            "val_intrinsics_inv",
            val_posed_imgs.posed_imgs.intrinsics.inverse(),
            persistent=False
        )
        self.register_buffer(                                                   # (self.val_img_height, self.val_img_width, 2)
            "val_img_pixel_pos",
            torch.stack(torch.meshgrid(torch.arange(val_img_width),
                                        torch.arange(val_img_height),
                                        indexing="xy"),
                        dim=2).to(torch.get_default_dtype()),
            persistent=False
        )

        try:
            if set(eval_target) == set([ "event_view" ]):
                test_posed_imgs = val_posed_imgs
            elif set(eval_target) == set([ "novel_view" ]):
                test_posed_imgs = datasets.PosedImage(
                    dataset_directory, "test", permutation_seed=None
                )
            else:
                raise NotImplementedError

            test_img_height, test_img_width = (
                test_posed_imgs.posed_imgs.img.shape[-2:]
            )
            self.test_min_normalized_pixel_value = (
                test_posed_imgs.min_normalized_pixel_value
            )
            self.test_max_normalized_pixel_value = (
                test_posed_imgs.max_normalized_pixel_value
            )
            self.register_buffer(
                "test_intrinsics_inv",
                test_posed_imgs.posed_imgs.intrinsics.inverse(),
                persistent=False
            )
            self.register_buffer(                                               # (self.test_img_height, self.test_img_width, 2)
                "test_img_pixel_pos",
                torch.stack(torch.meshgrid(torch.arange(test_img_width),
                                           torch.arange(test_img_height),
                                           indexing="xy"),
                            dim=2).to(torch.get_default_dtype()),
                persistent=False
            )
        except FileNotFoundError:
            pass

        # set the render background as a parameter due to affine ambiguity
        # in the reconstructed log intensity, if applicable
        if alpha_over_white_bg:
            self.render_bkgd = "parameter"
        else:
            self.render_bkgd = None

        # log, checkpoint & save hyperparameters to `hparams` attribute
        self.save_hyperparameters(
            "git_head_hash",
            "min_modeled_intensity",
            "checkpoint_filepath",
            "contrast_threshold",
            "refractory_period",
            "nerf",
            "loss",
            "metric",
            "optimizer",
            "lr_scheduler",
        )

        # instantiate model components
        self.contrast_threshold = self._build_contrast_threshold(
            dataset_directory
        )
        self.refractory_period = self._build_refractory_period(
            dataset_directory
        )

        camera_poses = datasets.CameraPose(
            dataset_directory, None
        )
        self.nerf = self._build_nerf(camera_poses, gpus)
        self.trajectory = self._build_trajectory(camera_poses)

        # load parameters & buffers of model components from a checkpoint &
        # freeze them, if required
        self._load_model_component_state_dicts()
        self._freeze_model_components()

        # instantiate loss components
        self.loss = loss_metric.loss.Loss(loss.weight, loss.error_fn)
        self.metric = loss_metric.metric.Metric(metric.lpips_net)

    def _build_contrast_threshold(self, dataset_directory):
        return event_generation_params.ContrastThreshold(dataset_directory)

    def _build_refractory_period(self, dataset_directory):
        return event_generation_params.RefractoryPeriod(dataset_directory)
    
    def _build_nerf(self, camera_poses, gpus):
        # deduce some NeRF model arguments
        if self.hparams.nerf.aabb == "auto":
            aabb = torch.cat(
                ( camera_poses.camera_poses.T_wc_position.min(dim=0).values,
                  camera_poses.camera_poses.T_wc_position.max(dim=0).values )
            ).tolist()
        else:
            aabb = self.hparams.nerf.aabb

        contraction_type = {
            "aabb": nerfacc.ContractionType.AABB,
            "sphere": nerfacc.ContractionType.UN_BOUNDED_SPHERE,
            "tanh": nerfacc.ContractionType.UN_BOUNDED_TANH
        }[self.hparams.nerf.contraction_type]

        if self.hparams.nerf.render_step_size == "auto":
            aabb_min = torch.tensor(aabb[:self.NUM_DIM])
            aabb_max = torch.tensor(aabb[self.NUM_DIM:])
            render_step_size = (
                math.sqrt(self.NUM_DIM) * torch.max(aabb_max - aabb_min).item()
                / self.MAX_NUM_SAMPLES_PER_RAY
            )
        else:
            render_step_size = self.hparams.nerf.render_step_size

        if self.has_bayer_filter:
            radiance_dim = 3
        else:
            radiance_dim = 1

        # instantiate NeRF model
        """
        NOTE:
            Temporary fix for uncontrollable default GPU memory allocation on
            device 0, during the initialization of `tcnn.modules.Module`
            (eg. `tcnn.Encoding` in `NGPradianceField`). It does not resolve
            multi-GPU & multi-node cases.
        """
        with torch.cuda.device(gpus[0]):
            return nerf.NeRF(
                aabb,
                contraction_type,
                self.hparams.nerf.occ_grid,
                self.hparams.nerf.near_plane,
                self.hparams.nerf.far_plane,
                render_step_size,
                self.render_bkgd,
                self.hparams.nerf.cone_angle,
                self.hparams.nerf.early_stop_eps,
                self.hparams.nerf.alpha_thre,
                self.hparams.nerf.test_chunk_size,
                self.hparams.nerf.arch,
                arch_config=self.hparams.nerf[self.hparams.nerf.arch],
                num_dim=self.NUM_DIM,
                radiance_dim=radiance_dim
            )

    def _build_trajectory(self, camera_poses):
        return trajectories.LinearTrajectory(camera_poses)

    def _load_model_component_state_dicts(self):
        # return, if none of the model components are required to load
        # parameters & buffers from a checkpoint
        requires_load = any(
            self.hparams[model_component].load_state_dict
            for model_component in self.MODEL_COMPONENTS
        )
        if not requires_load:
            return

        checkpoint = torch.load(
            self.hparams.checkpoint_filepath, map_location=torch.device('cpu')
        )
        for model_component in self.MODEL_COMPONENTS:
            if self.hparams[model_component].load_state_dict:
                getattr(self, model_component).load_state_dict(
                    modules.extract_descendent_state_dict(
                        checkpoint["state_dict"], model_component
                    )
                )
                print(f"Loaded the state dictionary of \"{model_component}\""
                       " from checkpoint!")

    def _freeze_model_components(self):
        for model_component in self.MODEL_COMPONENTS:
            if self.hparams[model_component].freeze:
                modules.freeze(getattr(self, model_component))
                print(f"Frozen the parameters of \"{model_component}\"!")

    def forward(self, batch):
        pass

    def on_train_epoch_start(self):
        # empty unoccupied cached GPU memory
        torch.cuda.empty_cache()

    def training_step(self, batch, batch_index):
        batch = easydict.EasyDict(batch)
        batch.size = batch.event.start_ts.numel()
        for normalized_samples in batch.normalized.values():
            assert normalized_samples.shape[1] == batch.size

        # remove the extra batch dim. of 1 on raw events & normalized samples
        for key, value in batch.event.items():
            batch.event[key] = value.squeeze(dim=0)                             # (batch.size, ...)
        for key, value in batch.normalized.items():
            batch.normalized[key] = value.squeeze(dim=0)                        # (batch.size)        

        # cast the event color channel indices to `torch.int64`, if relevant
        if self.has_bayer_filter:
            batch.event.channel_idx = batch.event.channel_idx.to(torch.int64)

        # correct the raw events based on a realistic event generation model
        batch.event = self.contrast_threshold(batch.event)
        batch.event = self.refractory_period(batch.event)

        # derive the ts. for log-intensity / log-intensity gradient supervision
        if self.hparams.loss.weight.log_intensity_diff > 0:
            batch.diff = {}
            batch.diff.ts_diff = (                                              # (batch.size)
                (batch.event.end_ts - batch.event.start_ts)
                * batch.normalized.ts_diff
            )
            batch.diff.start_ts = torch.lerp(                                   # (batch.size)
                input=batch.event.start_ts,
                end=torch.max(batch.event.end_ts - batch.diff.ts_diff,
                              batch.event.start_ts),                            # to alleviate issues due to numerical errors
                weight=batch.normalized.diff_start_ts
            )
            batch.diff.end_ts = torch.min(                                      # (batch.size)
                batch.diff.start_ts + batch.diff.ts_diff, batch.event.end_ts    # to alleviate issues due to numerical errors
            )
        else:
            batch.diff = None

        if self.hparams.loss.weight.log_intensity_grad > 0:
            batch.grad = {}
            if self.hparams.loss.weight.log_intensity_diff == 0:
                start_ts = batch.event.start_ts
                end_ts = batch.event.end_ts
            else:   # elif self.hparams.loss.weight.log_intensity_diff > 0:
                start_ts = batch.diff.start_ts
                end_ts = batch.diff.end_ts
            batch.grad.ts = torch.lerp(                                         # (batch.size)
                input=start_ts, end=end_ts, weight=batch.normalized.grad_ts
            )
            del start_ts
            del end_ts
            
            # require gradients to compute log-intensity gradients at this ts
            batch.grad.ts.requires_grad_()
        else:
            batch.grad = None
        batch.pop("normalized")

        # infer the camera poses / extrinsics at the supervision timestamps
        if self.hparams.loss.weight.log_intensity_grad > 0:
            batch.grad.T_wc_position, batch.grad.T_wc_orientation = (           # (batch.size, 3), (batch.size, 3, 3)
                self.trajectory(batch.grad.ts)
            )
        if self.hparams.loss.weight.log_intensity_diff > 0:
            batch.diff.start_T_wc_position, batch.diff.start_T_wc_orientation \
                = self.trajectory(batch.diff.start_ts)                          # (batch.size, 3), (batch.size, 3, 3)
            batch.diff.end_T_wc_position, batch.diff.end_T_wc_orientation \
                = self.trajectory(batch.diff.end_ts)                            # (batch.size, 3), (batch.size, 3, 3)
            batch.diff.pop("start_ts")
            batch.diff.pop("end_ts")

        # update the NeRF acceleration occupancy grid, only if this is the
        # first batch of the gradient accumulation batches
        if (batch_index % self.trainer.accumulate_grad_batches == 0):
            self.nerf.update_occ_grid(
                step=self.global_step,
                T_wc_position=self.trajectory.T_wc_position                     # (C, 3)
            )

        # infer the log-intensity / log-intensity gradients of the pixels at
        # the supervision camera poses
        if self.hparams.loss.weight.log_intensity_grad > 0:
            batch.grad.intensity, batch.grad.opacity, _, \
            batch.grad.mean_num_samples_per_ray, \
            batch.grad.is_valid = self.render_pixels(                           # ([3,] batch.size), (batch.size), float, (batch.size)
                self.train_intrinsics_inv, batch.event.position,
                batch.grad.T_wc_position, batch.grad.T_wc_orientation
            )
            if self.has_bayer_filter:
                batch.grad.intensity = self.bayering(                           # (batch.size)
                    batch.grad.intensity, batch.event.channel_idx
                )
            batch.grad.log_intensity = batch.grad.intensity.log()               # (batch.size)
            batch.grad.log_intensity_grad = autograd.gradient(                  # (batch.size)
                outputs=batch.grad.log_intensity, inputs=batch.grad.ts,
                create_graph=True
            )
            batch.grad.mean_ray_occ_rate = torch.mean(
                batch.grad.opacity > 0, dtype=torch.get_default_dtype()
            )

            del _
            batch.grad.pop("ts")
            batch.grad.pop("T_wc_position")
            batch.grad.pop("T_wc_orientation")
            batch.grad.pop("intensity")
            batch.grad.pop("log_intensity")
            batch.grad.pop("opacity")
        if self.hparams.loss.weight.log_intensity_diff > 0:
            batch.diff.start_intensity, batch.diff.start_opacity, _, \
            batch.diff.start_mean_num_samples_per_ray, \
            batch.diff.is_start_valid = self.render_pixels(                     # ([3,] batch.size), (batch.size), float, (batch.size)
                self.train_intrinsics_inv, batch.event.position,
                batch.diff.start_T_wc_position,
                batch.diff.start_T_wc_orientation
            )
            batch.diff.end_intensity, batch.diff.end_opacity, _, \
            batch.diff.end_mean_num_samples_per_ray, \
            batch.diff.is_end_valid, = self.render_pixels(                      # ([3,] batch.size), (batch.size), float, (batch.size)
                self.train_intrinsics_inv, batch.event.position,
                batch.diff.end_T_wc_position,
                batch.diff.end_T_wc_orientation
            )
            if self.has_bayer_filter:
                batch.diff.start_intensity = self.bayering(                     # (batch.size)
                    batch.diff.start_intensity, batch.event.channel_idx
                )
                batch.diff.end_intensity = self.bayering(                       # (batch.size)
                    batch.diff.end_intensity, batch.event.channel_idx
                )
            batch.diff.start_log_intensity = batch.diff.start_intensity.log()   # (batch.size)
            batch.diff.end_log_intensity = batch.diff.end_intensity.log()       # (batch.size)
            batch.diff.log_intensity_diff = batch.diff.end_log_intensity \
                                            - batch.diff.start_log_intensity    # (batch.size)
            batch.diff.start_mean_ray_occ_rate = torch.mean(
                batch.diff.start_opacity > 0, dtype=torch.get_default_dtype()
            )
            batch.diff.end_mean_ray_occ_rate = torch.mean(
                batch.diff.end_opacity > 0, dtype=torch.get_default_dtype()
            )
            batch.diff.is_valid = batch.diff.is_start_valid \
                                  | batch.diff.is_end_valid                     # (batch.size)

            del _
            batch.diff.pop("start_T_wc_position")
            batch.diff.pop("start_T_wc_orientation")
            batch.diff.pop("end_T_wc_position")
            batch.diff.pop("end_T_wc_orientation")
            batch.diff.pop("start_intensity")
            batch.diff.pop("end_intensity")
            batch.diff.pop("start_log_intensity")
            batch.diff.pop("end_log_intensity")
            batch.diff.pop("start_opacity")
            batch.diff.pop("end_opacity")
        batch.event.pop("position")
        if self.has_bayer_filter:
            batch.event.pop("channel_idx")

        # update the dynamic batch size so that the total no. of samples across
        # all rays are approximately constant for every batch
        batch.mean_num_samples_per_ray = self.update_train_batch_size(
            batch.grad, batch.diff, batch_index
        )

        # compute the (parameter-normalized) loss terms
        batch.mean_loss = self.loss.compute(
            batch.event, batch.grad, batch.diff
        )
        for loss_name, mean_loss_value in batch.mean_loss.items():
            mean_contrast_reciprocal = (
                1 / self.contrast_threshold.mean_contrast_threshold
            )
            loss_param_weight = {
                None: 1.0,
                "mean_contrast_reciprocal": mean_contrast_reciprocal,
                "mean_contrast_reciprocal_sq": mean_contrast_reciprocal ** 2,
            }[self.hparams.loss.param_weight[loss_name]]
            batch.mean_loss[loss_name] = loss_param_weight * mean_loss_value

        # derive the final loss as the weighted sum of the loss terms
        batch.weighted_mean_loss = easydict.EasyDict({
            loss_name: mean_loss_value * self.hparams.loss.weight[loss_name]
            for loss_name, mean_loss_value in batch.mean_loss.items()
        })
        train_loss = sum(batch.weighted_mean_loss.values())

        # log some quantities for monitoring & debugging
        self.log("train/loss", train_loss, prog_bar=True)
        for loss_name, mean_loss_value in batch.mean_loss.items():
            self.log(f"train/{loss_name}", mean_loss_value)

        if not self.hparams.contrast_threshold.freeze:
            self.log("train/p2n_contrast_threshold_ratio",
                     self.contrast_threshold.p2n_contrast_threshold_ratio)
        if not self.hparams.refractory_period.freeze:
            self.log("train/refractory_period",
                     self.refractory_period.refractory_period)

        batch.mean_ray_occ_rate =  self.derive_mean_value(
            dict_keys_pairs=[ (batch.grad, ["mean_ray_occ_rate"]),
                              (batch.diff, ["start_mean_ray_occ_rate",
                                            "end_mean_ray_occ_rate"]) ]
        )
        batch.mean_valid_rate = self.derive_mean_value(
            dict_keys_pairs=[ (batch.grad, ["is_valid"]),
                              (batch.diff, ["is_start_valid",
                                            "is_end_valid"]) ],
            value_transform=tensor_ops.bool_mean
        )
        self.log("train/batch_size", batch.size)
        self.log("train/mean_num_samples_per_ray",
                 batch.mean_num_samples_per_ray)
        self.log("train/mean_ray_occ_rate", batch.mean_ray_occ_rate)
        self.log("train/mean_valid_rate", batch.mean_valid_rate)

        return train_loss

    def validation_step(self, batch, batch_index):
        stage = easydict.EasyDict({
            "intrinsics_inv": self.val_intrinsics_inv,
            "img_pixel_pos": self.val_img_pixel_pos
        })
        return self.evaluation_step(batch, batch_index, stage)

    def test_step(self, batch, batch_index):
        stage = easydict.EasyDict({
            "intrinsics_inv": self.test_intrinsics_inv,
            "img_pixel_pos": self.test_img_pixel_pos
        })
        return self.evaluation_step(batch, batch_index, stage)

    def evaluation_step(self, batch, batch_index, stage):
        batch = easydict.EasyDict(batch)
        batch.size = len(batch.img)
        assert batch.size == 1

        # remove the extra batch dimension of 1
        for key, value in batch.items():
            if key == "size":
                continue
            batch[key] = value.squeeze(dim=0)                                   # (...)

        # extract the target intensity image
        target_intensity_img = batch.img                                        # ([3,] img_height, img_width)
        img_height, img_width = target_intensity_img.shape[-2:]
        assert img_height == stage.img_pixel_pos.shape[0] \
               and img_width == stage.img_pixel_pos.shape[1]

        # expand the camera position & orientation wrt. world frame according
        # to the image size
        batch.T_wc_position = batch.T_wc_position.view(1, 1, 3)                 # (1, 1, 3)
        batch.T_wc_orientation = batch.T_wc_orientation.view(1, 1, 3, 3)        # (1, 1, 3, 3)
        batch.T_wc_position = batch.T_wc_position.expand(                       # (img_height, img_width, 3)
            img_height, img_width, -1
        )
        batch.T_wc_orientation = batch.T_wc_orientation.expand(                 # (img_height, img_width, 3, 3)
            img_height, img_width, -1, -1
        )

        # infer the predicted log intensity image
        pred_intensity_img, _, _, _, _ = (                                      # ([3,] img_height, img_width)
            self.render_pixels(stage.intrinsics_inv, stage.img_pixel_pos,
                               batch.T_wc_position, batch.T_wc_orientation)
        )

        return {
            "sample_id": batch.sample_id,
            "pred_intensity_img": pred_intensity_img,
            "target_intensity_img": target_intensity_img
        }

    def validation_epoch_end(self, outputs):
        stage = easydict.EasyDict({
            "name": "val",
            "min_normalized_pixel_value": self.val_min_normalized_pixel_value,
            "max_normalized_pixel_value": self.val_max_normalized_pixel_value
        })
        return self.evaluation_epoch_end(outputs, stage)

    def test_epoch_end(self, outputs):
        stage = easydict.EasyDict({
            "name": "test",
            "min_normalized_pixel_value": self.test_min_normalized_pixel_value,
            "max_normalized_pixel_value": self.test_max_normalized_pixel_value
        })
        return self.evaluation_epoch_end(outputs, stage)

    def evaluation_epoch_end(self, outputs, stage):
        # gather & concatenate / stack the outputs across all devices
        outputs = self.all_gather(outputs)

        if outputs[0]["sample_id"].dim() == 1:                                  # ie. (PosedImage.NORMALIZED_SAMPLE_ID_CHAR_LEN)
            merge_fn = torch.stack
        elif outputs[0]["sample_id"].dim() == 2:                                # ie. (1, PosedImage.NORMALIZED_SAMPLE_ID_CHAR_LEN)
            merge_fn = torch.cat

        sample_id = merge_fn([ output["sample_id"] for output in outputs ])     # (batch_size, PosedImage.NORMALIZED_SAMPLE_ID_CHAR_LEN)
        pred_intensity_img = merge_fn(                                          # (batch_size, [3,] img_height, img_width)
            [ output["pred_intensity_img"] for output in outputs ]
        )
        target_intensity_img = merge_fn(                                        # (batch_size, [3,] img_height, img_width)
            [ output["target_intensity_img"] for output in outputs ]
        )
        del outputs

        # convert the unicode code point tensors of sample IDs to strings
        sample_id = self.unicode_code_pt_tensor_to_str(sample_id)               # (batch_size) list of `str`

        # cache the log directory to prevent deadlock later while defining the
        # predicted intensity image folder path, when `DistributedDataParallel`
        # is used
        log_dir = self.trainer.log_dir

        # compute & log the metrics & intensity images, if this process has
        # global rank of 0
        if not self.trainer.is_global_zero:
            return

        # perform remaining computations on CPU to prevent out-of-GPU-memory
        # due to large evaluation dataset
        pred_intensity_img = pred_intensity_img.cpu()
        target_intensity_img = target_intensity_img.cpu()

        # infer the batch & image sizes
        batch_size = len(target_intensity_img)
        img_height, img_width = target_intensity_img.shape[-2:]

        # insert a channel dimension of 1 for grayscale images
        if not self.has_bayer_filter:
            pred_intensity_img = pred_intensity_img.unsqueeze(dim=1)            # (batch_size, 1, img_height, img_width)
            target_intensity_img = target_intensity_img.unsqueeze(dim=1)        # (batch_size, 1, img_height, img_width)

        # derive the predicted & target log intensity images
        pred_log_intensity_img = pred_intensity_img.log()                       # (batch_size, 1/3, img_height, img_width)
        target_log_intensity_img = target_intensity_img.log()                   # (batch_size, 1/3, img_height, img_width)
        del pred_intensity_img

        # align the affine-ambiguous predicted log intensities to the target
        # log intensities via an optimal affine transformation
        # (ie. optimize log_intensity_scale & log_intensity_offset s.t.
        #  log_intensity_scale * pred_log_intensity_img + log_intensity_offset
        #  ≈ target_log_intensity_img)
        pred_log_intensity_img = torch.nn.functional.pad(                       # (batch_size, 1/3, img_height, img_width, 2)
            pred_log_intensity_img.unsqueeze(dim=-1),                           # (batch_size, 1/3, img_height, img_width, 1)
            pad=(0, 1), mode="constant", value=1
        )
        target_log_intensity_img = target_log_intensity_img.unsqueeze(dim=-1)   # (batch_size, 1/3, img_height, img_width, 1)

        # flatten the non-channel dims of the log intensity images
        pred_log_intensity_img = pred_log_intensity_img.transpose(0, 1)         # (1/3, batch_size, img_height, img_width, 2)
        target_log_intensity_img = target_log_intensity_img.transpose(0, 1)     # (1/3, batch_size, img_height, img_width, 1)
        pred_log_intensity_img = pred_log_intensity_img.flatten(                # (1/3, batch_size * img_height * img_width, 2)
            start_dim=1, end_dim=3
        )
        target_log_intensity_img = target_log_intensity_img.flatten(            # (1/3, batch_size * img_height * img_width, 1)
            start_dim=1, end_dim=3
        )

        # compute the least squares affine alignment in float64 for higher
        # numerical accuracy
        pred_log_intensity_img = pred_log_intensity_img.to(torch.float64)
        log_intensity_scale_offset, _, _, _ = torch.linalg.lstsq(               # (1/3, 2, 1)
            pred_log_intensity_img, target_log_intensity_img.to(torch.float64)
        )
        pred_log_intensity_img = pred_log_intensity_img \
                                 @ log_intensity_scale_offset                   # (1/3, batch_size * img_height * img_width, 1)
        pred_log_intensity_img = pred_log_intensity_img.to(
            target_log_intensity_img.dtype
        )

        # swap the batch & channel dimensions of the log intensity images
        pred_log_intensity_img = pred_log_intensity_img.view(                   # (1/3, batch_size, img_height, img_width)
            -1, batch_size, img_height, img_width
        )
        pred_log_intensity_img = pred_log_intensity_img.transpose(0, 1)         # (batch_size, 1/3, img_height, img_width)
        pred_intensity_img = pred_log_intensity_img.exp()                       # (batch_size, 1/3, img_height, img_width)
        del pred_log_intensity_img, target_log_intensity_img

        # compute the metric terms on the intensity images iteratively to
        # prevent out-of-memory, due to large evaluation dataset
        pred_intensity_img = pred_intensity_img.to(self.device)
        target_intensity_img = target_intensity_img.to(self.device)
        metric = self.metric.init_batch_metric()
        for pred_intensity_img_sample, target_intensity_img_sample in zip(      # (1/3, img_height, img_width), (1/3, img_height, img_width)
            pred_intensity_img, target_intensity_img
        ):
            sample_metric = self.metric.compute(
                pred_intensity_img_sample, target_intensity_img_sample,
                min_target_val=stage.min_normalized_pixel_value,
                max_target_val=stage.max_normalized_pixel_value
            )
            for metric_name, metric_value in sample_metric.items():
                metric[metric_name].append(metric_value)
        for metric_name, metric_value in metric.items():
            metric[metric_name] = sum(metric_value) / batch_size

        # log some quantities for monitoring & debugging
        self.log(
            f"{stage.name}/epoch", self.current_epoch,
            prog_bar=True, logger=False, rank_zero_only=True
        )
        for metric_name, metric_value in metric.items():
            self.log(
                f"{stage.name}/{metric_name}", metric_value,
                prog_bar=True, rank_zero_only=True
            )

        # log the predicted & target intensity images, if logging is required
        if self.logger is None:
            return
        self.logger.experiment.add_image(
            f"{stage.name}/pred_intensity_img",
            tensor_ops.normalize_range(
                pred_intensity_img[0, ...],                                     # (1/3, img_height, img_width)
                min=stage.min_normalized_pixel_value,
                max=stage.max_normalized_pixel_value
            ).clamp(min=0, max=1),
            global_step=self.global_step,
        )

        # log the target intensity image, if this is the first epoch
        if self.current_epoch == 0:
            self.logger.experiment.add_image(
                f"{stage.name}/target_intensity_img",
                tensor_ops.normalize_range(
                    target_intensity_img[0, ...],                                   # (1/3, img_height, img_width)
                    min=stage.min_normalized_pixel_value,
                    max=stage.max_normalized_pixel_value
                ),
                global_step=self.global_step
            )
        del target_intensity_img

        # save the predicted intensity images to disk, if required
        if not self.eval_save_pred_intensity_img:
            return

        # normalize, clip & quantize the predicted intensity images to
        # [0, 2 ** self.PREDICTED_INTENSITY_IMAGE_BIT_DEPTH - 1]
        max_pixel_value = 2 ** self.PREDICTION_BIT_DEPTH - 1
        pred_intensity_img = pred_intensity_img.cpu()
        pred_intensity_img = max_pixel_value * tensor_ops.normalize_range(
            pred_intensity_img,
            min=stage.min_normalized_pixel_value,
            max=stage.max_normalized_pixel_value
        ).clamp(min=0, max=1)
        pred_intensity_img = pred_intensity_img.round()

        if self.PREDICTION_BIT_DEPTH == 8:
            pred_intensity_img_dtype = np.uint8
        elif self.PREDICTION_BIT_DEPTH == 16:
            pred_intensity_img_dtype = np.uint16
        else:
            raise NotImplementedError
        pred_intensity_img = pred_intensity_img.numpy().astype(
            pred_intensity_img_dtype
        )

        # convert the predicted intensity images to OpenCV format
        pred_intensity_img = pred_intensity_img.transpose(0, 2, 3, 1)           # (batch_size, img_height, img_width, 1/3) in Gray/RGB format
        if self.has_bayer_filter:
            pred_intensity_img = np.stack([                                     # (batch_size, img_height, img_width, 3) in BGR format
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR)                            # (batch_size, img_height, img_width, 3) in BGR format
                for img in pred_intensity_img                                   # (img_height, img_width, 3) in RGB format
            ], axis=0)

        # create the folder to save the predicted intensity images
        pred_intensity_img_folder = os.path.join(
            log_dir, self.PREDICTIONS_FOLDER_NAME
        )
        os.makedirs(pred_intensity_img_folder, exist_ok=True)

        # iteratively save the predicted intensity images
        for id, img in zip(sample_id, pred_intensity_img):
            pred_intensity_img_path = os.path.join(
                pred_intensity_img_folder,
                id + self.PREDICTION_FILE_EXTENSION
            )
            cv2.imwrite(pred_intensity_img_path, img)

    def configure_optimizers(self):
        # partition parameters
        contrast_threshold_params = list(self.contrast_threshold.parameters())
        refractory_period_params = list(self.refractory_period.parameters())
        nerf_mlp_params = [
            param for param_name, param in self.named_parameters()
            if param_name.startswith("nerf.radiance_field.mlp")
        ]
        other_params = list(
            set(self.parameters()) - set(refractory_period_params)
            - set(contrast_threshold_params) - set(nerf_mlp_params)
        )
        assert len(list(self.parameters())) == sum(
            map(len, [ contrast_threshold_params, refractory_period_params,
                       nerf_mlp_params, other_params ])
        )

        # instantiate optimizer
        if self.hparams.optimizer.algo == "adam":
            optimizer = torch.optim.Adam(
                [ { "params": contrast_threshold_params,
                    "lr": self.hparams.optimizer.lr.contrast_threshold },
                  { "params": refractory_period_params,
                    "lr": self.refractory_period.max_refractory_period \
                          * self.hparams.optimizer.relative_lr \
                                                  .refractory_period },
                  { "params": nerf_mlp_params,
                    "weight_decay": self.hparams.loss.weight \
                                                     .nerf_mlp_weight_decay },
                  { "params": other_params } ],
                lr=self.hparams.optimizer.lr.default
            )
        else:
            raise NotImplementedError

        # instantiate learning rate scheduler
        if self.hparams.lr_scheduler.algo == "multi_step_lr":
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.hparams.lr_scheduler.multi_step_lr.milestones,
                gamma=self.hparams.lr_scheduler.multi_step_lr.gamma
            )
        else:
            raise NotImplementedError
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": self.hparams.lr_scheduler.interval
            }
        }

    def on_train_start(self):
        if self.logger is None:
            return

        # define the metrics to track for hyperparameter tuning
        self.logger.log_hyperparams(
            self.hparams,
            {
                "val/l1": float("inf"),
                "val/psnr": float("-inf"),
                "val/ssim": -1,
                "val/lpips": float("inf"),
            }
        )

    def render_pixels(
        self,
        intrinsics_inverse,                                                     # ([[M,] N,] 3, 3)
        pixel_position,                                                         # ([M,] N, 2)
        T_wc_position,                                                          # ([M,] N, 3)
        T_wc_orientation                                                        # ([M,] N, 3, 3)
    ):
        ray_origin, ray_direction = self.nerf.pixel_params_to_ray(              # ([M,] N, 3), ([M,] N, 3)
            intrinsics_inverse, pixel_position,
            T_wc_position, T_wc_orientation
        )
        intensity, opacity, depth, mean_num_samples_per_ray = self.nerf(        # ([M,] N [, 3]), ([M,] N), ([M,] N), float
            ray_origin, ray_direction
        )               
                    

        if intensity.dim() > opacity.dim():
            intensity = intensity.permute(-1, *range(opacity.dim()))            # (3, [M,] N)
        intensity = intensity + self.hparams.min_modeled_intensity              # ([3, M,] N)
        if self.render_bkgd is None:
            is_valid = opacity > 0                                              # ([M,] N)
        else:   # ie. elif self.render_bkgd == "parameter":
            is_valid = torch.ones_like(opacity, dtype=torch.bool)               # ([M,] N)

        """
        NOTE:
            The inferred depth is actually the expected ray termination
            distance, which is measured along the ray, not the principal axis /
            camera space z-axis. Its relationship with the actual depth is:
            ray termination distance * cos(angle between ray & principal axis)
            = depth.
        """
        cam_principal_axis_direction = T_wc_orientation[..., 2]                 # ([M,] N, 3)
        depth = depth * torch.sum(                                              # ([M,] N)
            ray_direction * cam_principal_axis_direction, dim=-1                # ([M,] N, 3)
        )
        return intensity, opacity, depth, mean_num_samples_per_ray, is_valid

    def bayering(self, intensity, channel_idx):
        channel_idx = channel_idx.view(1, -1)                                   # (1, batch.size)
        bayered_intensity = intensity.gather(dim=0, index=channel_idx)          # (1, batch.size)
        return bayered_intensity.squeeze(dim=0)                                 # (batch.size)

    @staticmethod
    def derive_mean_value(
        dict_keys_pairs,
        value_transform=torch.nn.Identity()
    ):
        # filter non-existent dictionaries & invalid keys
        dict_keys_pairs = [ (dict, [key for key in keys if key in dict.keys()])
                            for dict, keys in dict_keys_pairs
                            if not dict is None ]
        values = [ value_transform(dict[key])
                   for dict, keys in dict_keys_pairs
                   for key in keys ]    
        mean_value = sum(values) / len(values)
        return mean_value

    def update_train_batch_size(self, batch_grad, batch_diff, batch_index):
        # consolidate the mean no. of samples per ray
        mean_num_samples_per_ray = self.derive_mean_value(
            dict_keys_pairs=[ (batch_grad, ["mean_num_samples_per_ray"]),
                              (batch_diff, ["start_mean_num_samples_per_ray",
                                            "end_mean_num_samples_per_ray"]) ]
        )
        
        # reduce the mean no. of samples per ray across all devices
        mean_num_samples_per_ray = torch.tensor(mean_num_samples_per_ray)
        mean_num_samples_per_ray = torch.mean(
            self.all_gather(mean_num_samples_per_ray)
        )

        # derive & update the new training batch size (for a single device),
        # only if no gradient accumulation is done or this is the 2nd to last
        # batch of the gradient accumulation batches
        """
        NOTE:
            The new training batch size will only take effect two batches from
            the current (ie. in batch `batch_index + 2`) due to the 1 batch
            pre-fetching in PyTorch Lightning.

        Reference:
            https://pytorch-lightning.readthedocs.io/en/1.8.3/guides/data.html#iterable-datasets
        """
        if (
            (self.trainer.accumulate_grad_batches > 1)
            and ((batch_index % self.trainer.accumulate_grad_batches)
                 != (self.trainer.accumulate_grad_batches - 2))
        ):
            return mean_num_samples_per_ray

        new_train_batch_size = int(
            self.train_ray_sample_batch_size / mean_num_samples_per_ray
        )

        self.trainer.datamodule.train_dataset.batch_size = new_train_batch_size
        for normalized_sampler in (
            self.trainer.datamodule.train_normalized_sampler.datasets
        ):
            normalized_sampler.size = new_train_batch_size
        
        return mean_num_samples_per_ray

    @staticmethod
    def unicode_code_pt_tensor_to_str(batch_unicode_code_pt_tensor):            # (batch.size, S)
        # for each sample in the batch, convert all Unicode code point integers
        # to characters, then concatenate them to form a string & finally
        # remove any trailing spaces used for padding 
        batch_str = [
            "".join(map(chr, sample_unicode_code_pt_tensor)).rstrip()
            for sample_unicode_code_pt_tensor in batch_unicode_code_pt_tensor
        ]
        return batch_str
