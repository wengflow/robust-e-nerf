import sys
import os
import pathlib
import shutil
import subprocess
import argparse
import yaml
import easydict
import torch
import pytorch_lightning as pl

# insert the project / script parent directory into the module search path
PROJECT_DIR = os.path.join(sys.path[0], '..')
sys.path.insert(1, PROJECT_DIR)
import robust_e_nerf as ren

STAGES = [ "train", "val", "test" ]
METRICS_FILENAME = "metrics.yaml"


def main(args):
    # load the config from the config file
    with open(args.config) as f:
        config = easydict.EasyDict(yaml.full_load(f))

    # obtain the git HEAD hash
    config.git_head_hash = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=PROJECT_DIR
    ).decode('ascii').strip()

    # seed all pseudo-random generators
    config.seed = pl.seed_everything(config.seed, workers=True)

    # set the float32 matrix multiplication precision on CUDA devices
    torch.set_float32_matmul_precision(config.float32_matmul_precision)
    
    # instantiate the data module & model
    datamodule = ren.data.datamodule.DataModule(
        config.seed,
        config.eval_target,
        config.trainer.num_nodes,
        config.trainer.gpus,
        **config.data
    )
    model = ren.models.robust_e_nerf.RobustENeRF(
        config.git_head_hash,
        config.eval_target,
        config.trainer.num_nodes,
        config.trainer.gpus,
        config.model.min_modeled_intensity,
        config.model.eval_save_pred_intensity_img,
        config.model.checkpoint_filepath,
        config.model.contrast_threshold,
        config.model.refractory_period,
        config.model.nerf,
        config.loss,
        config.metric,
        config.optimizer,
        config.lr_scheduler,
        config.data.dataset_directory,
        config.data.alpha_over_white_bg,
        config.data.train_eff_ray_sample_batch_size
    )

    # instantiate the trainer & its components
    if getattr(config.trainer, "checkpoint_callback", True):
        checkpoint_callback = pl.callbacks.ModelCheckpoint(**config.checkpoint)
        callbacks = [ checkpoint_callback ]
    else:
        callbacks = None

    if getattr(config.trainer, "logger", True):
        logger = pl.loggers.tensorboard.TensorBoardLogger(
            default_hp_metric=False, **config.logger
        )   
    else:
        logger = False
    if hasattr(config.trainer, "logger"):
        config.trainer.pop("logger")

    plugins = {
        None: None,
        "ddp_cpu": pl.plugins.DDPPlugin(find_unused_parameters=False),
        "ddp": pl.plugins.DDPPlugin(find_unused_parameters=False),
        "ddp_spawn": pl.plugins.DDPSpawnPlugin(find_unused_parameters=False)
    }[config.trainer.accelerator]

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        plugins=plugins,
        replace_sampler_ddp=True,
        sync_batchnorm=True,
        terminate_on_nan=True,
        multiple_trainloader_mode="min_size",
        **config.trainer
    )

    # save the config to the log directory, if the current process has a global
    # rank of zero, logging is done & not resuming training from a checkpoint
    if (
        trainer.is_global_zero
        and logger is not False
        and getattr(config.trainer, "resume_from_checkpoint", None) is None
    ):
        pathlib.Path(trainer.logger.log_dir).mkdir(parents=True)
        shutil.copy2(args.config, trainer.logger.log_dir)

    # train, validate or test the model
    if args.stage == "train":
        trainer.fit(model, datamodule)
    elif args.stage == "val":
        metrics = trainer.validate(model, datamodule)
    elif args.stage == "test":
        metrics = trainer.test(model, datamodule)

    # save the validation / test metrics to the log dir, if the current process
    # has a global rank of zero and logging is done
    if (
        args.stage != "train"
        and trainer.is_global_zero
        and (logger is not False)
    ):
        metrics_filepath = os.path.join(
            trainer.logger.log_dir, METRICS_FILENAME
        )
        with open(metrics_filepath, 'w') as f:
            yaml.dump(metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Training, validation & testing script of Robust e-NeRF")
    )
    parser.add_argument(
        "stage", type=str, choices=STAGES,
        help="Train, validation or test mode."
    )
    parser.add_argument(
        "config", type=str, help="Path to a configuration file in yaml format."
    )
    args = parser.parse_args()

    main(args)
