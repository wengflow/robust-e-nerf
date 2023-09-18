import functools
import torch
import pytorch_lightning as pl
from . import datasets, samplers
from .. import utils


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        seed,
        eval_target,
        num_nodes,
        gpus,
        dataset_directory,
        train_dataset_ratio,
        val_dataset_ratio,
        test_dataset_ratio,
        train_dataset_perm_seed,
        eval_dataset_perm_seed,
        alpha_over_white_bg,
        train_init_eff_batch_size,
        train_eff_ray_sample_batch_size,
        val_eff_batch_size,
        test_eff_batch_size,
        num_workers_per_node
    ):
        super().__init__()
        for stage_dataset_ratio in [
            train_dataset_ratio, val_dataset_ratio, test_dataset_ratio
        ]:
            assert isinstance(stage_dataset_ratio, int) \
                   or (isinstance(stage_dataset_ratio, float)
                       and 0.0 < stage_dataset_ratio <= 1.0)

        """
        NOTE:
            Enforce main-process data loading to support dynamic batch size
            during training
        """
        assert num_workers_per_node == 0

        # save some non-hyperparameter configs as attributes
        self.eval_target = eval_target
        self.dataset_directory = dataset_directory
        self.train_dataset_ratio = train_dataset_ratio
        self.val_dataset_ratio = val_dataset_ratio
        self.test_dataset_ratio = test_dataset_ratio
        self.train_dataset_perm_seed = train_dataset_perm_seed
        self.eval_dataset_perm_seed = eval_dataset_perm_seed
        self.alpha_over_white_bg = alpha_over_white_bg
        self.val_eff_batch_size = val_eff_batch_size
        self.test_eff_batch_size = test_eff_batch_size

        # log, checkpoint & save hyperparameters to `hparams` attribute
        self.save_hyperparameters(
            "seed",
            "train_init_eff_batch_size",
            "train_eff_ray_sample_batch_size"
        )

        if gpus is None:    # ie. running on CPU
            self.train_batch_size = train_init_eff_batch_size
            self.val_batch_size = val_eff_batch_size
            self.test_batch_size = test_eff_batch_size
            self.num_workers = num_workers_per_node
        else:               # ie. running on GPU
            # derive gpu numbers
            num_gpus_per_node = len(gpus)
            num_gpus = num_nodes * num_gpus_per_node

            # derive the batch size per GPU across all nodes
            self.train_batch_size = train_init_eff_batch_size // num_gpus
            self.val_batch_size = val_eff_batch_size // num_gpus
            self.test_batch_size = test_eff_batch_size // num_gpus

            # derive the no. of workers for each subprocess associated to a GPU
            self.num_workers = num_workers_per_node // num_gpus_per_node

    def setup(self, stage):
        # instantiate datasets with transforms & uv samplers
        if stage in (None, "fit"):
            # prevent identical batches across device processes during
            # training, when using `DistributedDataParallel`
            process_seed = torch.initial_seed()
            if torch.distributed.is_initialized():
                process_seed += torch.distributed.get_rank()
            self.train_generator = torch.Generator()
            self.train_generator.manual_seed(process_seed)

            self.train_dataset = self._build_dataset("train")
            self.val_dataset = self._build_dataset("val")
            self.train_normalized_sampler = self._build_normalized_sampler()
        if stage in (None, "validate"):
            self.val_dataset = self._build_dataset("val")
        if stage in (None, "test"):
            self.test_dataset = self._build_dataset("test")

    def _build_dataset(self, stage):
        if stage == "train":
            dataset = datasets.Event(
                self.dataset_directory,
                self.train_dataset_perm_seed
            )
        else:   # elif stage in [ "val", "test" ]
            if set(self.eval_target) == set([ "event_view" ]):
                dataset = datasets.PosedImage(
                    self.dataset_directory, "train",
                    self.eval_dataset_perm_seed, self.alpha_over_white_bg
                )
            elif set(self.eval_target) == set([ "novel_view" ]):
                dataset = datasets.PosedImage(
                    self.dataset_directory, stage,
                    self.eval_dataset_perm_seed, self.alpha_over_white_bg
                )
            else:
                raise NotImplementedError

        # extract & return a subset of the dataset
        stage_dataset_ratio = {
            "train": self.train_dataset_ratio,
            "val": self.val_dataset_ratio,
            "test": self.test_dataset_ratio
        }[stage]
        if isinstance(stage_dataset_ratio, int):
            stage_eff_batch_size = {
                "train": self.hparams.train_init_eff_batch_size,
                "val": self.val_eff_batch_size,
                "test": self.test_eff_batch_size
            }[stage]
            dataset_subset_len = stage_dataset_ratio * stage_eff_batch_size
            assert dataset_subset_len <= len(dataset)
        else:   # elif isinstance(self.dataset_ratio, float)
            dataset_subset_len = int(stage_dataset_ratio * len(dataset))
        dataset_subset = utils.datasets.TrimDataset(
            dataset, start_index=0, end_index=dataset_subset_len
        )

        # convert the map-style event training dataset to iterable-style for
        # more time & memory efficient dataloading
        if stage == "train":
            dataset_subset = utils.datasets.IterableMapDataset(
                map_dataset=dataset_subset, batch_size=self.train_batch_size,
                generator=self.train_generator
            )

        return dataset_subset

    def _build_normalized_sampler(self):
        return utils.datasets.JoinDataset(
            datasets=( self._build_normalized_ts_diff_sampler(),
                       self._build_normalized_diff_start_ts_sampler(),
                       self._build_normalized_grad_ts_sampler() ),
            dataset_keys=( "ts_diff", "diff_start_ts", "grad_ts" )
        )

    def _build_normalized_ts_diff_sampler(self):
        """
        Instantiate a normalized timestamp difference (i.e. interval) sampler,
        which is an iterable-style dataset, that yields unlimited batches of
        normalized timestamp difference samples with values in the range of
        [0, 1] distributed according to the dirac-delta function, with
        shape `(self.train_batch_size)`
        """
        sampler = samplers.DiracDeltaSampler(
            center=1, size=self.train_batch_size,
            dtype=torch.float64
        )
        return sampler

    def _build_normalized_diff_start_ts_sampler(self):
        """
        Instantiate a normalized difference start timestamp sampler, which is
        an iterable-style dataset, that yields unlimited batches of normalized
        difference start timestamp samples with values in the range of [0, 1]
        distributed uniformly, with shape `(self.train_batch_size)`
        """
        return samplers.UniformSampler(
            low=0, high=1, size=self.train_batch_size, dtype=torch.float64,
            generator=self.train_generator
        )

    def _build_normalized_grad_ts_sampler(self):
        """
        Instantiate a normalized gradient timestamp sampler, which is an
        iterable-style dataset, that yields unlimited batches of normalized
        gradient timestamp samples with values in the range of [0, 1]
        distributed according to a truncated normal, with shape
        `(self.train_batch_size)`
        """
        sampler = samplers.TruncatedNormalSampler(
            low=0, high=1, size=self.train_batch_size,
            mean=0.5, std=0.25, dtype=torch.float64,                            # [0, 1] is within 2 * std
            generator=self.train_generator
        )
        return sampler

    def train_dataloader(self):
        """
        NOTE:
            `pl.Trainer(replace_sampler_ddp=True, accelerator=ddp/ddp_spawn, 
            ...)` implicitly replaces the `sampler` of `dataset_dataloader`
            with `DistributedSampler(shuffle=True, drop_last=False, ...)`
        """
        dataset_dataloader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(self.num_workers > 0)
        )
        normalized_sampler_dataloader = torch.utils.data.DataLoader(
            dataset=self.train_normalized_sampler,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(self.num_workers > 0)
        )

        # does not return an `easydict` due to its lack of support of
        # initialization with a list of key-value tuples, which causes issues
        # when using `DistributedDataParallel`
        return {
            "event": dataset_dataloader,
            "normalized": normalized_sampler_dataloader
        }


    def val_dataloader(self):
        """
        NOTE:
            1. `pl.Trainer(replace_sampler_ddp=True, accelerator=ddp/ddp_spawn, 
               ...)` implicitly replaces the `sampler` of `dataset_dataloader`
               with `DistributedSampler(shuffle=True, drop_last=False, ...)`.
            2. If `len(self.val_dataset)` is not divisible by `num_replicas`,
               validation is not entirely accurate, irrespective of
               `drop_last`.
        """
        dataset_dataloader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,            
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(self.num_workers > 0)
        )
        return dataset_dataloader

    def test_dataloader(self):
        """
        NOTE:
            1. `pl.Trainer(replace_sampler_ddp=True, accelerator=ddp/ddp_spawn, 
               ...)` implicitly replaces the `sampler` of `dataset_dataloader`
               with `DistributedSampler(shuffle=True, drop_last=False, ...)`.
            2. If `len(self.test_dataset)` is not divisible by `num_replicas`,
               testing is not entirely accurate, irrespective of `drop_last`.
        """
        dataset_dataloader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(self.num_workers > 0)
        )
        return dataset_dataloader
