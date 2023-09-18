import numpy as np
import torch


class JoinDataset(torch.utils.data.IterableDataset):
    def __init__(self, datasets, dataset_keys):
        for dataset in datasets:
            assert isinstance(dataset, torch.utils.data.IterableDataset)
        for dataset_key in dataset_keys:
            assert isinstance(dataset_key, str)

        self.datasets = datasets
        self.dataset_keys = dataset_keys

    def __iter__(self):
        for data in zip(*self.datasets):
            yield dict(zip(self.dataset_keys, data))


class IterableMapDataset(torch.utils.data.IterableDataset):
    def __init__(self, map_dataset, batch_size, generator=None):
        self.map_dataset = map_dataset
        self.batch_size = batch_size
        self.generator = generator

    def __iter__(self):
        while True:
            sampled_indices = torch.randint(
                len(self.map_dataset), size=( self.batch_size, ),
                generator=self.generator
            )
            yield self.map_dataset[sampled_indices]


class TrimDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, start_index, end_index):
        """
        Trim `dataset` to samples with indices between [`start_index`, 
        `end_index`).
        """
        self.dataset = dataset
        self.start_index = start_index
        self.trimmed_len = end_index - start_index

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.trimmed_len
