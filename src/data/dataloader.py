# -*- coding: utf-8 -*-
from pathlib import Path

from torch.utils.data import DataLoader
from pyxis.torch import TorchDataset


def get_dataloader(dataset_path, dataset_name, batch_size):
    dataset_path_full = Path(dataset_path).joinpath(dataset_name)
    dataset = TorchDataset(str(dataset_path_full))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)
