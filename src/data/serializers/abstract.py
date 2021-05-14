# -*- coding: utf-8 -*-
from pathlib import Path
from abc import ABC, abstractmethod

import pyxis as px
from tqdm import tqdm


class AbstractSerializer(ABC):
    def __init__(self, target_path):
        self._target_path = target_path

    def _get_target_dataset_path(self, dataset_name):
        target_dataset_path = Path(self._target_path).joinpath(f'{dataset_name}')
        target_dataset_path.parent.mkdir(exist_ok=True, parents=True)
        return str(target_dataset_path)

    @abstractmethod
    def _get_dataset(self, dataset_name):
        pass

    @abstractmethod
    def _preprocess_data(self, data):
        pass

    def run(self):
        dataset_names = ['train', 'test', 'val']
        for dataset_name in dataset_names:
            print(f'saving dataset {dataset_name}')
            target_dataset_path = self._get_target_dataset_path(dataset_name)
            with px.Writer(dirpath=target_dataset_path, map_size_limit=10000000, ram_gb_limit=1024) as db:
                dataset = self._get_dataset(dataset_name)
                for data in tqdm(dataset):
                    preprocessed_data = self._preprocess_data(data)
                    db.put_samples(preprocessed_data)
