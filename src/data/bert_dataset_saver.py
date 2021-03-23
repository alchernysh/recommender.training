# -*- coding: utf-8 -*-
import torch
import numpy as np
import pyxis.torch as pxt
from transformers import BertConfig, BertModel

from src.data.abstract_dataset_saver import DatasetSaver


class BertDatasetSaver(DatasetSaver):
    def __init__(self, target_path, device):
        super(BertDatasetSaver, self).__init__(target_path)
        self._device = device
        config = BertConfig.from_json_file(
            'resources/sentence_ru_cased_L-12_H-768_A-12_pt/bert_config.json'
        )
        self._bert_model = BertModel.from_pretrained(
            'resources/sentence_ru_cased_L-12_H-768_A-12_pt',
            config=config
        )
        self._bert_model.to(device)
        self._bert_model.eval()

    def _get_dataset(self, dataset_name):
        dataset = pxt.TorchDataset(f'data/processed/tokenized/{dataset_name}')
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, drop_last=True)
        return data_loader

    def _get_bert_features(self, tokens_list):
        bert_features, _ = self._bert_model(
            tokens_list[0],
            attention_mask=tokens_list[1],
            token_type_ids=tokens_list[2]
        )
        return bert_features

    def _preprocess_data(self, data):
        preprocessed_data = dict()
        for i in range(2):
            tokens_list = [
                data[f'input_ids_{i}'].view(-1, 128).to(dtype=torch.int64).to(self._device),
                data[f'attention_mask_{i}'].view(-1, 128).to(dtype=torch.int64).to(self._device),
                data[f'token_type_ids_{i}'].view(-1, 128).to(dtype=torch.int64).to(self._device),
            ]
            bert_features = self._get_bert_features(tokens_list)
            preprocessed_data.update(
                {
                    f'bert_features_{i}': bert_features.cpu().detach().numpy()
                }
            )
        preprocessed_data.update({'similarity': np.array(data['similarity'].view(-1, 1).detach().numpy())})
        return preprocessed_data
