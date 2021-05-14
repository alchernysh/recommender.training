# -*- coding: utf-8 -*-
import torch
import numpy as np
from transformers import BertConfig, BertModel

from src.data.serializers.abstract import AbstractSerializer
from src.data.dataloader import get_dataloader


class BertFeaturedSerializer(AbstractSerializer):
    def __init__(self, origin_path, target_path, device, batch_size):
        super().__init__(target_path)
        self._device = device
        self._batch_size = batch_size
        self._origin_path = origin_path
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
        data_loader = get_dataloader(self._origin_path, dataset_name, self._batch_size)
        return data_loader

    def _prepare_data(self, data):
        return data.view(-1, 128).to(dtype=torch.int64).to(self._device)
    
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
                self._prepare_data(data[f'input_ids_{i}']),
                self._prepare_data(data[f'attention_mask_{i}']),
                self._prepare_data(data[f'token_type_ids_{i}']),
            ]
            bert_features = self._get_bert_features(tokens_list)
            preprocessed_data.update(
                {
                    f'bert_features_{i}': bert_features.cpu().detach().numpy()
                }
            )
        preprocessed_data.update(
            {
                'similarity': np.array(
                    data['similarity'].view(-1, 1).detach().numpy()
                )
            }
        )
        return preprocessed_data
