# -*- coding: utf-8 -*-
import torch.nn as nn
from transformers import BertConfig, BertModel

from src.models.embedding import EmbeddingModel


class BertEmbeddingModel(EmbeddingModel):
    def __init__(self):
        super().__init__()
        config = BertConfig.from_json_file(
            'resources/sentence_ru_cased_L-12_H-768_A-12_pt/bert_config.json'
        )
        self._bert_model = BertModel.from_pretrained(
            'resources/sentence_ru_cased_L-12_H-768_A-12_pt',
            config=config
        )

    def forward(self, x):
        input_ids, attention_masks, token_type_ids = x
        x, _ = self._bert_model(
            input_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )
        x = super().forward(x)
        return x
