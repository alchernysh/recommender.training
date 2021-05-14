# -*- coding: utf-8 -*-
import json
from pathlib import Path

import numpy as np
from transformers import BertTokenizer, BertConfig

from src.data.serializers.abstract import AbstractSerializer


class TokenizedSerializer(AbstractSerializer):
    def __init__(self, target_path):
        super().__init__(target_path)
        config = BertConfig.from_json_file(
            'resources/sentence_ru_cased_L-12_H-768_A-12_pt/bert_config.json'
        )
        self._tokenizer = BertTokenizer.from_pretrained(
            'resources/sentence_ru_cased_L-12_H-768_A-12_pt',
            from_pt=True,
            config=config,
            do_lower_case=True,
        )

    def _get_dataset(self, dataset_name):
        with Path(f'data/raw/{dataset_name}.json').open() as f:
            dataset = json.load(f)
        return dataset

    def _tokenize_sentence(self, encoded_sentence):
        decoded_sentence = [encoded_sentence.decode('utf-8')]
        tokens_dict = self._tokenizer.batch_encode_plus(
            decoded_sentence,
            add_special_tokens=True,
            max_length=128,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="np",
        )
        return tokens_dict

    def _preprocess_data(self, data):
        preprocessed_data = dict()
        for i in range(2):
            with open(f'data/raw/texts/{data["sentence_ids"][i]}.txt') as f:
                sentence = str.encode(f.read())
            token_dict = self._tokenize_sentence(sentence)
            for key, val in token_dict.items():
                preprocessed_data.update(
                    {
                        f'{key}_{i}': val,
                    }
                )
        preprocessed_data.update({'similarity': np.array([data['similarity']])})
        return preprocessed_data
