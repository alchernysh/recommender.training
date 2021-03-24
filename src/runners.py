# -*- coding: utf-8 -*-
import torch

from src.data.serializers.tokenized import TokenizedSerializer
from src.data.serializers.bert_featured import BertFeaturedSerializer


def serialize_tokenized(config):
    target_path = config.dataset.path.tokenized

    tokenized_serializer = TokenizedSerializer(target_path=target_path)
    tokenized_serializer.run()

def serialize_bert_featured(config):
    origin_path = config.dataset.path.tokenized
    target_path = config.dataset.path.bert_featured
    device = torch.device(f'{config.device.name}:{config.device.number}')
    batch_size = config.dataset.serializer.batch_size

    bert_featured_serializer = BertFeaturedSerializer(
        origin_path=origin_path,
        target_path=target_path,
        device=device,
        batch_size=batch_size
    )
    bert_featured_serializer.run()
