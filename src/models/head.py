# -*- coding: utf-8 -*-
import torch.nn as nn

from src.models.embedding import EmbeddingModel


class HeadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._embedding_net = EmbeddingModel()
        self._cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, x, y):
        x = self._embedding_net(x)
        y = self._embedding_net(y)
        res = self._cos_sim(x, y)
        return res
