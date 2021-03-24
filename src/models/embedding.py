# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class EmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AvgPool1d(5)
        self.max_pool = nn.MaxPool1d(5)
        self.dropout = nn.Dropout(0.3)
        self.flatten = torch.nn.Flatten()
        self.fc = nn.Linear(39168, 1024)

    def forward(self, x):
        x = torch.cat(
            (self.avg_pool(x), self.max_pool(x)),
            dim=1
        )
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
