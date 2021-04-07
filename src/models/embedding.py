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
        self.fc1 = nn.Linear(39168, 1024)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(1024, 1024)

    def forward(self, x):
        x = torch.cat(
            (self.avg_pool(x), self.max_pool(x)),
            dim=1
        )
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        return x
