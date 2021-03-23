# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_dim1,
                 hidden_dim2,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout,
                 pad_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_index)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim1,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            batch_first=True)
        self.fc1 = nn.Linear(hidden_dim1 * 2, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        packed_embedded = pack_padded_sequence(embedded, text_lengths, batch_first=True) 

        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        rel = self.relu(cat)
        dense1 = self.fc1(rel)
        drop = self.dropout(dense1)
        preds = self.fc2(drop)
        return preds


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
