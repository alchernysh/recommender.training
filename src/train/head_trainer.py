# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from src.model import HeadModel
from src.data.dataloader import get_dataloader


class HeadModelTraining(pl.LightningModule):
    def __init__(self, dataset_path, batch_size):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.model = HeadModel()
        self.loss = nn.L1Loss()
        self.metric = nn.MSELoss()

    def forward(self, x, y):
        res = self.model(x, y)
        return res

    def training_step(self, batch, batch_idx):
        x1 = batch['bert_features_0']
        x2 = batch['bert_features_1']
        y = batch['similarity']
        y_pred = self(x1, x2)
        loss = self.loss(y_pred, y)
        metric = self.metric(y_pred, y)
        tensorboard_logs = {
            'train_metric': metric,
            'train_loss': loss,
        }
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_metric', metric, prog_bar=False)
        return {"loss": loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x1 = batch['bert_features_0']
        x2 = batch['bert_features_1']
        y = batch['similarity']
        y_pred = self(x1, x2)
        loss = self.loss(y_pred, y)
        metric = self.metric(y_pred, y)
        tensorboard_logs = {
            'val_metric': metric,
            'val_loss': loss,
        }
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_metric', metric, prog_bar=False)
        return {"loss": loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9)
        lr_scheduler = {
            'scheduler': optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=0.001,
                max_lr=0.1,
                step_size_up=50,
                mode="triangular2"
            ),
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 100,
        }

        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        data_loader = get_dataloader(self.dataset_path, 'train', self.batch_size)
        return data_loader

    def val_dataloader(self):
        data_loader = get_dataloader(self.dataset_path, 'test', self.batch_size)
        return data_loader

    def test_dataloader(self):
        data_loader = get_dataloader(self.dataset_path, 'val', self.batch_size)
        return data_loader
