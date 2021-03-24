# -*- coding: utf-8 -*-
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter


class TensorboardCallback(pl.callbacks.base.Callback):
    def __init__(self, train_id):
        super().__init__()
        self._writer = SummaryWriter(f'logs/{train_id}/tensorboard/')

    def on_batch_end(self, trainer, pl_module):
        global_step = trainer.global_step
        lr = trainer.lr_schedulers[0]['scheduler']._last_lr[0]
        self._writer.add_scalar('lr', lr, global_step=global_step)
        metrics = trainer.callback_metrics
        loss_train = metrics['train_loss']
        metric_train = metrics['train_metric']
        self._writer.add_scalars('loss', {'train': loss_train}, global_step=global_step)
        self._writer.add_scalars('metric', {'train': metric_train}, global_step=global_step)

    def on_validation_end(self, trainer, pl_module):
        global_step = trainer.global_step
        metrics = trainer.callback_metrics
        if metrics != {}:
            loss_val = metrics['val_loss']
            metric_val = metrics['val_metric']
            self._writer.add_scalars('loss', {'val': loss_val}, global_step=global_step)
            self._writer.add_scalars('metric', {'val': metric_val}, global_step=global_step)
