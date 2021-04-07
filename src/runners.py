# -*- coding: utf-8 -*-
import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data.serializers.tokenized import TokenizedSerializer
from src.data.serializers.bert_featured import BertFeaturedSerializer
from src.train.head_trainer import HeadModelTraining
from src.train.callbacks.tensorboard import TensorboardCallback


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


def get_train_id(config):
    if config.train.mode == 'start':
        train_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    elif config.train.mode == 'continue':
        train_id = config.train.id
    else:
        ValueError(config.train.mode)
    return train_id


def train(config):

    train_id = get_train_id(config)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'logs/{train_id}/savings/',
        filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}',
        verbose=True,
        mode='min',
        save_top_k=1,
        save_last=True,
        monitor='val_loss',
    )
    tensorboard_callback = TensorboardCallback(train_id)
    training_model = HeadModelTraining(config)
    trainer_params = {
        'gpus': 0 if config.device.name == 'cpu' else config.device.number,
        'max_epochs': config.train.epochs,
        'progress_bar_refresh_rate': 20,
        'callbacks': [tensorboard_callback, checkpoint_callback],
        'logger': False,

    }

    if config.train.mode == 'continue':
        trainer_params.update(
            {
                'resume_from_checkpoint': f'logs/{train_id}/savings/last.ckpt'
            }
        )
    
    trainer = pl.Trainer(**trainer_params)
    trainer.fit(training_model)
