# -*- coding: utf-8 -*-
import json
from pathlib import Path
import datetime
from collections import OrderedDict

import onnx
import ml2rt
import torch
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import BertTokenizer, BertConfig

from src.data.serializers.tokenized import TokenizedSerializer
from src.data.serializers.bert_featured import BertFeaturedSerializer
from src.train.head_trainer import HeadModelTraining
from src.train.callbacks.tensorboard import TensorboardCallback
from src.models.bert_embedding import BertEmbeddingModel


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
        filename='{step}-{val_loss:.4f}',
        verbose=True,
        mode='min',
        save_top_k=1,
        save_last=True,
        monitor='val_loss',
        save_weights_only=False,
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

def generate_embeddings(config):
    processed_data = list()

    bert_config = BertConfig.from_json_file(
        'resources/sentence_ru_cased_L-12_H-768_A-12_pt/bert_config.json'
    )
    tokenizer = BertTokenizer.from_pretrained(
        'resources/sentence_ru_cased_L-12_H-768_A-12_pt',
        from_pt=True,
        config=bert_config,
        do_lower_case=True,
    )

    model = BertEmbeddingModel()
    state_dict = torch.load(config.save.init_weights)['state_dict']
    state_dict = OrderedDict((k.replace('model._embedding_net.', ''), v) for k, v in state_dict.items())
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    with Path('data/raw/desc_filtered.json').open() as f:
        data = json.load(f)

    for item in tqdm(data[:10]):
        with open(f'data/raw/texts/{item["id"]}.txt') as f:
            text = f.read()

        tokens_dict = tokenizer.batch_encode_plus(
            [text],
            add_special_tokens=True,
            max_length=128,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="pt",
        )

        token_list = [
            tokens_dict['input_ids'],
            tokens_dict['attention_mask'],
            tokens_dict['token_type_ids']
        ]
        embedding = model(token_list)
        processed_data.append(
            {
                'id': item['id'],
                'topics': item['topics'],
                'embedding': embedding.detach().cpu().numpy()[0],
            }
        )

    topics = set()
    for item in processed_data:
        topics.update(item['topics'])

    embeddings = dict()
    for topic in tqdm(topics):
        filtered_data = list(filter(lambda x: topic in x['topics'], processed_data))
        topics_embeddings = np.array(list(map(lambda x: x['embedding'], filtered_data)))
        embeddings.update(
            {
                topic: np.mean(topics_embeddings, axis=0).tolist()
            }
        )

    with Path(config.generate_embeddings.embeddings_json).open('w') as f:
        json.dump(embeddings, f, indent=4)
    print(f'embeddings saved in {config.generate_embeddings.embeddings_json}')

def save(config):
    model = BertEmbeddingModel()
    state_dict = torch.load(config.save.init_weights)['state_dict']
    state_dict = OrderedDict((k.replace('model._embedding_net.', ''), v) for k, v in state_dict.items())
    model.load_state_dict(state_dict, strict=False)

    example_inputs = torch.randint(0, 1, (3, 1, 128))
    example_outputs = torch.rand((1, 1042))
    torchscript_model = torch.jit.trace(model, example_inputs)
    torchscript_model.eval()

    dynamic_axes = {
        'input': {1: 'batch_size'},
        'output': {0: 'batch_size'}
    }

    torch.onnx.export(
        torchscript_model,
        example_inputs,
        example_outputs=example_outputs,
        f='resources/bert.onnx',
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        export_params=True,
        do_constant_folding=True,
        opset_version=11,
    )

    onnx_model = onnx.load('resources/bert.onnx')
    ml2rt.save_onnx(onnx_model, 'resources/bert.rai')

    print(f'model saved in resources/bert.rai')

