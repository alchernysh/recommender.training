# -*- coding: utf-8 -*-
# from src.train import lightning_train
import argparse

import torch
import random
import numpy as np

from src.utils.config import Config
from src.runners import serialize_bert_featured, serialize_tokenized, train, save, generate_embeddings


np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


commands = {
    'serialize_bert_featured': serialize_bert_featured,
    'serialize_tokenized': serialize_tokenized,
    'train': train,
    'generate_embeddings': generate_embeddings,
    'save': save,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', type=str)

    args = parser.parse_args()
    config = Config()
    if not Config.validate_configs():
        exit(1)
    commands[args.command](config)
