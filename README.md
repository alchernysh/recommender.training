# Recommender training

This repo is part of recommender project. It contains code for training the model. Code for training written in `torch` more precise in [Lightning](https://www.pytorchlightning.ai/) framework. Bert is used as feature extractor. Before training features extract in Lightning Memory-Mapped Database (LMDB) via [ml-pyxis](https://github.com/vicolab/ml-pyxis).

## Requirements

- Ubuntu >= 18.04
- Python >= 3.8

## Installation

If you are using *pip* run command below to install python dependencies.

```shell
pip3 install -r requirements.txt
```

If you are using *Pipenv* you can install dependencies by command:

```shell
 pipenv install
```

## Usage

Firstly you need download [archive](https://drive.google.com/file/d/19OHlcFURzsBiwnFRHfqKTO2Lqpx_d3RF/view?usp=sharing) with dataset and unpack it to folder `data` (for default settings). Besides you need download [Bert weights](http://files.deeppavlov.ai/deeppavlov_data/bert/sentence_ru_cased_L-12_H-768_A-12_pt.tar.gz) for russian language and place it to folder `resources`.

Parameters for tasks are stored in `config.yml`. So before running you need create local config by coping `config.yml` to `config-default.yml` and change default parameters to actual.

Entrypoint is file *run.py*. For running programm in sertain mode you shoud run this in root directory.

```shell
python run.py <task>
```

where `task` represents current task.

For example if you want to train model you need run command:

```shell
python run.py train
```

### Avalable tasks

#### 1. *serialize_tokenized*

Task for serializing tokenized content of articles to LMDB. Script takes as input dataset generated in repo recommender.dataset.

#### 2. *serialize_bert_featured*

Task for serializing features generated by Bert to LMDB. Script takes as input result of `serialize_tokenized` task.

#### 3. *train*

Task for training model. Input is dataset, serialized during `serialize_bert_featured` task. Weights and metrics are saved in `logs` folder. You can visualize metrics with `tensorboard`.

#### 4. *generate_embeddings*

Task for generating embeddings for each topics in dataset. For this we calculate mean for embeddings of articles of same topics.

#### 5. *save*

Task for saving model in ONNX format for serving it in [redisai](https://oss.redislabs.com/redisai).

## Results

Mean absolute error (MAE) is used as loss function.

After 363175 steps MAE decreased to 0.06.

Training curve described on graph below.

![training curve](https://i.imgur.com/qXsHqsd.png)
