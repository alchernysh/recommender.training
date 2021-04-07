# -*- coding: utf-8 -*-
from functools import partial

import torch.optim as optim


def LambdaLR(optimizer, multiplier):

    def lr_lambda(epoch):
        return multiplier ** epoch

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def MultiplicativeLR(optimizer, multiplier):

    def lr_lambda(epoch):
        return multiplier ** epoch

    return optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_lambda)


def get_lr_scheduler(config):
    schedulers_map = {
        'cyclic_lr': optim.lr_scheduler.CyclicLR,
        'lambda_lr': LambdaLR,
        'multiplicative_lr': MultiplicativeLR,
        'step_lr': optim.lr_scheduler.StepLR,
        'multi_step_lr': optim.lr_scheduler.MultiStepLR,
        'exponential_lr': optim.lr_scheduler.ExponentialLR,
        'cosine_annealing_lr': optim.lr_scheduler.CosineAnnealingLR,
        'one_cycle_lr': optim.lr_scheduler.OneCycleLR,
        'cosine_annealing_warm_restarts': optim.lr_scheduler.CosineAnnealingWarmRestarts,
    }

    learning_rate = config.train.learning_rate

    return partial(
        schedulers_map[learning_rate],
        **config.learning_rate_schedulers[learning_rate]
    )
