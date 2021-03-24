# -*- coding: utf-8 -*-
import yaml
from easydict import EasyDict


class Config():
    def __init__(self, source='config.yml'):
        d = _read_config(source=source)
        for k, v in d.items():
            setattr(self, k, v)

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            raise AttributeError(name)

    def __getitem__(self, name):
        return getattr(self, name)

    @staticmethod
    def validate_configs(user_config='config.yml', default_config='config-default.yml'):
        user_dict = _read_config(user_config)
        sample_dict = _read_config(default_config)

        str_to_print = str(user_config)
        result1 = _compare_dict_keys(user_dict, sample_dict, str_to_print, enable_warnings=True)

        str_to_print = str(default_config)
        result2 = _compare_dict_keys(sample_dict, user_dict, str_to_print, enable_warnings=False)
        return result1 and result2

def _read_config(source):
    if isinstance(source, str):
        with open(source, 'r') as stream:
            config = EasyDict(yaml.safe_load(stream))
        if config is None:
            print(f'{source} is empty. Fill it, please.')
            exit()
    else:
        raise TypeError('Unexpected source to load config')
    return config

def _compare_dict_keys(subconfig1, subconfig2, str_to_print, enable_warnings=True):
    result = True
    if type(subconfig1) != type(subconfig2):
        print(f'{str_to_print} have different types: {type(subconfig1)} and {type(subconfig2)}')
        result = False
    elif isinstance(subconfig2, dict):
        for key, value in subconfig2.items():
            if key not in subconfig1:
                print(str_to_print, _get_delimiter_key(), _do_bold(key))
                result = False
            elif isinstance(value, dict) or isinstance(value, list):
                new_str = f'{str_to_print}{_get_delimiter_key_spaces()}{key}'
                cfg1, cfg2 = subconfig1[key], value
                result = _compare_dict_keys(cfg1, cfg2, new_str, enable_warnings=enable_warnings) and result
    elif isinstance(subconfig2, list):
        if len(subconfig1) == len(subconfig2):
            for idx, (cfg1, cfg2) in enumerate(zip(subconfig1, subconfig2)):
                if isinstance(cfg1, dict) or isinstance(cfg2, list):
                    new_str = f'{str_to_print}[{idx}]'
                    result = _compare_dict_keys(cfg1, cfg2, new_str, enable_warnings=enable_warnings) and result
        elif enable_warnings:
            pos = str_to_print.index('>')
            key_description = str_to_print[pos + 2:]
            print(f'Warning: {key_description} have different length')

    return result


def _do_bold(s):
    bold = '\033[1m'
    end_bold = '\033[0m'
    return bold + s + end_bold


def _get_delimiter_key():
    return '->'


def _get_delimiter_key_spaces():
    return ' ' + _get_delimiter_key() + ' '
