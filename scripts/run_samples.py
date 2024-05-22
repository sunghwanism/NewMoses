import os
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
import datetime

import importlib.util
import pandas as pd
import torch

from moses.models_storage import ModelsStorage
from moses.vae import VAE


def load_module(name, path):
    dirname = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(dirname, path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module

MODELS = ModelsStorage()
split_dataset = load_module('split_dataset', 'split_dataset.py')
sampler_script = load_module('sample', 'sample.py')

def get_model_path(config, model, model_starttime):
    if len(config.experiment_suff) > 0:
        unique_folder = f'{config.data}_{model}_{config.experiment_suff}_{model_starttime}'
    else:
        unique_folder = f'{config.data}_{model}_{model_starttime}'
        
    unique_folder_path = os.path.join(config.checkpoint_dir, unique_folder)
            
    return os.path.join(
        unique_folder_path, model + config.experiment_suff + f'_model_{config.load_epoch}.pt'
    )

def get_log_path(config, model, model_starttime):
    if len(config.experiment_suff) > 0:
        unique_folder = f'{config.data}_{model}_{config.experiment_suff}_{model_starttime}'
    else:
        unique_folder = f'{config.data}_{model}_{model_starttime}'
    unique_folder_path = os.path.join(config.checkpoint_dir, unique_folder)

    return os.path.join(
        unique_folder_path, model + config.experiment_suff + '_log.txt'
    )


def get_config_path(config, model,model_starttime):
    if len(config.experiment_suff) > 0:
        unique_folder = f'{config.data}_{model}_{config.experiment_suff}_{model_starttime}'
    else:
        unique_folder = f'{config.data}_{model}_{model_starttime}'
        
    unique_folder_path = os.path.join(config.checkpoint_dir, unique_folder)
        
    return os.path.join(
        unique_folder_path, model + config.experiment_suff + '_config.pt'
    )


def get_vocab_path(config, model, model_starttime):
    if len(config.experiment_suff) > 0:
        unique_folder = f'{config.data}_{model}_{config.experiment_suff}_{model_starttime}'
    else:
        unique_folder = f'{config.data}_{model}_{model_starttime}'
        
    unique_folder_path = os.path.join(config.checkpoint_dir, unique_folder)
        
    return os.path.join(
        unique_folder_path, model + config.experiment_suff + '_vocab.pt'
    )


def get_generation_path(config, model, model_starttime):
    if len(config.experiment_suff) > 0:
        unique_folder = f'{config.data}_{model}_{config.experiment_suff}_{model_starttime}'
    else:
        unique_folder = f'{config.data}_{model}_{model_starttime}'
        
    unique_folder_path = os.path.join(config.checkpoint_dir, unique_folder)
        
    return os.path.join(
        unique_folder_path,
        model + config.experiment_suff + '_generated.csv'
    )

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='all',
                        choices=['all'] + MODELS.get_model_names(),
                        help='Which model to run')
    
    parser.add_argument('--test_path',
                        type=str, required=False,
                        help='Path to test molecules csv')
    parser.add_argument('--test_scaffolds_path',
                        type=str, required=False,
                        help='Path to scaffold test molecules csv')
    parser.add_argument('--train_path',
                        type=str, required=False,
                        help='Path to train molecules csv')
    parser.add_argument('--ptest_path',
                        type=str, required=False,
                        help='Path to precalculated test npz')
    parser.add_argument('--ptest_scaffolds_path',
                        type=str, required=False,
                        help='Path to precalculated scaffold test npz')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory for checkpoints')
    parser.add_argument('--n_samples', type=int, default=30000,
                        help='Number of samples to sample')
    parser.add_argument('--n_jobs', type=int, default=4,
                        help='Number of threads')
    parser.add_argument('--device', type=str, default='cpu',
                        help='GPU device index in form `cuda:N` (or `cpu`)')
    parser.add_argument('--metrics', type=str, default='metrics.csv',
                        help='Path to output file with metrics')
    parser.add_argument('--train_size', type=int, default=None,
                        help='Size of training dataset')
    parser.add_argument('--test_size', type=int, default=None,
                        help='Size of testing dataset')
    parser.add_argument('--experiment_suff', type=str, default='',
                        help='Experiment suffix to break ambiguity')
    parser.add_argument('--data', type=str, default='ZINC', 
                        choices=['ZINC', 'QM9'], help='Dataset to use')
    parser.add_argument('--use_selfies', type=int, default=0,
                        choices=[0, 1], help='Use selfies format')
    
    parser.add_argument('--wandb_entity', type=str,
                        help='Wandb entity name')
    parser.add_argument('--wandb_project', type=str, default='Moses',
                        help='Wandb project name')
    parser.add_argument('--nowandb', type=int, default=1,
                        choices=[0, 1], help='Disable wandb')
    parser.add_argument('--debug_mode', type=int, default=0,
                        choices=[0, 1], help='Debug mode')
    
    
    parser.add_argument('--model_save_time', type=str,
                        help='Model save time')
    parser.add_argument('--load_epoch', type=str,
                        help='Model epoch to load')
    
    return parser


def sample_from_model(config, model, model_starttime):
    print('Sampling...')
    model_path = get_model_path(config, model, model_starttime)
    config_path = get_config_path(config, model, model_starttime)
    vocab_path = get_vocab_path(config, model, model_starttime)

    assert os.path.exists(model_path), (
        "Can't find model path for sampling: '{}'".format(model_path)
    )
    assert os.path.exists(config_path), (
        "Can't find config path for sampling: '{}'".format(config_path)
    )
    assert os.path.exists(vocab_path), (
        "Can't find vocab path for sampling: '{}'".format(vocab_path)
    )

    sampler_parser = sampler_script.get_parser()
    sampler_config = sampler_parser.parse_known_args(
        [model] + sys.argv[1:] +
        ['--device', config.device,
         '--model_load', model_path,
         '--config_load', config_path,
         '--vocab_load', vocab_path,
         '--gen_save', get_generation_path(config, model, model_starttime, ),
         '--n_samples', str(config.n_samples)]
    )[0]
    
    
    sampler_config.data = config.data
    
    if config.use_selfies:
        sampler_config.use_selfies = True
    else:
        sampler_config.use_selfies = False
    
    dict1 = vars(config)
    dict2 = vars(sampler_config)
    
    whole_config = dict1.copy()
    whole_config.update(dict2)
    whole_config = argparse.Namespace(**whole_config)
    
    sampler_script.main(model, whole_config)


def main(config):
    
    save_time = config.model_save_time
    model = config.model
    sample_from_model(config, model, save_time)

if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)
