import os
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

import importlib.util
import pandas as pd

from moses.models_storage import ModelsStorage


def load_module(name, path):
    dirname = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(dirname, path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module

MODELS = ModelsStorage()
split_dataset = load_module('split_dataset', 'split_dataset.py')
eval_script = load_module('eval', 'eval.py')

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
    parser.add_argument('--n_samples', type=int, default=100,
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
    
    parser.add_argument('--wandb_entity', type=str,
                        help='Wandb entity name')
    parser.add_argument('--wandb_project', type=str, default='Moses',
                        help='Wandb project name')
    parser.add_argument('--nowandb', type=int, default=1,
                        choices=[0, 1], help='Disable wandb')
    
    
    parser.add_argument('--model_save_time', type=str,
                        help='Model save time')
    parser.add_argument('--load_epoch', type=str,
                        help='Model epoch to load')
    
    parser.add_argument('--use_selfies', type=int, default=0,
                        choices=[0, 1], help='Use selfies format')
    
    return parser


def eval_metrics(config, model, test_path, test_scaffolds_path,
                 ptest_path, ptest_scaffolds_path, train_path, model_starttime):
    print('Computing metrics...')
    eval_parser = eval_script.get_parser()
    
    args = [
        '--gen_path', get_generation_path(config, model, model_starttime),
        '--n_jobs', str(config.n_jobs),
        '--device', config.device,
    ]
    if test_path is not None:
        args.extend(['--test_path', test_path]) # test.csv
    if test_scaffolds_path is not None:
        args.extend(['--test_scaffolds_path', test_scaffolds_path]) # test_scaffolds.csv
    if ptest_path is not None:
        args.extend(['--ptest_path', ptest_path]) # stats_test.npz
    if ptest_scaffolds_path is not None:
        args.extend(['--ptest_scaffolds_path', ptest_scaffolds_path]) # stats_test_scaffolds.npz
    if train_path is not None:
        args.extend(['--train_path', train_path]) # train.csv
    
    eval_config = eval_parser.parse_args(args)
    
    eval_config.data = config.data
    
    if config.use_selfies:
        eval_config.use_selfies = True
    else:
        eval_config.use_selfies = False
    
    dict1 = vars(config)
    dict2 = vars(eval_config)
    
    whole_config = dict1.copy()
    whole_config.update(dict2)
    whole_config = argparse.Namespace(**whole_config)
    
    metrics = eval_script.main(whole_config, print_metrics=False)

    return metrics



def main(config):
    
    save_time = config.model_save_time
    model = config.model
    
    train_path = f"moses/dataset/data/{config.data}/train.csv"
    test_path = f"moses/dataset/data/{config.data}/test.csv"
    test_scaffolds_path = None
    ptest_path = f"moses/dataset/data/{config.data}/test_stats.npz"
    ptest_scaffolds_path = config.ptest_scaffolds_path
    
    model_metrics = eval_metrics(config, model, test_path, test_scaffolds_path,
                                 ptest_path, ptest_scaffolds_path, train_path, save_time)
    
    print("Metrics computed successfully!")
    
    table = pd.DataFrame([model_metrics]).T
    if len(config.experiment_suff) > 0:
        metrics_path = f'{config.data}_{model}_{config.experiment_suff}_{save_time}'
    else:
        metrics_path = f'{config.data}_{model}_{save_time}'
    metrics_path = os.path.join(config.checkpoint_dir, metrics_path, config.metrics+f'_{model}.csv')
    table.to_csv(metrics_path, header=False)
    
if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)
