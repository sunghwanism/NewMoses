import os
import argparse
import sys
sys.path.append("./moses")
import datetime

import importlib.util
import pandas as pd
import wandb

from models_storage import ModelsStorage


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
trainer_script = load_module('train', 'train.py')
sampler_script = load_module('sample', 'sample.py')


def get_model_path(config, model, model_starttime):
    if len(config.experiment_suff) > 0:
        unique_folder = f'{config.data}_{model}_{config.experiment_suff}_{model_starttime}'
    else:
        unique_folder = f'{config.data}_{model}_{model_starttime}'
        
    unique_folder_path = os.path.join(config.checkpoint_dir, unique_folder)
    
    if not os.path.exists(unique_folder_path):
        os.mkdir(unique_folder_path)
        
    unique_folder_path = os.path.join(config.checkpoint_dir, config.test_path)
        
    return os.path.join(
        unique_folder_path, model + config.experiment_suff + '_model.pt'
    )

def get_log_path(config, model, model_starttime):
    if len(config.experiment_suff) > 0:
        unique_folder = f'{config.data}_{model}_{config.experiment_suff}_{model_starttime}'
    else:
        unique_folder = f'{config.data}_{model}_{model_starttime}'
    unique_folder_path = os.path.join(config.checkpoint_dir, unique_folder)
    if not os.path.exists(unique_folder_path):
        os.mkdir(unique_folder_path)
        
    return os.path.join(
        unique_folder_path, model + config.experiment_suff + '_log.txt'
    )


def get_config_path(config, model,model_starttime):
    if len(config.experiment_suff) > 0:
        unique_folder = f'{config.data}_{model}_{config.experiment_suff}_{model_starttime}'
    else:
        unique_folder = f'{config.data}_{model}_{model_starttime}'
        
    unique_folder_path = os.path.join(config.checkpoint_dir, unique_folder)
    if not os.path.exists(unique_folder_path):
        os.mkdir(unique_folder_path)
        
    return os.path.join(
        unique_folder_path, model + config.experiment_suff + '_config.pt'
    )


def get_vocab_path(config, model, model_starttime):
    if len(config.experiment_suff) > 0:
        unique_folder = f'{config.data}_{model}_{config.experiment_suff}_{model_starttime}'
    else:
        unique_folder = f'{config.data}_{model}_{model_starttime}'
        
    unique_folder_path = os.path.join(config.checkpoint_dir, unique_folder)
    if not os.path.exists(unique_folder_path):
        os.mkdir(unique_folder_path)
        
    return os.path.join(
        unique_folder_path, model + config.experiment_suff + '_vocab.pt'
    )


def get_generation_path(config, model, model_starttime):
    if len(config.experiment_suff) > 0:
        unique_folder = f'{config.data}_{model}_{config.experiment_suff}_{model_starttime}'
    else:
        unique_folder = f'{config.data}_{model}_{model_starttime}'
        
    unique_folder_path = os.path.join(config.checkpoint_dir, unique_folder)
    if not os.path.exists(unique_folder_path):
        os.mkdir(unique_folder_path)
        
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
    parser.add_argument('--data', default='ZINC', 
                        help='Dataset to use')
    parser.add_argument('--use_selfies', default=False,
                        help='Use selfies format')
    
    parser.add_argument('--wandb_entity', type=str,
                        help='Wandb entity name')
    parser.add_argument('--wandb_project', type=str, default='Moses',
                        help='Wandb project name')
    parser.add_argument('--nowandb', type=int, default=0,
                        choices=[0, 1], help='Disable wandb')
    
    return parser


def init_wandb(config):
    if not config.nowandb:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        wandb.init(project=config.wandb_project, entity=config.wandb_entity, config=config)
        config.wandb_url = wandb.run.get_url()


def train_model(config, model, train_path, test_path, model_starttime):
    print('Training...')
    model_path = get_model_path(config, model, model_starttime)
    config_path = get_config_path(config, model, model_starttime)
    vocab_path = get_vocab_path(config, model, model_starttime)
    log_path = get_log_path(config, model, model_starttime)

    if os.path.exists(model_path) and \
            os.path.exists(config_path) and \
            os.path.exists(vocab_path):
        return

    trainer_parser = trainer_script.get_parser()

    args = [
        '--device', config.device,
        '--model_save', model_path,
        '--config_save', config_path,
        '--vocab_save', vocab_path,
        '--log_file', log_path,
        '--n_jobs', str(config.n_jobs),
        '--data', config.data,

        # SELFIES로 할지말지 결정하는 argument 추가
        '--use_selfies', str(config.use_selfies)
    ]
        
    if train_path is not None:
        args.extend(['--train_load', train_path])
        
    if test_path is not None:
        args.extend(['--val_load', test_path])

    trainer_config = trainer_parser.parse_known_args(
         [model] + sys.argv[1:] + args
    )[0]
    
    trainer_config.data = config.data
    
    if config.use_selfies:
        trainer_config.use_selfies = True
    else:
        trainer_config.use_selfies = False
        
    dict1 = vars(config)
    dict2 = vars(trainer_config)
    
    whole_config = dict1.copy()
    whole_config.update(dict2)
    whole_config = argparse.Namespace(**whole_config)
    
    print("-----"*25)
    print("[Whole config]")
    print(whole_config)
    print("-----"*25)
    
    init_wandb(whole_config)
    trainer_script.main(model, whole_config)


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
         '--gen_save', get_generation_path(config, model, model_starttime),
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
        args.extend(['--test_path', test_path])
    if test_scaffolds_path is not None:
        args.extend(['--test_scaffolds_path', test_scaffolds_path])
    if ptest_path is not None:
        args.extend(['--ptest_path', ptest_path])
    if ptest_scaffolds_path is not None:
        args.extend(['--ptest_scaffolds_path', ptest_scaffolds_path])
    if train_path is not None:
        args.extend(['--train_path', train_path])
    
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
    if not os.path.exists(config.checkpoint_dir):
        os.mkdir(config.checkpoint_dir)
        
    model_starttime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    train_path = config.train_path
    test_path = config.test_path
    test_scaffolds_path = config.test_scaffolds_path
    ptest_path = config.ptest_path
    ptest_scaffolds_path = config.ptest_scaffolds_path

    models = (MODELS.get_model_names()
              if config.model == 'all'
              else [config.model])
    for model in models:
        train_model(config, model, train_path, test_path, model_starttime)
        sample_from_model(config, model, model_starttime)

    for model in models:
        model_metrics = eval_metrics(config, model,
                                     test_path, test_scaffolds_path,
                                     ptest_path, ptest_scaffolds_path,
                                     train_path,
                                     model_starttime)
        table = pd.DataFrame([model_metrics]).T
        if not config.nowandb:
            wandb.log({'metrics': table})
        if len(models) == 1:
            metrics_path = ''.join(
                os.path.splitext(config.metrics)[:-1])+f'_{model}.csv'
        else:
            metrics_path = config.metrics
        table.to_csv(metrics_path, header=False)


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)
