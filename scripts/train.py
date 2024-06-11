import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
import torch
import rdkit

from moses.script_utils import add_train_args, read_data_csv, set_seed
from moses.models_storage import ModelsStorage
from moses.dataset import get_dataset

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

MODELS = ModelsStorage()


def get_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title='Models trainer script', description='available models'
    )
    for model in MODELS.get_model_names():
        add_train_args(
            MODELS.get_model_train_parser(model)(
                subparsers.add_parser(model)
            )
        )
    return parser


def main(model, config):
    set_seed(config.seed)
    device = torch.device(config.device)

    if config.config_save is not None:
        if isinstance(config.reg_prop_tasks, str):
            config.reg_prop_tasks = [c.strip() for c in config.reg_prop_tasks.split(',')]
        torch.save(config, config.config_save)

    # For CUDNN to work properly
    if device.type.startswith('cuda'):
        torch.cuda.set_device(device.index or 0)
    if config.train_load is None:
        train_data = get_dataset('train', config)
    else:
        train_data = read_data_csv(config.train_load, config)
    if config.val_load is None:
        val_data = get_dataset('test', config)
    else:
        val_data = read_data_csv(config.val_load, config)

    trainer = MODELS.get_model_trainer(model)(config)

    if config.vocab_load is not None:
        assert os.path.exists(config.vocab_load), \
            'vocab_load path does not exist!'
        vocab = torch.load(config.vocab_load)
    else:
        vocab = trainer.get_vocabulary(train_data, config)

    print(f'Vocabulary size: {len(vocab)}')

    for idx in range(len(vocab)):
        print(vocab.id2char(idx))

    if config.vocab_save is not None:
        torch.save(vocab, config.vocab_save)

    model = MODELS.get_model_class(model)(vocab, config).to(device)
    trainer.fit(model, train_data, val_data)

    model = model.to('cpu')
    torch.save(model.state_dict(), config.model_save)


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_args()
    model = sys.argv[1]
    main(model, config)
