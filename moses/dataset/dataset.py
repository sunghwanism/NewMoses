import os
import numpy as np
import pandas as pd

import selfies as sf

AVAILABLE_SPLITS = ['train', 'test', 'test_scaffolds']


def get_dataset(split='train', config=None):
    """
    Loads MOSES dataset

    Arguments:
        split (str): split to load. Must be
            one of: 'train', 'test', 'test_scaffolds'

    Returns:
        list with SMILES strings
    """
    if split not in AVAILABLE_SPLITS:
        raise ValueError(
            f"Unknown split {split}. "
            f"Available splits: {AVAILABLE_SPLITS}"
        )
    base_path = os.path.dirname(__file__)
    if split not in AVAILABLE_SPLITS:
        raise ValueError(
            f"Unknown split {split}. "
            f"Available splits: {AVAILABLE_SPLITS}")
    path = os.path.join(base_path, f'data/{config.data}', split+'.csv')
    smiles = pd.read_csv(path)['SMILES'][:10].values

    print(f"in data loading : {config.use_selfies}")
    if config.use_selfies:
        selfies = [sf.encoder(smile) for smile in smiles]
        return selfies

    else:
        return smiles


def get_statistics(split='test', config=None):
    base_path = os.path.dirname(__file__)
    path = os.path.join(base_path, f'data/{config["data"]}', split+'_stats.npz')
    return np.load(path, allow_pickle=True)['stats'].item()

