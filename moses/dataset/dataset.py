import os
import copy
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
        config: Configuration object with attributes 
            'data', 'reg_prop_tasks', and 'use_selfies'

    Returns:
        list: If 'reg_prop_tasks' in config, returns data with 'y'. 
                Otherwise, returns a list with SMILES or SELFIES strings.
    """
    if split not in AVAILABLE_SPLITS:
        raise ValueError(
            f"Unknown split {split}. "
            f"Available splits: {AVAILABLE_SPLITS}"
        )
    base_path = os.path.dirname(__file__)
    path = os.path.join(base_path, f'data/{config.data}', split+'.csv')
    df = pd.read_csv(path)

    if hasattr(config, 'reg_prop_tasks'):
        if isinstance(config.reg_prop_tasks, str):
            config.reg_prop_tasks = [c.strip() for c in config.reg_prop_tasks.split(',')]
        cols = copy.copy(config.reg_prop_tasks)
        cols.insert(0, 'SELFIES' if config.use_selfies else 'SMILES')
        data = df[cols].values
        return data

    return df['SELFIES'].values if config.use_selfies else df['SMILES'].values

# TODO: consider selfies statistics
def get_statistics(split='test', config=None):
    base_path = os.path.dirname(__file__)
    path = os.path.join(base_path, f'data/{config.data}', split+'_stats.npz')
    return np.load(path, allow_pickle=True)['stats'].item()