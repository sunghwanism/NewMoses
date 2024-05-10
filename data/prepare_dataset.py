import argparse
import gzip
import csv
import logging
from functools import partial
from multiprocessing import Pool
import pandas as pd

from tqdm.auto import tqdm
from rdkit import Chem

import os
import sys
sys.path.append("../moses")

from moses.metrics import mol_passes_filters, compute_scaffold


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prepare dataset")


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--output', '-o',
        type=str, default='./data/dataset_v1.csv',
        help='Path for constructed dataset'
    )
    parser.add_argument(
        '--seed', type=int, default=0, help='Random state'
    )
    parser.add_argument(
        '--data', type=str,
        default='QM9',
        help='type of dataset'
    )
    parser.add_argument(
        '--zinc', type=str,
        default='./data/11_p0.smi.gz',
        help='path to .smi.gz file with ZINC smiles'
    )
    parser.add_argument(
        '--qm9', type=str,
        default='./data/qm9_cleaned.txt.gz',
        help='path to .smi.gz file with QM9 smiles'
    )
    parser.add_argument(
        '--n_jobs', type=int, default=1,
        help='number of processes to use'
    )
    parser.add_argument(
        '--keep_ids', action='store_true', default=False,
        help='Keep ZINC ids in the final csv file'
    )
    parser.add_argument(
        '--isomeric', action='store_true', default=False,
        help='Save isomeric SMILES (non-isomeric by default)'
    )
    return parser


def process_molecule(mol_row, isomeric):
 #   mol_row = mol_row.decode('utf-8')
    smiles, _id = mol_row.split()
    if not mol_passes_filters(smiles):
        return None
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles),
                              isomericSmiles=isomeric)
    return _id, smiles


def unzip_dataset(path):
    logger.info("Unzipping dataset")
    with gzip.open(path) as smi:
        lines = smi.readlines()
        lines = [line.replace(b'\r\r\n', b'\n').replace(b'\t', b' ').decode('utf-8') for line in lines]

    return lines

def unzip_dataset_csv(path):
    logger.info("Unzipping dataset")
    with gzip.open(path, 'rt', encoding='utf-8') as file:
        reader = csv.reader(file)
        lines = [item for sublist in reader for item in sublist if item]  # Flatten and remove empty items
    return lines


def filter_lines(lines, n_jobs, isomeric):
    logger.info('Filtering SMILES')
    with Pool(n_jobs) as pool:
        process_molecule_p = partial(process_molecule, isomeric=isomeric)
        dataset = [
            x for x in tqdm(
                pool.imap_unordered(process_molecule_p, lines),
                total=len(lines),
                miniters=1000
            )
            if x is not None
        ]
        dataset = pd.DataFrame(dataset, columns=['ID', 'SMILES'])
        dataset = dataset.sort_values(by=['ID', 'SMILES'])
        dataset = dataset.drop_duplicates('ID')
        dataset = dataset.sort_values(by='ID')
        dataset = dataset.drop_duplicates('SMILES')
        dataset['scaffold'] = pool.map(
            compute_scaffold, dataset['SMILES'].values
        )
    return dataset


def split_dataset(dataset, seed):
    logger.info('Splitting the dataset')
    scaffolds = pd.value_counts(dataset['scaffold'])
    scaffolds = sorted(scaffolds.items(), key=lambda x: (-x[1], x[0]))
    test_scaffolds = set([x[0] for x in scaffolds[9::10]])
    dataset['SPLIT'] = 'train'
    test_scaf_idx = [x in test_scaffolds for x in dataset['scaffold']]
    dataset.loc[test_scaf_idx, 'SPLIT'] = 'test_scaffolds'
    test_idx = dataset.loc[dataset['SPLIT'] == 'train'].sample(
        frac=0.1, random_state=seed
    ).index
    dataset.loc[test_idx, 'SPLIT'] = 'test'
    dataset.drop('scaffold', axis=1, inplace=True)
    return dataset


def main(config):
    if config.data == 'ZINC':
        lines = unzip_dataset(config.zinc)
    if config.data == 'QM9':
        lines = unzip_dataset(config.qm9)
    dataset = filter_lines(lines, config.n_jobs, config.isomeric)
    dataset = split_dataset(dataset, config.seed)
    if not config.keep_ids:
        dataset.drop('ID', 1, inplace=True)
    dataset.to_csv(config.output, index=None)

if __name__ == '__main__':
    parser = get_parser()
    config, unknown = parser.parse_known_args()
    if len(unknown) != 0:
        raise ValueError("Unknown argument "+unknown[0])
    main(config)
