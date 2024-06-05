import argparse
import os
import sys
import torch
import rdkit
import pandas as pd
import numpy as np
import wandb
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from moses.models_storage import ModelsStorage
from moses.script_utils import add_opt_args, set_seed, ManualAdamOpt
from sklearn.gaussian_process import GaussianProcessRegressor
from moses.metrics import QED, SA, logP
from rdkit import Chem
import selfies as sf


lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

MODELS = ModelsStorage()


def get_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title='Models mol optimizer script', description='available models')
    for model in MODELS.get_model_names():
        add_opt_args(subparsers.add_parser(model))
    return parser


def collate_fun(data, model):
    '''
    transform molecule strings to embedded tensors
    data: list of strings
    '''
    x = data.copy()
    x_ids = [model.vocabulary.string2ids(string) for string in x]

    # sort by token length not by string length
    combined = list(zip(x, x_ids))
    combined_sorted = sorted(combined, key=lambda pair: len(pair[-1]), reverse=True)
    x_sorted, _ = zip(*combined_sorted)

    tensors = [model.string2tensor(string)
               for string in x_sorted]
    return tensors, x_sorted

def fit_gp(Z, y):
    clf = GaussianProcessRegressor(random_state=42)
    clf.fit(Z, y)
    return clf

def objective_f(x, clf):
    return clf.predict(x.reshape(1, -1))[0]

def load_props(mol_strings, config):
    '''
    mol_strings: list of strings indicating each molecules. either selfies or smiles'''
    if config.use_selfies:
        df = pd.DataFrame(mol_strings, columns=['SELFIES'])
        df.insert(0, 'SMILES', df['SELFIES'].apply(sf.decoder))

    else: 
        df = pd.DataFrame(mol_strings, columns=['SMILES'])

    df['Romol'] = df['SMILES'].apply(Chem.MolFromSmiles)
  

    df['QED'] = df['Romol'].apply(QED)
    df['SA'] = df['Romol'].apply(SA)
    df['logP'] = df['Romol'].apply(logP)
    df['objective'] = 5*df['QED'] - df['SA']
    df = df.drop(columns=['Romol'])
    return df


def main(model, config):
    set_seed(config.seed)
    device = torch.device(config.device)

    # For CUDNN to work properly:
    if device.type.startswith('cuda'):
        torch.cuda.set_device(device.index or 0)

    model_config = torch.load(config.config_load)
    model_vocab = torch.load(config.vocab_load)
    model_state = torch.load(config.model_load)

    model = MODELS.get_model_class(model)(model_vocab, model_config)
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()

##########
    
    ## Load the GPR training data
    # load Z_train data
    print('Loading GPR training data...')

    train_data = pd.read_csv(config.gpr_fit_path)
    if model_config.use_selfies:
        
        mol_train = train_data['SELFIES'].values
    else:
        mol_train = train_data['SMILES'].values
    X_tensors, mol_sorted = collate_fun(mol_train, model)
    X_train = tuple(data.to(model.device) for data in X_tensors)

    Z_train, _, _ = model.forward_encoder(X_train)  # use mu as Z_train
    Z_train = Z_train.detach().cpu().numpy()

    # load y_train data 
    sorted_df = load_props(mol_sorted, model_config)
    y_train = sorted_df['objective'].values

    ## Train the Gaussian Process model
    print('Training the Gaussian Process model...')
    clf = fit_gp(Z_train, y_train)

    # Load the starting molecules for optimization
    print('Loading the starting molecules for optimization...')
    df = pd.read_csv(config.opt_start_path)
    if model_config.use_selfies:
        initial_mol = df['SELFIES'].values
    else:
        initial_mol = df['SMILES'].values
    initial_x, initial_sorted = collate_fun(initial_mol, model)
    initial_z, _, _ = model.forward_encoder(initial_x)
    
    initial_z = initial_z.detach().cpu().numpy()

    # Adam Gradient Ascent using the class
    opt_Z = []
    pred_objs = []

    for j in range(initial_z.shape[0]):
        print(f'Optimizing molecule {j}...')
        initial_point = np.array(initial_z[j])
        optimizer = ManualAdamOpt(objective_f, clf, learning_rate=config.opt_lr, max_iter=config.opt_iter, \
                                  tolerance=config.opt_tol, beta1=config.opt_b1, beta2=config.opt_b2, epsilon=config.opt_eps)
        opt_z = optimizer.optimize(initial_point)
        pred_obj = clf.predict(opt_z.reshape(1,-1))

        opt_Z.append(opt_z)
        pred_objs.append(pred_obj)        
        
        if j%1 == 0:
            optZ_np = np.array(opt_Z)
            optZ_torch = torch.tensor(optZ_np, dtype=torch.float32) 
            #optZ_torch = torch.from_numpy(optZ_np)
            #optZ_torch = torch.tensor(optZ_torch, dtype=torch.float32)
            pred_objs_np = np.array(pred_objs)

            # decode optimized Z to mols
            model.eval()
            opt_mols = model.sample(optZ_torch.shape[0], z=optZ_torch, test=True)

            # save optimized mols with related infos
            
            optimized_df = load_props(opt_mols, model_config)
            optimized_df['opt_z'] = opt_Z
            optimized_df['pred objective'] = pred_objs_np

            starting_df = load_props(initial_sorted, model_config)
            starting_df.columns = [col + '_ini' for col in starting_df.columns]
            optimized_df = pd.concat([starting_df, optimized_df], axis=1)

            optimized_df.to_csv(config.opt_save, index=False)
    

    
    print('Optimized molecules saved!')

if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_args()
    model = sys.argv[1]
    main(model, config)
    
