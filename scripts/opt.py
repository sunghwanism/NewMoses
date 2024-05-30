import argparse
import os
import sys
import torch
import rdkit
import pandas as pd
import wandb
from tqdm.auto import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from moses.models_storage import ModelsStorage
from moses.script_utils import add_opt_args, set_seed, ManualAdamOpt
from sklearn.gaussian_process import GaussianProcessRegressor


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
    processed_data = tuple(data.to(model.device) for data in tensors)
    return processed_data

def fit_gp(Z, y):
    clf = GaussianProcessRegressor(random_state=42)
    clf.fit(Z, y)
    return clf

def objective_f(x, clf):
    return clf.predict(x.reshape(1, -1))[0]


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

    samples = []
    n = config.n_iter
    with tqdm(total=config.n_iter, desc='Optimizing Molecules') as T:
        while n > 0:
            current_samples = model.sample(
                min(n, config.n_batch), config.max_len
            )
            samples.extend(current_samples)

            n -= len(current_samples)
            T.update(len(current_samples))

##########
    
    # Load the GPR training data
    train_data = pd.read_csv(config.gpr_fit_path)

    mol_train = train_data['MOL'].values
    X_train = collate_fun(mol_train, model)
    Z_train, _, _ = model.forward_encoder(X_train)  # use mu as Z_train

    y_train = train_data['objective'].values
    
    # Train the Gaussian Process model
    clf = fit_gp(Z_train, y_train)

    # Load the starting molecules for optimization
    df = pd.read_csv(config.opt_start_path)
    initial_mol = df['MOL'].values
    initial_x = collate_fun(initial_mol, model)
    initial_z, _, _ = model.forward_encoder(initial_x)
    
    initial_z.detach().cpu().numpy()

    # Adam Gradient Ascent using the class
    for j in range(initial_z.shape[0]):
        initial_point = np.array(initial_z[j])
        optimizer = ManualAdamOpt(objective_f, clf, learning_rate=config.opt_lr, max_iter=config.opt_iter, \
                                  tolerance=config.opt_tol, beta1=config.opt_b1, beta2=config.opt_b2, epsilon=config.opt_eps)
        optimal_x = optimizer.optimize(initial_point)

            

    if config.use_selfies:
        samples = pd.DataFrame(samples, columns=['SELFIES'])
    else:
        samples = pd.DataFrame(samples, columns=['SMILES'])
        
    if not config.nowandb:
        wandb.log({'samples': samples})
    samples.to_csv(config.gen_save, index=False)


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_args()
    model = sys.argv[1]
    main(model, config)
