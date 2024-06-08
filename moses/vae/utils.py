import os
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from rdkit import Chem

from moses.utils import fast_verify
from moses.vae import VAE, VAETrainer
from moses.vae_property import VAEPROPERTYTrainer
from moses.dataset import get_dataset


class VAEUtils(object):
    def __init__(self, vocab, config, model_path):
        self.vocab = vocab
        self.config = config
        
        # Load model
        self.model = VAE(vocab, config)
        self.model.load_state_dict(torch.load(model_path))
        print('Model loaded from', model_path)
        self.model.eval()
               
        # Load data as numpy array
        config.reg_prop_tasks = ['logP', 'qed', 'SAS', 'obj']
        self.train = get_dataset('train', config=config)
        self.test = get_dataset('test', config=config)
        
        self.encode, self.decode = self.enc_dec_functions()

        self.estimate_estandarization()
        return
    
    def estimate_estandarization(self):
        print('Standarization: estimating mu and std values ...', end='')

        # sample Z space
        samples = self.random_molecules(size=50000)
        batch = 2500
        self.config.n_batch = batch
        Z = np.zeros((len(samples), self.config.d_z))
        trainer = VAEPROPERTYTrainer(self.config)
        train_loader = trainer.get_dataloader(self.model, samples, shuffle=True)
        with torch.no_grad():
            for step, input_batch in tqdm(enumerate(train_loader)):
                x_batch, y_batch = input_batch
                x_batch = tuple(data.to(self.model.device) for data in x_batch)
                y_batch = y_batch.to(self.model.device)
                
                mu = self.encode(x_batch, standardize=False) 
                Z[step*batch:(step+1)*batch, :] = mu.copy() 

        self.mu = np.mean(Z, axis=0)
        self.std = np.std(Z, axis=0)
        self.Z = self.standardize_z(Z)

        print('done!')
        return
    
    def standardize_z(self, z):
        return (z - self.mu) / self.std

    def unstandardize_z(self, z):
        return (z * self.std) + self.mu
    
    def random_molecules(self, size=None):
        if size is None:
            return self.train
        else:
            np.random.seed(self.config.seed)
            indices = np.random.choice(len(self.train), size)
            return self.train[indices]     
    
    def perturb_z(self, z, noise_norm, constant_norm=False):
        if noise_norm > 0.0:
            noise_vector = np.random.normal(0, 1, size=z.shape)
            noise_vector = noise_vector / np.linalg.norm(noise_vector, axis=1, keepdims=True)
            if constant_norm:
                return z + (noise_norm * noise_vector)
            else:
                noise_amp = np.random.uniform(0, noise_norm, size=(z.shape[0], 1))
                return z + (noise_amp * noise_vector)
        else:
            return z
        
    def smiles_distance_z(self, moles, z0):
        x =[self.model.string2tensor(string) for string in moles]
        z_rep = self.encode(x) 
        return np.linalg.norm(z0 - z_rep, axis=1)
    
    def prep_mol_df(self, smiles, z):
        df = pd.DataFrame({'smiles': smiles})
        sort_df = pd.DataFrame(df[['smiles']].groupby(
            by='smiles').size().rename('count').reset_index())
        df = df.merge(sort_df, on='smiles')
        df.drop_duplicates(subset='smiles', inplace=True)
        df = df[df['smiles'].apply(fast_verify)]
        if len(df) > 0:
            df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
        if len(df) > 0:
            df = df[pd.notnull(df['mol'])]
        if len(df) > 0:
            df['length'] = df['smiles'].apply(len)
            df.sort_values(by='length', ascending=False, inplace=True)
            df['distance'] = self.smiles_distance_z(df['smiles'], z)
            df['frequency'] = df['count'] / float(sum(df['count']))
            df = df[['smiles', 'distance', 'count', 'frequency', 'mol']]
            df.sort_values(by='distance', inplace=True)
            df.reset_index(drop=True, inplace=True)
        return df

    def z_to_smiles(self,
                    z,
                    decode_attempts=250,
                    noise_norm=0.0,
                    constant_norm=False,
                    early_stop=None):
        if not (early_stop is None):
            Z = np.tile(z, (25, 1))
            Z = self.perturb_z(Z, noise_norm, constant_norm)
            smiles = self.decode(Z, standardize=False)
            df = self.prep_mol_df(smiles, z)
            if len(df) > 0:
                low_dist = df.iloc[0]['distance']
                if low_dist < early_stop:
                    return df

        Z = np.tile(z, (decode_attempts, 1))
        Z = self.perturb_z(Z, noise_norm)
        smiles = self.decode(Z, standardize=False)
        df = self.prep_mol_df(smiles, z)
        return df
    
    def enc_dec_functions(self, standardized=True):
        print('Using standarized functions? {}'.format(standardized))
        
        @torch.no_grad()
        def encode(X, standardize=standardized):
            if standardize:
                mu, _, _, _ = self.model.forward_encoder(X)
                return self.standardize_z(mu.detach().cpu().numpy())
            else:
                mu, _, _, _ = self.model.forward_encoder(X)
                return mu.detach().cpu().numpy()
        
        @torch.no_grad()
        def decode(z, standardize=standardized):
                if standardize:
                    z = self.unstandardize_z(z)
                    z = torch.tensor(z).float().to(self.model.device)
                    return self.model.sample(n_batch=z.shape[0], 
                                             z=z, 
                                             test=True)
                else:
                    z = torch.tensor(z).float().to(self.model.device)
                    return self.model.sample(n_batch=z.shape[0], 
                                             z=z, 
                                             test=True)

        return encode, decode
                
    @torch.no_grad()
    def ls_sampler_w_prop(self, size=None, batch=2500, return_smiles=False):
        if self.train is None:
            print('use this sampler only for external property files')
            return

        np.random.seed(999)
        idxs = np.random.choice(len(self.train), size)
        samples = self.train[idxs]
        
        self.config.n_batch = batch
        X = []
        Z = np.zeros((len(samples), self.config.d_z))
        Y = np.zeros((len(samples), len(self.config.reg_prop_tasks)))
        trainer = VAEPROPERTYTrainer(self.config)
        train_loader = trainer.get_dataloader(self.model, samples, shuffle=False)
        with torch.no_grad():
            for step, input_batch in tqdm(enumerate(train_loader)):
                x_batch, y_batch = input_batch
                x_batch = tuple(data.to(self.model.device) for data in x_batch)
                y_batch = y_batch.to(self.model.device)
                
                mu = self.encode(x_batch, standardize=False)
                X.extend([self.model.tensor2string(x) for x in x_batch])
                Z[step*batch:(step+1)*batch, :] = mu.copy()
                Y[step*batch:(step+1)*batch, :] = y_batch.cpu().numpy()

        if return_smiles:
            return Z, Y, X
        
        return Z, Y