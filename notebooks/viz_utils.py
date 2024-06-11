import os
import sys
sys.path.append("..")

import numpy as np
import pandas as pd

import torch


from moses.vae import VAE
from moses.vae_property import VAEPROPERTY
from moses.vae.trainer import VAETrainer
from moses.vae_property.trainer import VAEPROPERTYTrainer 

import selfies as sf
from rdkit import Chem
from rdkit.Chem import PandasTools

# SLERP 함수 정의
def slerp(val, low, high):
    omega = np.arccos(np.clip(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high  # LERP (선형 보간)
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def InterpolationLoader(dataPATH, model_type, data_type, best_epoch, i_1, i_2):
    
    train_df = pd.read_csv(os.path.join(dataPATH, 'train.csv'))
    
    if model_type == "vae_property":
        if data_type == 'selfies':
            folder_path = "../checkpoints/ZINC250K_vae_property_obj_proploss_w0.1_selfies"
        else:
            folder_path = "../checkpoints/ZINC250K_vae_property_obj_proploss_w0.1_smiles"
            
    elif model_type == "vae":
        if data_type == 'selfies':
            folder_path = "../checkpoints/ZINC250K_vae_selfies"
        else:
            folder_path = "../checkpoints/ZINC250K_vae_smiles"      
    
    config = torch.load(f'{folder_path}/{model_type}_config.pt')
    vocab = torch.load(f'{folder_path}/{model_type}_vocab.pt')
    config.reg_prop_tasks = ['obj']
    
    print(f"Use Selfies: {config.use_selfies}")
    if model_type == "vae_property":
        print(config.reg_prop_tasks)

    cols = ['SELFIES' if config.use_selfies else 'SMILES', 'logP', 'qed', 'SAS', 'obj']
    
    train_df = train_df.iloc[[i_1, i_2],:]    
    train_data = train_df[cols].values
    
    print("Sample Input shape", train_data.shape)

    if model_type == "vae_property":
        #  model_path = f'{folder_path}/vae_property_model_0{best_epoch}.pt'
        model_path = f'{folder_path}/vae_property_model.pt'
        model = VAEPROPERTY(vocab, config)
        model.load_state_dict(torch.load(model_path))
        trainer = VAEPROPERTYTrainer(config)

    elif model_type == "vae":
        model_path = f'{folder_path}/vae_model.pt'
        model = VAE(vocab, config)
        model.load_state_dict(torch.load(model_path))
        trainer = VAETrainer(config)

    loader = trainer.get_dataloader(model, train_data, shuffle=False)
    
    model.eval()

    x_list = []
    z_list = []
    mu_list = []
    logvar_list = []
    y_list = []
    if model_type == 'vae_property':
        for step, batch in enumerate(loader):
            x = batch[0]
            x_list.extend(x)
                
            if model_type == 'vae_property':
                y = batch[1]
                y_list.extend(np.array(y).squeeze())

            mu, logvar, z, _ = model.forward_encoder(x)
            z_list.extend(z.detach().cpu().numpy())
            mu_list.extend(mu.detach().cpu().numpy())
            logvar_list.extend(logvar.detach().cpu().numpy())
            
    else:
        for step, batch in enumerate(loader):

            x_list.extend(batch)
            mu, logvar, z, _ = model.forward_encoder(batch)
            z_list.extend(z.detach().cpu().numpy())
            mu_list.extend(mu.detach().cpu().numpy())
            logvar_list.extend(logvar.detach().cpu().numpy())
            
    z_list = np.array(z_list)
    logvar_list = np.array(logvar_list)
    
    print("z_list shape", z_list.shape)
    print("logvar_list shape", logvar_list.shape)
    
    if model_type == 'vae_property':
        y_list = np.array(y_list)
        print("y_list shape", y_list.shape)
       
    return z_list, y_list, logvar_list, train_data, model


def z_to_smiles(model, origin, interpolated_z, data_type, steps, temp=1.0, argmax=True):
    
    gen_mol = model.sample(len(interpolated_z), max_len=100, z=interpolated_z, temp=temp, test=argmax)
    
    values = []
    if data_type == 'selfies':
        
        values.append(origin[0])
        values.extend(gen_mol)
        values.append(origin[1])

        viz_df = pd.DataFrame({"SELFIES": values})
        smiles = []
        
        for selfies in viz_df['SELFIES']:
            smiles.append(sf.decoder(selfies))

        viz_df['RoMol'] = smiles
        viz_df['RoMol'] = viz_df['RoMol'].apply(Chem.MolFromSmiles)
        display(PandasTools.FrameToGridImage(viz_df, column='RoMol', legendsCol='SELFIES', molsPerRow=steps+2))
        
    else:
        values.append(origin[0])
        values.extend(gen_mol)
        values.append(origin[1])
        
        viz_df = pd.DataFrame({"SMILES": values})
        viz_df['RoMol'] = viz_df['SMILES'].apply(Chem.MolFromSmiles)
        display(PandasTools.FrameToGridImage(viz_df, column='RoMol', legendsCol='SMILES', molsPerRow=steps+2))
        
    return viz_df
