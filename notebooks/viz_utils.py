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
from moses.metrics import QED, SA, logP
from rdkit.Chem import PandasTools

import gpytorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ProbabilityOfImprovement
from botorch.optim import optimize_acqf

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt


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


def ready_gpr(sample_num, data_type='smiles', model_name='vae_property_obj_w0.1'):
    
    train_df = pd.read_csv("../checkpoints/opimize_gpr/gpr_fit_ZINC250K_df.csv")[:sample_num]
    test_df = pd.read_csv("../checkpoints/opimize_gpr/gpr_test_ZINC250K_df.csv")
    start_df = pd.read_csv("../checkpoints/opimize_gpr/opt_start_ZINC250K_df.csv")
    
    print(f'gpr train: {train_df.shape}')
    print(f'gpr test: {test_df.shape}')
    print(f'gpr start: {start_df.shape}')
    
    # Choose model
    folder_path = f"../checkpoints/ZINC250K_{model_name}_{data_type}"
    
    if 'property' in model_name:
        config = torch.load(f'{folder_path}/vae_property_config.pt')
        vocab = torch.load(f'{folder_path}/vae_property_vocab.pt')
    else:
        config = torch.load(f'{folder_path}/vae_config.pt')
        vocab = torch.load(f'{folder_path}/vae_vocab.pt')
        config.reg_prop_tasks = ['obj']
    
    print(f"Use Selfies: {config.use_selfies}")
    print(config.reg_prop_tasks)
    
    cols = ['SELFIES' if config.use_selfies else 'SMILES', 'logP', 'qed', 'SAS', 'obj']
    
    train_data = train_df[cols].values
    test_data = test_df[cols].values
    
    if 'property' in model_name:
        model_path = f'{folder_path}/vae_property_model.pt'
        model = VAEPROPERTY(vocab, config)
        model.load_state_dict(torch.load(model_path))
        trainer = VAEPROPERTYTrainer(config)
        
    else:
        model_path = f'{folder_path}/vae_model_060.pt'
        model = VAE(vocab, config)
        model.load_state_dict(torch.load(model_path))
        trainer = VAETrainer(config)
    
    train_loader = trainer.get_dataloader(model, train_data, shuffle=False)
    test_loader = trainer.get_dataloader(model, test_data, shuffle=False)
    
    model.eval()

    x_list = []
    z_list = []
    mu_list = []
    logvar_list = []
    y_list = []

    for step, batch in enumerate(train_loader):
        x = batch[0]
        y = batch[1]
        x_list.extend(x)
        y_list.extend(np.array(y).squeeze())

        mu, logvar, z, _ = model.forward_encoder(x)
        z_list.extend(z.detach().cpu().numpy())
        mu_list.extend(mu.detach().cpu().numpy())
        logvar_list.extend(logvar.detach().cpu().numpy())



    y_list = np.array(y_list).squeeze()
    GP_train_y = y_list.reshape(-1, y_list.shape[-1])

    train_data_df = pd.DataFrame(GP_train_y, columns=['logP', 'qed', 'SAS', 'obj'])
    train_data_df = pd.concat([train_data_df , pd.DataFrame({'z': z_list, 'mu': mu_list, 'logvar': logvar_list})], axis=1)
    train_data_df.insert(0, 'SELFIES' if config.use_selfies else 'SMILES', [vocab.ids2string(point.cpu().detach().numpy()) for point in x_list])

    test_x_list = []
    test_z_list = []
    test_mu_list = []
    test_logvar_list = []
    test_y_list = []


    # y_list = y_list.squeeze()

    for step, batch in enumerate(test_loader):
        x = batch[0]
        y = batch[1]
        test_x_list.extend(x)
        test_y_list.extend(np.array(y).squeeze())

        mu, logvar, z, _ = model.forward_encoder(x)
        test_z_list.extend(z.detach().cpu().numpy())
        test_mu_list.extend(mu.detach().cpu().numpy())
        test_logvar_list.extend(logvar.detach().cpu().numpy())


    test_y_list = np.array(test_y_list).squeeze()
    GP_test_y = test_y_list.reshape(-1, test_y_list.shape[-1])

    test_data_df = pd.DataFrame(GP_test_y, columns=['logP', 'qed', 'SAS', 'obj'])
    test_data_df = pd.concat([test_data_df , pd.DataFrame({'z': test_z_list, 'mu': test_mu_list, 'logvar': test_logvar_list})], axis=1)
    test_data_df
    # test_data_df.insert(0, 'SELFIES' if config.use_selfies else 'SMILES', [vocab.ids2string(point.cpu().detach().numpy()) for point in test_x_list])
    
    GP_Train_x = torch.tensor(np.array([x for x in train_data_df['z']]))
    GP_Test_x = torch.tensor(np.array([x for x in test_data_df['z']]))

    GP_Train_y = np.array([x for x in train_data_df['obj']])
    GP_Test_y = np.array([x for x in test_data_df['obj']])
    
    
    return GP_Train_x, GP_Train_y, GP_Test_x, GP_Test_y, train_data_df, test_data_df, model, vocab, config
    
    
def generate_df(GP_Train_x, index_list, model, config, nan_qed, nan_sa, temp=1.0, test=True):

    gen = model.sample(len(GP_Train_x), max_len=100, z=torch.tensor(GP_Train_x), temp=temp, test=test)
    gen_df = pd.DataFrame(gen, columns=['gen_SELFIES' if config.use_selfies else 'gen_SMILES'])

    if config.use_selfies:
        gen_df['gen_SMILES'] = [sf.decoder(x) for x in gen_df['gen_SELFIES']]
        mol = gen_df['gen_SMILES'].apply(Chem.MolFromSmiles)
    else:
        mol = gen_df['gen_SMILES'].apply(Chem.MolFromSmiles)

    qed_list = []
    sa_list = []
    null_cnt = 0

    for i, gen_mol in enumerate(mol):
        if gen_mol is None:
            qed_list.append(nan_qed)
            sa_list.append(nan_sa)
            null_cnt += 1
            
        else:
            qed = QED(gen_mol)
            sa = SA(gen_mol)
            qed_list.append(qed)
            sa_list.append(sa)
            
    gen_df['gen_qed'] = qed_list
    gen_df['gen_sa'] = sa_list
    gen_df['obj'] = 5*gen_df['gen_qed'] - gen_df['gen_sa']
    gen_df['iter'] = index_list
    gen_df['z_value'] = [z_save for z_save in GP_Train_x]
    
    
    print(f"Null SMILES: {null_cnt}")
    print(f"# of Unique smiles", len(gen_df.gen_SMILES.unique()))
    
    return gen_df


def calculate_qed_sa(model, config, nan_qed, nan_sa, z, temp, test=True):
    gen = model.sample(len(z), max_len=100, z=z, temp=temp, test=test)
    gen_df = pd.DataFrame(gen, columns=['gen_SELFIES' if config.use_selfies else 'gen_SMILES'])

    if config.use_selfies:
        gen_df['gen_SMILES'] = [sf.decoder(x) for x in gen_df['gen_SELFIES']]
        mol = gen_df['gen_SMILES'].apply(Chem.MolFromSmiles)
    else:
        mol = gen_df['gen_SMILES'].apply(Chem.MolFromSmiles)

    qed_list = []
    sa_list = []

    for i, gen_mol in enumerate(mol):
        if gen_mol is None:
            qed_list.append(nan_qed)
            sa_list.append(nan_sa)
            
        else:
            qed = QED(gen_mol)
            sa = SA(gen_mol)
            qed_list.append(qed)
            sa_list.append(sa)
            
    gen_df['gen_qed'] = qed_list
    gen_df['gen_sa'] = sa_list

    return gen_df['gen_qed'].values, gen_df['gen_sa'].values

# 목적 함수 정의
def objective_function(model, config, nan_qed, nan_sa, new_z, temp, test):
    qed, sa = calculate_qed_sa(model, config, nan_qed, nan_sa, new_z, temp, test)
    return 5 * qed - sa

# 초기 데이터 수집
def initial_data(GP_Train_x, GP_Test_y):
    train_z = GP_Train_x
    train_y = torch.tensor(GP_Test_y)
    
    return train_z, train_y.unsqueeze(-1)


# GPR 모델 학습
def train_gp(train_z, train_y):
    gp = SingleTaskGP(train_z, train_y)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)
    
    return gp

# 획득 함수 최적화
def optimize_acq(gp, bounds, train_y):
    # acqf = ExpectedImprovement(gp, best_f=train_y.max().item())
    acqf = ProbabilityOfImprovement(gp, best_f=train_y.max().item())
    new_z, _ = optimize_acqf(acqf, bounds=bounds, q=1, num_restarts=5, raw_samples=20)
    
    return new_z


def reshape_z(z):
    return np.array(z).squeeze()


def vizualizeMol(gen_df, data_type):
    
    gen_df['RoMol'] = gen_df['gen_SMILES'].apply(Chem.MolFromSmiles)

    if data_type == 'selfies':
        display(PandasTools.FrameToGridImage(gen_df, column='RoMol', legendsCol='gen_SELFIES', molsPerRow=len(gen_df)))
    else:
        display(PandasTools.FrameToGridImage(gen_df, column='RoMol', legendsCol='gen_SMILES', molsPerRow=len(gen_df)))
        
        
        
def viz_latent_with_optim(df, z_df, all_df, data_type, model_type, base_pca='mu'):
    
    if model_type == 'vae_property':
        if data_type == 'selfies':
            folder_path = "../checkpoints/ZINC250K_vae_property_obj_proploss_w0.1_selfies"
        else:
            folder_path = "../checkpoints/ZINC250K_vae_property_obj_proploss_w0.1_smiles"
            
        config = torch.load(f'{folder_path}/vae_property_config.pt')
        vocab = torch.load(f'{folder_path}/vae_property_vocab.pt')
            
    elif model_type == 'vae':
        if data_type == 'selfies':
            folder_path = "../checkpoints/ZINC250K_vae_selfies"
        else:
            folder_path = "../checkpoints/ZINC250K_vae_smiles"
            
        config = torch.load(f'{folder_path}/vae_config.pt')
        vocab = torch.load(f'{folder_path}/vae_vocab.pt')

    
    if data_type == 'selfies':
        print(f"Use Selfies: {config.use_selfies}")
        print(config.reg_prop_tasks)

    cols = ['SELFIES' if config.use_selfies else 'SMILES', 'logP', 'qed', 'SAS', 'obj']
    data = df[cols].values

    if model_type == 'vae_property':
        model_path = f'{folder_path}/vae_property_model.pt'
        model = VAEPROPERTY(vocab, config)
        model.load_state_dict(torch.load(model_path))
        trainer = VAEPROPERTYTrainer(config)

    elif model_type == 'vae':
        model_path = f'{folder_path}/vae_model_060.pt'

        model = VAE(vocab, config)
        model.load_state_dict(torch.load(model_path))
        trainer = VAETrainer(config)

    train_loader = trainer.get_dataloader(model, data, shuffle=False)
    
    model.eval()
    
    x_list = []
    z_list = []
    mu_list = []
    logvar_list = []
    y_list = []
    
    for step, batch in enumerate(train_loader):
        x = batch[0]
        y = batch[1]
        x_list.extend(x)
        y_list.extend(np.array(y).squeeze())

        mu, logvar, z, _ = model.forward_encoder(x)
        z_list.extend(z.detach().cpu().numpy())
        mu_list.extend(mu.detach().cpu().numpy())
        logvar_list.extend(logvar.detach().cpu().numpy())
    
    
    plt.figure(figsize=(12, 10))
    viz = PCA(n_components=2)
    if base_pca == 'mu':
        z_viz = viz.fit_transform(mu_list) # ZINC250k train data
    else:
        z_viz = viz.fit_transform(z_list) # ZINC250k train data
    
    y_list = np.array(y_list)[:, -1]
    
    minmaxscale = MinMaxScaler()
    z_viz = minmaxscale.fit_transform(z_viz)
    
    all_z_str = np.array([z.strip('[]').split() for z in z_df['z_value']])
    all_z = np.array([np.array(num, dtype=np.float32) for num in all_z_str])
    
    str_z = np.array([z.strip('[]').split() for z in z_df['z_value']])
    z_val = np.array([np.array(num, dtype=np.float32) for num in str_z])
    
    qed = z_df['gen_qed'].values
    sa = z_df['gen_sa'].values
    obj = 5 * qed - sa
    
    if len(z_val) > 2:
        z_val = [z_val[1], z_val[-2]]
        obj_list = [obj[1], obj[-2]]
    else:
        z_val = [z_val[0], z_val[-1]]
        obj_list = [obj[0], obj[-1]]
    
    print("plot obj values:", obj_list)
            
    # PCA
    gen_z_viz = viz.transform(z_val) # start, end
    all_z = viz.transform(all_z) # every z values along iteration in optimization (200)
    z_list = viz.transform(z_list) # ZINC250k train data
    
    # scaling
    all_z_scaler = MinMaxScaler()
    all_z_scaler.fit(z_list)
    
    # gen_z_viz = minmaxscale.transform(gen_z_viz)
    gen_z_viz = all_z_scaler.transform(gen_z_viz)
    
    
    scatter = plt.scatter(z_viz[:, 0], z_viz[:, 1], c=y_list, cmap='viridis', marker='.', s=10, alpha=0.5, edgecolors='none')
    gen_scatter = plt.scatter(gen_z_viz[:, 0], gen_z_viz[:, 1], c=['black', 'darkred'], marker='x', s=100, alpha=1.0)
            
    for j, color in enumerate(['black', 'darkred']):
        label = "start" if color == "black" else "end"
        plt.text(gen_z_viz[j, 0]+0.05, gen_z_viz[j, 1], label, color=color, fontsize=12, ha='right')
        
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    plt.colorbar(scatter)
    plt.show()
    
    
# def plot_latent_vector()