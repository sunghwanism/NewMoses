# NewMoses


## Dependency
torch <= 1.13

## Environment setup
create and activate conda environment named ```moses``` with ```python=3.8```
```sh
conda create -n moses python=3.8 -y \
conda activate moses \
pip install -r requirements.txt \
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

if you use MacBook (or DO NOT USE cuda), use this code
```sh
conda create -n moses python=3.8 -y \
conda activate moses \
pip install -r requirements.txt \
pip install torch==1.12.0+cpu torchvision==0.13.0+cpu torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cpu
```

## Running the benchmark data and model
- Only train one model on each process b/c of wandb tracking
- run the below code for training the model
- You can use benchmark model: aae, char_rnn, latentgan, organ and vae
- You can use benchmark dataset: QM9, ZINC
- If you use cuda, add '--device cuda:{device_idx}', else --device cpu
- If you want to train model with selfies format, you add the '--use_selfies 1' when you run the scripts/run.py
    - if you don't add it, the model is trained by smiles format automatically
    - (!!Caution!!) if you use --use_selfies with any kind of format (ex: 0,1, ..., True, False...), the model is trained by selfies

Example
'''sh
python scripts/run.py --device cuda:0 —model vae --use_selfies 1 --n_batch 2048
'''sh


## Adding the Dataset
If you train model using your model, add the splited dataset named train.csv, test.csv in moses > dataset > data > {datasetname} > files
For example, we have already make the directory for ZINC and QM9 dataset


## Reference code
We re-generate the code from https://github.com/molecularsets/moses for our project.


