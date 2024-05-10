# NewMoses


# Dependency
torch <= 1.13

## Environment setup
create and activate conda environment named ```moses``` with ```python=3.8```
```sh
conda create -n moses python=3.8 -y
conda activate moses
pip install -r requirements.txt
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

if you use MacBook (or DO NOT USE cuda), use this code
```sh
conda create -n moses python=3.8 -y
conda activate moses
pip install -r requirements.txt
pip install torch==1.12.0+cpu torchvision==0.13.0+cpu torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cpu
```