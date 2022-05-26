# Vae task

This version uses Python 3.8 with cuda 11.5

## Install
You can change the first line below if you want to use a different cuda version.

```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu115
pip3 install -r requirements.txt
```

## Usage
To run application, use the following command:

```
python3 main.py
```

Application steps:
1. Load input parameters
2. Prepare model
3. Load data
4. Train model
5. Plot results
6. Evaluate model

## Settings
There is a config file with default settings stored in Dataclass InputParams. 

```
@dataclass(frozen=True)
class InputParams:
    latent_dims: int = 2
    num_epochs: int = 40
    batch_size: int = 128
    capacity: int = 64
    learning_rate: float = 1e-3
    variational_beta: int = 1
    use_gpu: bool = True
    latents_lims: float = 3.66
```
