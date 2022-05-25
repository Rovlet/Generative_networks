from dataclasses import dataclass


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
