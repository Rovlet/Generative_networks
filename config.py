from dataclasses import dataclass

from rich import print


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

    def __post_init__(self) -> None:
        self.print_info()

    def print_info(self) -> None:
        print(self)
