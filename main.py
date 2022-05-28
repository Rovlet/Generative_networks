from typing import List, Tuple

import torch

from config import InputParams
from plots import Plots
from vae import VAE


class MainRoutine:
    def __init__(self, device: torch.device, vae_approach: VAE):
        self.device: torch.device = device
        self.vae_approach: VAE = vae_approach
        self.plots: Plots = Plots()

    def run(self) -> None:
        self.vae_approach.load_data()
        self.plots.plot_loss(self._train_and_get_train_loss())
        self.plots.plot_latent_params(self.vae_approach, self.device, *self._get_one_batch_of_data())
        self.vae_approach.evaluate_model_on_test_data(self.device)
        self.plots.plot_examples_of_generated_images_ae_outputs(self.vae_approach.model,
                                                                self.vae_approach.test_dataloader, self.device)
        self.plots.plot_latent_space(self.vae_approach, self.device)

    def _train_and_get_train_loss(self) -> List[float]:
        return self.vae_approach.train(self.device)

    def _get_one_batch_of_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        test_batch_x, test_batch_y = iter(self.vae_approach.test_dataloader).next()
        return test_batch_x, test_batch_y


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main_routine = MainRoutine(
        device=device,
        vae_approach=VAE(device=device, config=InputParams())
    )
    main_routine.run()


if __name__ == '__main__':
    main()
