from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import torch
import torchvision.utils


class Plots:
    plots_dir: str = 'plots/'
    figsize: Tuple[int] = (17, 17)
    num_interpolations: int = 10
    num_samples: int = 2920
    scatter_size: int = 800

    def plot_examples_of_generated_images_ae_outputs(self, vae, test_dataloader, device):
        plt.figure(figsize=self.figsize)
        images, _ = next(iter(test_dataloader))
        reconstruced_images = self.reconstruct_images(images, vae, device).cpu()

        # Reconstruct and visualise the images using the vae
        plt.figure(figsize=self.figsize)
        plt.imshow(torchvision.utils.make_grid(reconstruced_images[1:50], 10, 5).permute(1, 2, 0))
        plt.title("Some VAE reconstructions")
        plt.axis('off')
        plt.savefig(f'{self.plots_dir}vae_reconstruction.png')

    def plot_latent_params(self, approach, device, test_batch_x, test_batch_y) -> None:
        df = pd.DataFrame()
        df = approach.run_on_one_batch(df, test_batch_x, test_batch_y, device)
        df = df.set_index(['index'])

        size_exactly_as_std = False
        if size_exactly_as_std:
            size_max = 200
        else:
            size_max = None

        scatter = px.scatter(
            df.loc[df['image_ind'] < self.num_samples],
            x="mu_x", y="mu_y",
            animation_frame="epoch", animation_group="image_ind",
            size="std_x",
            color="class",
            hover_name="image_ind",
            color_discrete_sequence=px.colors.qualitative.Plotly,
            width=self.scatter_size,
            height=self.scatter_size,
            size_max=size_max,
            range_x=[-5, 5],
            range_y=[-5, 5])

        scatter.write_html(f'{self.plots_dir}latent_params.html')

    def plot_loss(self, losses):
        plt.plot(list(range(len(losses))), losses, label='train looses')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(f'{self.plots_dir}loss.png')

    def plot_latent_space(self, approach, device):
        nd = torch.distributions.Normal(loc=torch.as_tensor([0.]),
                                        scale=torch.as_tensor([1.]))

        with torch.no_grad():
            # create a sample grid in 2d latent space
            latent_interpolation = torch.linspace(0.001, 0.999, self.num_interpolations)
            latent_grid = torch.stack(
                (
                    latent_interpolation.repeat(self.num_interpolations, 1),
                    latent_interpolation[:, None].repeat(1, self.num_interpolations)
                ), dim=-1).view(-1, 2)

            # Without this, images would be distorted
            latent_grid = nd.icdf(latent_grid)

            # reconstruct images from the latent vectors
            latent_grid = latent_grid.to(device)
            image_recon = approach.model.decoder(latent_grid)
            image_recon = image_recon.cpu()

            plt.figure(figsize=self.figsize)
            plt.imshow(torchvision.utils.make_grid(image_recon.data[:self.num_interpolations ** 2],
                                                   self.num_interpolations).permute(1, 2, 0))
            plt.title("2D latent space")
            plt.axis('off')
            plt.savefig(f'{self.plots_dir}latent_space_2d.png')

            # plot image with latent interpolation 0.001
            plt.figure(figsize=self.figsize)
            plt.imshow(image_recon[0].permute(1, 2, 0), cmap='gray')
            plt.title("2D latent space")
            plt.axis('off')
            plt.savefig(f'{self.plots_dir}latent_space_2d_interpolation_001.png')

    def reconstruct_images(self, images, model, device):
        model.eval()
        with torch.no_grad():
            images, _, _ = model(images.to(device))
            images = images.clamp(0, 1)
            return images
