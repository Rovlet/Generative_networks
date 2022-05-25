import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import functional as F
from torchvision.datasets import FashionMNIST
import pandas as pd
import torch
from tqdm import tqdm
from typing import Callable, Optional, List
from config import InputParams
from models.vae import VariationalAutoencoder
import numpy as np


class VAE:
    test_dataloader: DataLoader
    train_dataloader: DataLoader
    model: VariationalAutoencoder
    config: InputParams
    optimizer: Adam

    def __init__(self, device: torch.device, config: InputParams) -> None:
        self.config = config
        self.model = VariationalAutoencoder(hidden_channels=self.config.capacity, latent_dim=self.config.latent_dims)
        self.model = self.model.to(device)
        self.optimizer = Adam(params=self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-5)

    def load_data(self) -> None:
        img_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = FashionMNIST(root='./data/MNIST', train=True, transform=img_transform, download=True)
        self.train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        test_dataset = FashionMNIST(root='./data/MNIST', download=True, train=False, transform=img_transform)
        self.test_dataloader = DataLoader(test_dataset, batch_size=max(10000, self.config.batch_size), shuffle=True)

    def train(self, device: torch.device) -> List[float]:
        self.model.train()
        train_loss_avg = []
        tqdm_bar = tqdm(range(1, self.config.num_epochs + 1), desc="epoch [loss: ...]")
        for i, epoch in enumerate(tqdm_bar):
            train_loss_averager = self.make_averager()

            batch_bar = tqdm(self.train_dataloader, leave=False, desc='batch', total=len(self.train_dataloader))
            for image_batch, _ in batch_bar:
                image_batch = image_batch.to(device)

                # vae reconstruction
                image_batch_recon, latent_mu, latent_logvar = self.model(image_batch)

                # reconstruction error
                loss = self.vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar)

                # backpropagation
                self.optimizer.zero_grad()
                loss.backward()

                # one step of the optmizer
                self.optimizer.step()

                self.refresh_bar(batch_bar, f"Train batch [Loss: {train_loss_averager(loss.item()):.3f}]")

            self.refresh_bar(tqdm_bar, f"Epoch {i} [Average loss: {train_loss_averager(None):.3f}]")
            train_loss_avg.append(train_loss_averager(None))
        return train_loss_avg

    def evaluate(self,  device: torch.device) -> None:
        self.model.eval()
        test_loss_averager = self.make_averager()
        with torch.no_grad():
            test_bar = tqdm(self.test_dataloader, total=len(self.test_dataloader), desc='batch [loss: ...]')
            for image_batch, _ in test_bar:
                image_batch = image_batch.to(device)

                # vae reconstruction
                image_batch_recon, latent_mu, latent_logvar = self.model(image_batch)

                # reconstruction error
                loss = self.vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar)

                self.refresh_bar(test_bar, f"test batch [loss: {test_loss_averager(loss.item()):.3f}]")
        print(f'Average test loss: {test_loss_averager(None)})')

    def run_on_one_batch(self, df_log, x, y, device, epoch=0):
        with torch.no_grad():
            x = x.to(device)
            x, mus, stddevs = self.model(x)
            x = x.to('cpu')
            mus = mus.to('cpu').data.numpy()
            stddevs = stddevs.to('cpu').mul(0.5).exp_().data.numpy()
        return self.save_in_dataframe(df_log, y, mus, stddevs, epoch)

    def vae_loss(self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        recon_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
        kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.config.variational_beta * kldivergence

    @staticmethod
    def make_averager() -> Callable[[Optional[float]], float]:
        """ Returns a function that maintains a running average
        :returns: running average function
        """
        count = 0
        total = 0

        def averager(new_value: Optional[float]) -> float:
            """ Running averager

            :param new_value: number to add to the running average,
                              if None returns the current average
            :returns: the current average
            """
            nonlocal count, total
            if new_value is None:
                return total / count if count else float("nan")
            count += 1
            total += new_value
            return total / count

        return averager

    @staticmethod
    def save_in_dataframe(df_log, labels, mus, stddevs, epoch):
        df = pd.DataFrame()

        df['index'] = np.arange(len(mus[:, 0])) * epoch
        df['image_ind'] = np.arange(len(mus[:, 0]))
        df['class'] = labels.data.numpy().astype(str)
        df['mu_x'] = mus[:, 0]
        df['mu_y'] = mus[:, 1]
        df['std_x'] = stddevs[:, 0]
        df['std_y'] = stddevs[:, 1]
        df['epoch'] = np.ones(len(mus[:, 0])) * epoch

        df_log = pd.concat([df_log, df])
        return df_log

    @staticmethod
    def refresh_bar(bar: tqdm, desc: str) -> None:
        bar.set_description(desc)
        bar.refresh()
