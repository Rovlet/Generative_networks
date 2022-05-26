from torch import nn
import torch


class Encoder(nn.Module):
    def __init__(self, hidden_channels: int, latent_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=hidden_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1)  # out: hidden_channels x 14 x 14

        self.conv2 = nn.Conv2d(in_channels=hidden_channels,
                               out_channels=hidden_channels * 2,
                               kernel_size=4,
                               stride=2,
                               padding=1)  # out: (hidden_channels x 2) x 7 x 7

        self.fc_mu = nn.Linear(in_features=hidden_channels * 2 * 7 * 7,
                               out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=hidden_channels * 2 * 7 * 7,
                                   out_features=latent_dim)

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        :param x: batch of images with shape [batch, channels, w, h]
        :returns: the predicted mean and log(variance)
        """
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))

        x = x.view(x.shape[0], -1)

        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)

        return x_mu, x_logvar


class Decoder(nn.Module):
    def __init__(self, hidden_channels: int, latent_dim: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels

        self.fc = nn.Linear(in_features=latent_dim,
                            out_features=hidden_channels * 2 * 7 * 7)

        self.conv2 = nn.ConvTranspose2d(in_channels=hidden_channels * 2,
                                        out_channels=hidden_channels,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=hidden_channels,
                                        out_channels=1,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1)

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: a sample from the distribution governed by the mean and log(var)
        :returns: a reconstructed image with size [batch, 1, w, h]
        """
        x = self.fc(x)
        x = x.view(x.size(0), self.hidden_channels * 2, 7, 7)
        x = self.activation(self.conv2(x))
        x = torch.sigmoid(self.conv1(x))
        return x


class VariationalAutoencoder(nn.Module):
    def __init__(self, hidden_channels: int, latent_dim: int):
        super().__init__()
        self.encoder = Encoder(hidden_channels=hidden_channels, latent_dim=latent_dim)
        self.decoder = Decoder(hidden_channels=hidden_channels, latent_dim=latent_dim)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Each data point in a VAE would get mapped to mean and log_variance vectors which would define the multivariate
        normal distribution around that input data point.
        A point is sampled from this distribution and is returned as the latent variable.
        This latent variable is fed to the decoder to produce the output.
        """

        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = (logvar * 0.5).exp()
            return torch.distributions.Normal(loc=mu, scale=std).rsample()
        else:
            return mu

