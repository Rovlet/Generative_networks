import torch
from config import InputParams
from plots import Plots
from vae import VAE

device = torch.device("cuda:0" if torch.cuda.is_available() and torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    config = InputParams()
    plots = Plots()
    print(f'Approach arguments = {config}')

    # Prepare model
    approach = VAE(device, config)

    # Load data
    approach.load_data()

    # Train model
    train_loss_avg = approach.train(device)

    # Plot loss function
    plots.plot_loss(train_loss_avg)

    # get one batch of data and plot latent params
    test_batch_x, test_batch_y = iter(approach.test_dataloader).next()
    plots.plot_latent_params(approach, test_batch_x, test_batch_y, device)

    # Evaluate model on test data
    approach.evaluate(device)

    # Plot examples of generated images
    plots.plot_ae_outputs(approach.model, approach.test_dataloader, device)

    # Plot latent space
    plots.plot_latent_space(approach, device)
