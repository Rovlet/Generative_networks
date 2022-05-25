import torch
import pandas as pd
from config import InputParams
from plots import Plots
from vae import VAE

device = torch.device("cuda:0" if torch.cuda.is_available() and torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    config = InputParams()
    print(f'Approach arguments = {config}')
    approach = VAE(device, config)

    approach.load_data()

    # train model
    train_loss_avg = approach.train(device)
    Plots.plot_loss(train_loss_avg)

    # prepare data for plotting latent params after training
    test_batch_x, test_batch_y = iter(approach.test_dataloader).next()
    df_log = pd.DataFrame()
    df_log = approach.run_on_one_batch(df_log, test_batch_x, test_batch_y, device)
    df_log = df_log.set_index(['index'])
    Plots.plot_latent_params(df_log)

    # evaluate model
    approach.evaluate(device)

    # plot examples of generated images
    Plots.plot_ae_outputs(approach.model, approach.test_dataloader, device)

    # plot latent space
    Plots.plot_latent_space(approach, device)
