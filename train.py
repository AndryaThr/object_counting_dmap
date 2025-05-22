"""Main script used to train networks."""
import os
from typing import Union, Optional, List

import click
import torch
import numpy as np
from matplotlib import pyplot

from data_loader import H5Dataset
from looper import Looper
from model import UNet, FCRN_A
from vgg import VGG16_FCN

@click.command()
@click.option('-d', '--dataset_name',
              type=click.Choice(['cell', 'mall', 'ucsd']),
              required=True,
              help='Dataset to train model on (expect proper HDF5 files).')
@click.option('-n', '--network_architecture',
              type=click.Choice(['UNet', 'FCRN_A', "VGG16_FCN"]),
              required=True,
              help='Model to train.')
@click.option('-lr', '--learning_rate', default=1e-2,
              help='Initial learning rate (lr_scheduler is applied).')
@click.option('-e', '--epochs', default=150, help='Number of training epochs.')
@click.option('--batch_size', default=8,
              help='Batch size for both training and validation dataloaders.')
@click.option('-hf', '--horizontal_flip', default=0.0,
              help='The probability of horizontal flip for training dataset.')
@click.option('-vf', '--vertical_flip', default=0.0,
              help='The probability of horizontal flip for validation dataset.')
@click.option('--unet_filters', default=64,
              help='Number of filters for U-Net convolutional layers.')
@click.option('--convolutions', default=2,
              help='Number of layers in a convolutional block.')
@click.option('--plot', is_flag=True, help="Generate a live plot.")
@click.option('-o', '--output_path', default=None, help="Output folder for model checkpoints.")
def train(dataset_name: str,
          network_architecture: str,
          learning_rate: float,
          epochs: int,
          batch_size: int,
          horizontal_flip: float,
          vertical_flip: float,
          unet_filters: int,
          convolutions: int,
          plot: bool,
          output_path: Optional[str]):
    """Train chosen model on selected dataset."""
    # use GPU if avilable
    opth = os.path.abspath('.') if output_path is None else os.path.abspath(output_path)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = {}     # training and validation HDF5-based datasets
    dataloader = {}  # training and validation dataloaders

    for mode in ['train', 'valid']:
        # expected HDF5 files in dataset_name/(train | valid).h5
        data_path = os.path.join(dataset_name, f"{mode}.h5")
        # turn on flips only for training dataset
        dataset[mode] = H5Dataset(data_path,
                                  horizontal_flip if mode == 'train' else 0,
                                  vertical_flip if mode == 'train' else 0)
        dataloader[mode] = torch.utils.data.DataLoader(dataset[mode],
                                                       batch_size=batch_size)

    # only UCSD dataset provides greyscale images instead of RGB
    input_channels = 1 if dataset_name == 'ucsd' else 3

    # initialize a model based on chosen network_architecture
    network = {
        'UNet': UNet,
        'FCRN_A': FCRN_A,
        'VGG16_FCN': VGG16_FCN
    }[network_architecture](input_filters=input_channels,
                            filters=unet_filters,
                            N=convolutions).to(device)

    # initialize loss, optimized and learning rate scheduler
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(network.parameters(),
                                lr=learning_rate,
                                momentum=0.9,
                                weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=20,
                                                   gamma=0.1)

    # if plot flag is on, create a live plot (to be updated by Looper)
    if plot:
        pyplot.ion()
        fig, plots = pyplot.subplots(nrows=2, ncols=2)
    else:
        plots = [None] * 2

    # create training and validation Loopers to handle a single epoch
    train_looper = Looper(network, device, loss, optimizer,
                          dataloader['train'], len(dataset['train']), plots[0])
    valid_looper = Looper(network, device, loss, optimizer,
                          dataloader['valid'], len(dataset['valid']), plots[1],
                          validation=True)

    # current best results (lowest mean absolute error on validation set)
    current_best = np.inf
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n")

        # run training epoch and update learning rate
        train_looper.run()
        lr_scheduler.step()

        # run validation epoch
        with torch.no_grad():
            result = valid_looper.run()

        # update checkpoint if new best is reached
        if result < current_best:
            current_best = result
            
            print(f"[!] New best result: {result}")
        
    
        save_path = os.path.join(opth, f"model_{network_architecture}_{(epoch + 1):03d}.pth")    
        torch.save({
            'epoch': epoch + 1, 
            'model_state_dict': network.state_dict(),
            'optimizer': optimizer.state_dict(), 
            'scheduler': lr_scheduler.state_dict(),
        }, save_path)

        print("\n", "-"*80, "\n", sep='')

    print(f"[Training done] Best result: {current_best}")

if __name__ == '__main__':
    train()

