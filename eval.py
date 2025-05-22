"""Script to evaluate a trained model on a set of images and ground truth density maps."""

import os
import click
import torch
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from model import UNet, FCRN_A, load_from_checkpoint
from vgg import VGG16_FCN

@click.command()
@click.option('-n', '--network_architecture',
              type=click.Choice(['UNet', 'FCRN_A']),
              required=True,
              help='Model architecture.')
@click.option('-c', '--checkpoint',
              type=click.Path(exists=True),
              required=True,
              help='Path to a checkpoint with weights.')
@click.option('--unet_filters', default=64,
              help='Number of filters for U-Net convolutional layers.')
@click.option('--convolutions', default=2,
              help='Number of layers in a convolutional block.')
@click.option('--one_channel',
              is_flag=True,
              help="Turn this on for one channel images (required for ucsd).")
@click.option('--pad',
              is_flag=True,
              help="Turn on padding for input images (required for ucsd).")
@click.option('-i', '--image_dir',
              type=click.Path(exists=True),
              required=True,
              help='Directory containing images for evaluation.')
@click.option('-g', '--label_dir',
              type=click.Path(exists=True),
              required=True,
              help='Directory containing ground truth density maps (as .npy files).')
def evaluate(
    network_architecture, 
    checkpoint, 
    unet_filters, 
    convolutions, 
    one_channel, 
    pad, 
    image_dir, 
    label_dir):
    """
    Evaluate the model on a set of images and ground truth density maps.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_channels = 1 if one_channel else 3

    # Initialize model
    network = {
        'UNet': UNet,
        'FCRN_A': FCRN_A,
        'VGG16_FCN': VGG16_FCN
    }[network_architecture](input_filters=input_channels,
                            filters=unet_filters,
                            N=convolutions).to(device)
    load_from_checkpoint(network, checkpoint)
    
    network.eval()

    # Collect image and gt paths
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    gt_files = sorted([f for f in os.listdir(label_dir) if f.lower().endswith('.npy')])

    assert len(image_files) == len(gt_files), "Number of images and ground truth maps must match."

    me, mae, mse = 0.0, 0.0, 0.0

    with torch.no_grad():
        for img_file, gt_file in zip(image_files, gt_files):
            img_path = os.path.join(image_dir, img_file)
            gt_path = os.path.join(label_dir, gt_file)

            img = Image.open(img_path)
            if pad:
                img = Image.fromarray(np.pad(np.array(img), 1, 'constant', constant_values=0))
            img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
            pred_dmap = network(img_tensor)
            pred_count = torch.sum(pred_dmap).item() / 100

            gt_dmap = np.load(gt_path)
            gt_count = np.sum(gt_dmap) / 100

            # Calculate metrics
            me += (pred_count - gt_count)
            mae += abs(pred_count - gt_count)
            mse += (pred_count - gt_count) ** 2

    me /= len(image_files)
    mae /= len(image_files)
    mse = (mse / len(image_files)) ** 0.5
    print(f"ME: {mae:.2f}, MAE: {mae:.2f}, MSE: {mse:.2f}")
    return me, mae, mse

if __name__ == "__main__":
    evaluate()