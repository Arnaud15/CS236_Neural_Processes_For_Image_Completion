import torch
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import NP

def complete_full_image(decoder, context, grid):
    z = context.unsqueeze(1).expand(-1, h * w, -1)
    # resize context to have one context per input coordinate
    grid_input = grid.unsqueeze(0).expand(batch_size, -1, -1)
    target_input = torch.cat([z, grid_input], dim=-1)
    return decoder.forward(target_input)

def complete_pixel(decoder, context, pixel):
    z = context.unsqueeze(1).expand(-1, h * w, -1)
    target_input = torch.cat([z, pixel], dim=-1)
    return decoder.forward(target_input)

def constant_completion(batch, encoder, encoder_to_distrib, decoder, initial_pixels, h=28, w=28):
    '''
    Image completion is implemented by batch (we complete a batch of images at once!).
    '''

    xs = np.linspace(0, 1, h)
    ys = np.linspace(0, 1, w)
    xx, yy = np.meshgrid(xs, ys)
    grid = torch.tensor(np.stack([xx, yy], axis=-1)).float().to(device).view(h * w, 2)  # size 784*2

    # initial_pixels: n_initial_pixels * 2
    context_values = batch[:,initial_pixels]

    # renormalize pixel coordinates
    context_pixels[:,0] /= batch.size(1)
    context_pixels[:,1] /= batch.size(2)

    # build encoder input
    context_data = torch.cat([context_values, context_pixels], dim=-1)
    context = encoder(context_data) # dimension: batch_size * len(initial_pixels) * hidden

    # build aggregated context distribution and sample from it.
    aggregated_context = context.mean(axis=1) # dimension: batch_size * hidden (one context per image)
    z = NP.sample_z(aggregated_context)

    image_probs = complete_full_image(decoder, context, grid)
    completed_image = torch.bernoulli(image_probs)
    return completed_image

def main():

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    batch_size = 1

    context_encoder = torch.load("models/context_encoder").to(device)
    context_to_dist = torch.load("models/context_to_dist").to(device)
    decoder = torch.load("models/decoder").to(device)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)

    return

if __name__ == "main":
    main()
