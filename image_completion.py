import torch
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import NP

def complete(context, pixels, decoder):
    z = context.unsqueeze(1).expand(-1, h * w, -1)
    # resize context to have one context per input coordinate
    pixels_input = pixels.unsqueeze(0).expand(batch_size, -1, -1)
    target_input = torch.cat([z, pixels_input], dim=-1)
    return decoder.forward(target_input)

def complete_full(context, decoder):

    xs = np.linspace(0, 1, h)
    ys = np.linspace(0, 1, w)
    xx, yy = np.meshgrid(xs, ys)
    grid = torch.tensor(np.stack([xx, yy], axis=-1)).float().to(device).view(h * w, 2)  # size 784*2

    return complete(context, grid, decoder)

def constant_completion(batch, encoder, encoder_to_distrib, decoder, initial_pixels, h=28, w=28):
    '''
    Image completion is implemented by batch (we complete a batch of images at once!).
    '''

    # initial_pixels: n_initial_pixels * 2
    context_values = batch[:,initial_pixels[:,0],initial_pixels[:,1]]

    # renormalize pixel coordinates
    context_pixels[:,0] /= (batch.size(1) - 1)
    context_pixels[:,1] /= (batch.size(2) - 1)

    # build encoder input
    context_data = torch.cat([context_values, context_pixels], dim=-1)
    context = encoder(context_data) # dimension: batch_size * len(initial_pixels) * hidden

    # build aggregated context distribution and sample from it.
    aggregated_context = context.mean(axis=1) # dimension: batch_size * hidden (one context per image).
    z = NP.sample_z(aggregated_context)

    image_probs = complete_full_image(decoder, context, grid) # get the image completion probabilities.
    completed_image = torch.bernoulli(image_probs)
    return completed_image

def build_image(images_batch):
    grid = make_grid(images_batch, nrow=n_rows)
    plt.imshow(np.transpose(grid.detach().numpy(), (1,2,0)), interpolation='nearest')
    plt.show()

def main():

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    batch_size = 32 # Complete 32 images at once.

    context_encoder = torch.load("models/context_encoder").to(device)
    context_to_dist = torch.load("models/context_to_dist").to(device)
    decoder = torch.load("models/decoder").to(device)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > .5).float())
        ])),
        batch_size=batch_size, shuffle=True)

    N = 50 # 50 random initial pixels.
    initial_pixels = torch.randint(0, 28, (batch_size, N, 2))
    first_batch = test_loader.next()

    completed_images = constant_completion(first_batch, encoder, encoder_to_distrib, decoder, initial_pixels)
    build_image(completed_images)

if __name__ == "__main__":
    main()
