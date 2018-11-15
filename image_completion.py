import torch
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from NP import Decoder, ContextToLatentDistribution, ContextEncoder, load_models, sample_z
from argparse import ArgumentParser


def complete(context, pixels, decoder, h=28, w=28, batch_size=32):
    z = context.unsqueeze(1).expand(-1, h * w, -1)
    # resize context to have one context per input coordinate
    pixels_input = pixels.unsqueeze(0).expand(batch_size, -1, -1)
    target_input = torch.cat([z, pixels_input], dim=-1)
    return decoder.forward(target_input)


def complete_full(context, decoder, h=28, w=28):
    xs = np.linspace(0, 1, h)
    ys = np.linspace(0, 1, w)
    xx, yy = np.meshgrid(xs, ys)
    grid = torch.tensor(np.stack([xx, yy], axis=-1)).float().to(context.device).view(h * w, 2)  # size 784*2

    return complete(context, grid, decoder), grid


def constant_completion(batch, encoder, encoder_to_distrib, decoder, initial_pixels, h=28, w=28):
    '''
    Image completion is implemented by batch (we complete a batch of images at once!).
    '''

    # initial_pixels: n_initial_pixels * 2
    # import pdb;
    # pdb.set_trace()

    context_values = batch.gather(index=initial_pixels[:, :, 0].unsqueeze(2).expand(-1, -1, batch.size(2)),
                                  dim=1)
    import pdb;
    pdb.set_trace()
    context_values = context_values.gather(
        index=initial_pixels[:, :, 1].unsqueeze(1).expand(-1, context_values.size(2),-1), dim=1)

    # renormalize pixel coordinates
    coords = initial_pixels.float()
    coords[:, 0] /= (batch.size(1) - 1)
    coords[:, 1] /= (batch.size(2) - 1)

    # build encoder input
    import pdb;
    pdb.set_trace()
    context_data = torch.cat([context_values, coords], dim=-1)
    context = encoder(context_data)  # dimension: batch_size * len(coords) * hidden

    # build aggregated context distribution and sample from it.
    aggregated_context = context.mean(axis=1)  # dimension: batch_size * hidden (one context per image).
    z = sample_z(aggregated_context)

    image_probs = complete_full(decoder, context)  # get the image completion probabilities.
    completed_image = torch.bernoulli(image_probs)
    return completed_image


def build_image(images_batch):
    grid = make_grid(images_batch, nrow=10)
    plt.imshow(np.transpose(grid.detach().numpy(), (1, 2, 0)), interpolation='nearest')
    plt.show()


def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > .5).float())
        ])),
        batch_size=args.bsize, shuffle=True)

    context_encoder = ContextEncoder()
    context_to_dist = ContextToLatentDistribution()
    decoder = Decoder()

    if args.resume_file is not None:
        load_models(args.resume_file, context_encoder, context_to_dist, decoder)

    context_encoder = context_encoder.to(device)
    context_to_dist = context_to_dist.to(device)
    decoder = decoder.to(device)

    N = 500  # 500 random initial pixels.
    initial_pixels = torch.randint(0, 28, (args.bsize, N, 2)).long()
    first_batch, _ = next(iter(test_loader))
    first_batch = first_batch.squeeze(1)
    completed_images = constant_completion(first_batch, context_encoder, context_to_dist, decoder, initial_pixels)
    build_image(completed_images)


parser = ArgumentParser()
parser.add_argument("--resume_file", type=str, default=None)
parser.add_argument("--bsize", type=int, default=32)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
