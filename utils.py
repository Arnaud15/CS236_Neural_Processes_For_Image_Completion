from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from math import pi as PI


def display_images(original_image, mask, reconstructed_image, h=28, w=28):
    '''

    :param original_image: bsize * c * h * w
    :param mask: bsize * h * w
    :param reconstructed_image: n_samples.bsize * h * w
    :param h:
    :param w:
    :return:
    '''
    bsize = original_image.size(0)
    original_image = original_image.view(bsize, -1, h, w).expand(-1, 3, -1, -1)
    mask = mask.view(bsize, 1, h, w).expand(-1, 3, -1, -1)
    masked_image = torch.zeros_like(original_image)
    masked_image[:, 2] = 1
    masked_image[mask == 1] = 0
    masked_image = masked_image + mask * original_image
    masked_image = torch.min(masked_image, torch.ones_like(masked_image))
    reconstructed_image = reconstructed_image.view(reconstructed_image.size(0), -1, h, w).expand(-1, 3, h, w)

    stacked = torch.cat([original_image, masked_image, reconstructed_image], dim=0)

    grid = make_grid(stacked, nrow=bsize)
    return grid.detach().cpu().numpy()


def display_images_CIFAR(original_image, image_mean, image_var, mask, h=32, w=32,
                         means_normalize=(0.4914, 0.4822, 0.4465), var_normalize=(0.2023, 0.1994, 0.2010)):
    bsize = original_image.size(0)
    mask = mask.view(bsize, 1, h, w).expand(-1, 3, -1, -1)
    masked_image = torch.zeros_like(original_image)
    masked_image[:, 2] = 1
    masked_image[mask == 1] = 0
    masked_image = masked_image + mask * original_image
    masked_image = torch.min(masked_image, torch.ones_like(masked_image))

    # TODO maybe normalize
    image_list = [original_image, masked_image]
    for i in range(0, image_mean.size(0), bsize):
        image_list = image_list + [image_mean[i:i + bsize]] + [image_var[i:i + bsize]]

    stacked = torch.cat(image_list, dim=0).cpu()
    ##TODO better
    # import pdb;
    # pdb.set_trace()
    grid = make_grid(stacked, nrow=bsize, normalize=True,range = (stacked.min().item(),stacked.max().item()))

    return grid.detach().cpu().numpy()


def save_images_batch(images_batch, file_name, h=28, w=28):
    images_batch = images_batch.view(-1, 1, h, w)
    grid = make_grid(images_batch, nrow=10)
    plt.imsave(file_name, np.transpose(grid.detach().numpy(), (1, 2, 0)))


def save_model(models_path, model_name, encoder, context_to_latent_dist, decoder,aggregator, device):
    file_path = os.path.join(models_path, model_name)
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    model_states_dict = {"encoder": encoder.cpu().state_dict(),
                         "context_to_latent_dist": context_to_latent_dist.cpu().state_dict(),
                         "decoder": decoder.cpu().state_dict(),
                         "aggregator":aggregator.cpu().state_dict()}
    torch.save(model_states_dict, file_path)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    context_to_latent_dist = context_to_latent_dist.to(device)
    aggregator = aggregator.to(device)
    print('Saved state dicts to {}'.format(file_path))


def load_models(file_path, encoder, context_to_latent_dist, decoder,aggregator):
    dict = torch.load(file_path)
    encoder.load_state_dict(dict["encoder"])
    context_to_latent_dist.load_state_dict(dict["context_to_latent_dist"])
    decoder.load_state_dict(dict["decoder"])
    # for compatibility
    if "aggregator" in dict:
        aggregator.load_state_dict(dict["aggregator"])


def make_mesh_grid(h, w):
    xs = np.linspace(0, 1, h)
    ys = np.linspace(0, 1, w)
    xx, yy = np.meshgrid(xs, ys)
    grid = torch.tensor(np.stack([xx, yy], axis=-1)).float()
    return grid


def random_mask_uniform(bsize, h, w, device):
    '''

    :param batch:
    :param grid:
    :param h:
    :param w:
    :return: encoder_input size (bsize,784,3) , mask size (bsize,784)
    '''

    ps = torch.rand(bsize, device=device).unsqueeze(1).expand(bsize, h * w)
    mask = torch.rand((bsize, h * w), device=device)
    mask = (mask >= ps).float()  # bsize * 784
    mask = mask.view(bsize, h, w)
    return mask


def random_mask(bsize, h, w, n_pixels_to_keep, device):
    a = np.array([np.random.choice(h * w, n_pixels_to_keep, replace=False) for _ in range(bsize)])
    mask = np.zeros((bsize, h * w))
    for i in range(bsize):
        mask[i, a[i, :]] = 1
    mask = torch.tensor(mask).float().to(device).view(bsize, h, w)
    return mask


def kl_normal(params_p, params_q):
    mu_p, logvar_p = params_p
    var_p = logvar_p.exp()
    mu_q, logvar_q = params_q
    var_q = logvar_q.exp()
    element_wise = 0.5 * (torch.log(var_q) - torch.log(var_p) + var_p / var_q + (mu_p - mu_q).pow(2) / var_q - 1)
    kl = element_wise.sum(-1)
    return kl


def sample_z(z_params):
    mu, logvar = z_params
    var = logvar.exp()
    sample = torch.randn(mu.shape).to(mu.device)
    z = mu + (torch.sqrt(var) * sample)
    return z


def log_normal(x, m, v):
    """
    Computes the elem-wise log probability of a Gaussian and then sum over the
    last dim. Basically we're assuming all dims are batch dims except for the
    last dim.

    Args:
        x: tensor: (batch, ..., dim): Observation
        m: tensor: (batch, ..., dim): Mean
        v: tensor: (batch, ..., dim): Variance
    """
    log_prob = (-0.5 * (x - m).pow(2) / v - 0.5 * torch.log(2 * PI * v)).sum(dim=-1)
    return log_prob
