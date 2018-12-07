from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os
import torch
import numpy as np

def save_images_batch(images_batch, file_name, h=28, w=28):
    images_batch = images_batch.view(-1, 1, h, w)
    grid = make_grid(images_batch, nrow=10)
    plt.imsave(file_name, np.transpose(grid.detach().numpy(), (1, 2, 0)))

def save_model(models_path, model_name, encoder, context_to_latent_dist, decoder, device):
    file_path = os.path.join(models_path, model_name)
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    model_states_dict = {"encoder": encoder.cpu().state_dict(),
                         "context_to_latent_dist": context_to_latent_dist.cpu().state_dict(),
                         "decoder": decoder.cpu().state_dict()}
    torch.save(model_states_dict, file_path)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    context_to_latent_dist = context_to_latent_dist.to(device)
    print('Saved state dicts to {}'.format(file_path))


def load_models(file_path, encoder, context_to_latent_dist, decoder):
    dict = torch.load(file_path)
    encoder.load_state_dict(dict["encoder"])
    context_to_latent_dist.load_state_dict(dict["context_to_latent_dist"])
    decoder.load_state_dict(dict["decoder"])

def make_mesh_grid(h,w):
    xs = np.linspace(0, 1, h)
    ys = np.linspace(0, 1, w)
    xx, yy = np.meshgrid(xs, ys)
    grid = torch.tensor(np.stack([xx, yy], axis=-1)).float()
    return grid

def random_mask_uniform(batch_shape,device, h=28, w=28):
    '''

    :param batch:
    :param grid:
    :param h:
    :param w:
    :return: encoder_input size (bsize,784,3) , mask size (bsize,784)
    '''
    # batch bsize * 1 * 28 * 28
    batch_size = batch_shape[0]

    ps = torch.rand(batch_size, device=device).unsqueeze(1).expand(batch_size, h * w)
    mask = torch.rand(batch_shape, device=device)
    mask = (mask >= ps).float()  # bsize * 784
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
    mu, var = z_params
    var = var.exp()
    sample = torch.randn(mu.shape).to(mu.device)
    z = mu + (torch.sqrt(var) * sample)
    return z