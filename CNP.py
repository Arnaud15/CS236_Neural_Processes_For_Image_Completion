import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import os
import time


class ContextEncoder(nn.Module):
    def __init__(self):
        super(ContextEncoder, self).__init__()
        self.layer1 = nn.Linear(3, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 64)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class TargetNetwork(nn.Module):
    def __init__(self):
        super(TargetNetwork, self).__init__()
        self.layer1 = nn.Linear(64 + 2, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x  # out0 = mu, out1 = log( sigma 2) for input context + x1, x2


def random_sampling(batch, grid, h=28, w=28):
    '''

    :param batch:
    :param grid:
    :param h:
    :param w:
    :return: encoder_input size (bsize,784,3) , mask size (bsize,784)
    '''
    # batch bsize * 1 * 28 * 28
    batch_size = batch.size(0)

    batch = batch.view(batch_size, -1)  # bsize * 784
    ps = torch.rand(batch_size).unsqueeze(1).expand(batch_size, h * w)
    mask = torch.rand(batch.size())
    mask = (mask >= ps).float()  # bsize * 784

    grid = grid.unsqueeze(0).expand(batch_size, h * w, 2)

    return torch.cat([batch.unsqueeze(-1), grid], dim=-1), mask


def loss_function(distribution_params, target_image):
    mu, logvar = distribution_params[:, :, 0], distribution_params[:, :, 1]
    loss = ((target_image - mu).pow(2) / (2 * logvar.exp().pow(2)) + 0.5 * logvar + .5 * np.log(2 * np.pi)).sum(dim=1).mean()
    return loss


def train(context_encoder, target_network, train_loader, optimizer, n_epochs, device, batch_size, h=28, w=28):
    context_encoder.train()
    target_network.train()
    xs = np.linspace(0, 1, h)
    ys = np.linspace(0, 1, w)
    xx, yy = np.meshgrid(xs, ys)
    grid = torch.tensor(np.stack([xx, yy], axis=-1)).float().to(device).view(h * w, 2)  # size 784*2

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        running_loss = 0.0
        last_log_time = time.time()
        for batch_idx, (batch, _) in enumerate(train_loader):

            if ((batch_idx % 100) == 0) and batch_idx > 1:
                print("epoch {} | batch {} | mean running loss {:.2f} | {:.2f} batch/s".format(epoch, batch_idx,
                                                                                               running_loss / 100,
                                                                                               100 / (
                                                                                                       time.time() - last_log_time)))
                last_log_time = time.time()
                running_loss = 0.0

            context_data, mask = random_sampling(batch=batch, grid=grid, h=h, w=w)
            # context data size (bsize,h*w,3) with 3 = (pixel value, coord_x,coord_y)

            context_full = context_encoder(context_data)  # size bsize,h with h =hidden size

            mask = mask.unsqueeze(-1)
            r_masked = (context_full * mask).sum(dim=1) / (1 + mask.sum(dim=1))  # bsize * hidden_size
            r_full = context_full.mean(dim=1)

            # resize context to have one context per input coordinate
            r_masked = r_masked.unsqueeze(1).expand(-1, h * w, -1)
            grid_input = grid.unsqueeze(0).expand(batch_size, -1, -1)
            target_input = torch.cat([r_masked, grid_input], dim=-1)  # TODO Check

            distribution_params = target_network.forward(target_input)
            loss = loss_function(distribution_params, batch.view(batch_size, h * w))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # add loss
            running_loss += loss.item()
            epoch_loss += loss.item()
        print("Epoch loss : {}".format(epoch_loss))
    return


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    batch_size = 32

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)

    context_encoder = ContextEncoder().to(device)
    target_network = TargetNetwork().to(device)
    full_model_params = list(context_encoder.parameters()) + list(target_network.parameters())
    optimizer = optim.RMSprop(full_model_params, lr=0.0001, momentum=0.05)

    train(context_encoder, target_network, train_loader, optimizer, 10, device, batch_size)


if __name__ == '__main__':
    main()
