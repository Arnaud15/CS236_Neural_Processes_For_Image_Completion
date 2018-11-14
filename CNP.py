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
        return torch.mean(self.layer3(x), dim=1)


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


def random_sampling(batch):
    N = np.random.randint(1, 785)
    batch_size = batch.shape[0]
    batch_context = []
    for k in range(batch_size):
        random_indices = torch.randint(0, 28, size=(N, 2)).long()
        train_X = random_indices.float() / 27
        train_Y = batch[k][0][random_indices[:, 0], random_indices[:, 1]]
        context_data = torch.cat((train_X, train_Y.view(N, 1)), dim=1)
        batch_context.append(context_data)
    batch_context = torch.stack(batch_context, dim=0)
    full_Y = batch.transpose(2, 3).contiguous().view(batch_size, 784)
    return batch_context, full_Y


def loss_function(distribution_params, full_Y):
    mu, logvar = distribution_params[:, :, 0], distribution_params[:, :, 1]
    loss = torch.sum((full_Y - mu).pow(2) / (2 * logvar.exp()) + 0.5 * logvar)
    return loss


def train(context_encoder, target_network, train_loader, optimizer, n_epochs, device, batch_size):
    context_encoder.train()
    target_network.train()
    full_range = np.linspace(0, 1, num=28)
    full_X = np.transpose([np.tile(full_range, 28), np.repeat(full_range, 28)])
    full_X = np.repeat(full_X[np.newaxis, :, :], batch_size, axis=0)
    full_X = torch.from_numpy(full_X)
    full_X = full_X.to(device).float()
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

            context_data, full_Y = random_sampling(batch)
            context_data, full_Y = context_data.to(device), full_Y.to(device)
            aggregated_context_embedding = context_encoder.forward(context_data)
            aggregated_context_embedding = aggregated_context_embedding.view(batch_size, 1, -1).expand(-1, 784, -1)
            target_input = torch.cat((aggregated_context_embedding, full_X), dim=2)
            distribution_params = target_network.forward(target_input)
            loss = loss_function(distribution_params, full_Y)

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
