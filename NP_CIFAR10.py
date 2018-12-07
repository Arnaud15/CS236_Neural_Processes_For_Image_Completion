import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import time
from argparse import ArgumentParser
from models import *
import os
from utils import sample_z, log_normal, make_mesh_grid, kl_normal, save_model, random_mask_uniform, random_mask, display_images, \
    load_models
from tensorboardX import SummaryWriter



def all_forward(batch, grid, mask, context_encoder, aggregator, context_to_dist):
    '''
    :param batch: input batch of size bsize,c,h,w
    :param grid: grid of size h,w,2
    :param mask: mask of size bsize,h,w
    :param context_encoder: model (2+c -> d)
    :param aggregator: aggregates mask and context encodings (k*d ->d)
    :param context_to_dist: encoded context to z distribution
    :return: z_params_full,z_params_masked of shape (bsize,d), (bsize,d) where d is the hidden dimension of the context
    '''
    bsize, c, h, w = batch.size()

    batch = batch.view(bsize, c, h * w).transpose(1, 2)  # bsize,h*w,c
    grid = grid.view(h * w, -1).unsqueeze(0).expand(bsize, -1, -1)  # bsize,h*w,2
    context_data = torch.cat([batch, grid], dim=2)

    context_full = context_encoder(context_data)  # bsize,h*w,d

    mask = mask.view(bsize, h * w, 1)
    r_masked = aggregator(context_full, mask=mask, agg_dim=1)  # bsize * hidden_size
    r_full = aggregator(context_full, mask=None, agg_dim=1)

    z_params_masked = context_to_dist(r_masked)
    z_params_full = context_to_dist(r_full)

    return z_params_full, z_params_masked

def compute_loss(batch, grid, mask, z_params_full, z_params_masked, h, w, decoder):
    ## compute loss
    z_full = sample_z(z_params_full)  # size bsize * hidden
    z_full = z_full.unsqueeze(1).expand(-1, h * w, -1)

    # resize context to have one context per input coordinate
    grid_input = grid.view(1, h * w, -1).expand(batch.size(0), -1, -1)
    target_input = torch.cat([z_full, grid_input], dim=-1)

    reconstructed_image_mean, reconstructed_image_variance = decoder(target_input)  # bsize,h*w,1
    reconstruction_loss = - (log_normal(batch.view(batch.size(0), h * w, 3), reconstructed_image_mean, reconstructed_image_variance) * (1 - mask.view(-1, h * w))).sum(dim=1).mean()

    kl_loss = kl_normal(z_params_full, z_params_masked).mean()
    return reconstruction_loss, kl_loss, reconstructed_image_mean, reconstructed_image_variance

def train(context_encoder, context_to_dist, decoder, aggregator, train_loader, test_loader, optimizer, n_epochs, device,
          save_path,
          summary_writer, save_every=10, h=32, w=32, log=1):
    context_encoder.train()
    decoder.train()
    grid = make_mesh_grid(h, w).to(device).view(h * w, 2)  # size 1024*2

    for epoch in range(n_epochs):
        running_loss = 0.0
        last_log_time = time.time()

        # Training
        train_loss = 0.0
        for batch_idx, (batch, _) in enumerate(train_loader):
            batch = batch.to(device)
            if ((batch_idx % 100) == 0) and batch_idx > 1:
                print("epoch {} | batch {} | mean running loss {:.2f} | {:.2f} batch/s".format(epoch, batch_idx,
                                                                                               running_loss / 100,
                                                                                               100 / (
                                                                                                       time.time() - last_log_time)))
                last_log_time = time.time()
                running_loss = 0.0

            mask = random_mask_uniform(bsize=batch.size(0), h=h, w=w, device=batch.device)
            z_params_full, z_params_masked = all_forward(batch, grid, mask, context_encoder, aggregator,
                                                         context_to_dist)

            reconstruction_loss, kl_loss, reconstructed_image_mean, reconstructed_image_variance = compute_loss(batch, grid, mask, z_params_full,
                                                                             z_params_masked, h, w,
                                                                             decoder)
            loss = reconstruction_loss + kl_loss
            # if batch_idx == 0 and log:
            #     if not os.path.exists("images"):
            #         os.makedirs("images")
            #     save_images_batch(batch.cpu(), "images/CIFAR_target_epoch_{}".format(epoch), h=h, w=w)
            #     save_images_batch(reconstructed_image_mean.cpu(), "images/CIFAR_reconstruct_epoch_{}".format(epoch), h=h, w=w)
            if batch_idx % 100 == 0:
                print("reconstruction {:.2f} | kl {:.2f}".format(reconstruction_loss, kl_loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # add loss
            running_loss += loss.item()
            train_loss += loss.item()

        print("Epoch train loss : {}".format(train_loss / len(train_loader)))
        if summary_writer is not None:
            summary_writer.add_scalar("train/loss", train_loss / len(train_loader), global_step=epoch)
        if (epoch % save_every == 0) and log and epoch > 0:
            save_model(save_path, "NP_CIFAR_model_epoch_{}.pt".format(epoch), context_encoder, context_to_dist,
                       decoder,
                       device)
        #Testing
        test_loss = 0.0
        
        for batch_idx, (batch, _) in enumerate(test_loader):

            batch = batch.to(device)
            mask = random_mask_uniform(bsize=batch.size(0), h=h, w=w, device=batch.device)

            with torch.no_grad():
                z_params_full, z_params_masked = all_forward(batch, grid, mask, context_encoder, aggregator,
                                                             context_to_dist)
                reconstruction_loss, kl_loss, reconstructed_image_mean, reconstructed_image_variance = compute_loss(batch, grid, mask, z_params_full,
                                                                                 z_params_masked, h, w,
                                                                                 decoder)
                loss = reconstruction_loss + kl_loss
                test_loss += loss.item()

        if summary_writer is not None:
            summary_writer.add_scalar("test/loss", test_loss / len(test_loader), global_step=epoch)

        # do examples
        #TODO CIFAR IMAGE
        # example_batch, _ = next(iter(test_loader))
        # example_batch = example_batch[:10]
        # for n_pixels in [50, 150, 450]:
        #     mask = random_mask(example_batch.size(0), h, w, n_pixels, device=example_batch.device)
        #     z_params_full, z_params_masked = all_forward(example_batch, grid, mask, context_encoder, aggregator,
        #                                                  context_to_dist)

        #     z_context = torch.cat(
        #         [sample_z(context_to_dist(z_params_masked)).unsqueeze(1).expand(-1, h * w, -1) for i in
        #          range(3)],
        #         dim=0)
        #     decoded_images = decoder(z_context).view(example_batch.size(0), h, w)
        #     import pdb;
        #     pdb.set_trace()
        #     stacked_images = display_images(original_image=example_batch, mask=mask, reconstructed_image=decoded_images)

        #     image = torch.tensor(image).transpose(0, 2).unsqueeze(0).transpose(2, 3)

        #     # import pdb;
        #     # pdb.set_trace()
        #     if summary_writer is not None:
        #         summary_writer.add_image("test_image/{}_pixels".format(n_pixels), image, global_step=epoch)
    return


def main(args):
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    summary_writer = SummaryWriter(log_dir=args.log_dir) if args.log else None

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader = torch.utils.data.DataLoader(

        datasets.CIFAR10('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.bsize, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=args.bsize, shuffle=True)

    context_encoder = ContextEncoderCIFAR()
    context_to_dist = ContextToLatentDistribution()
    decoder = DecoderCIFAR()

    if args.resume_file is not None:
        load_models(args.resume_file, context_encoder, context_to_dist, decoder)
    context_encoder = context_encoder.to(device)
    decoder = decoder.to(device)
    context_to_dist = context_to_dist.to(device)
    full_model_params = list(context_encoder.parameters()) + list(decoder.parameters()) + list(
        context_to_dist.parameters())
    optimizer = optim.Adam(full_model_params, lr=args.lr)

    if args.aggregator == "mean":
        aggregator = MeanAgregator()
    else:
        assert args.aggregator == "attention"
        aggregator = AttentionAggregator(128)

    train(context_encoder, context_to_dist, decoder, aggregator, train_loader, test_loader, optimizer, args.epochs,
          device,
          args.models_path, summary_writer=summary_writer, save_every=args.save_every, log=args.log)


parser = ArgumentParser()
parser.add_argument("--models_path", type=str, default="models/")
parser.add_argument("--save_model", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--bsize", type=int, default=32)
parser.add_argument("--resume_file", type=str, default=None)
parser.add_argument("--save_every", type=int, default=10)
parser.add_argument("--log_dir", type=str, default="logs")
parser.add_argument("--aggregator", type=str, choices=['mean', 'attention'], default='mean')
parser.add_argument("--log", type=int, default=1)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
