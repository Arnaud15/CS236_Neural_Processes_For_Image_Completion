from utils import *
from argparse import ArgumentParser
from models import *
from torchvision import datasets, transforms


def random_mask(bsize, n_pixels_to_keep, total_pixels=784):
    a = np.array([np.random.choice(total_pixels, n_pixels_to_keep, replace=False) for _ in range(bsize)])
    # b = np.arange(0, bsize, dtype=int)
    mask = np.zeros((bsize, total_pixels))
    for i in range(bsize):
        mask[i, a[i, :]] = 1
    return torch.tensor(mask).float()


def display_images(original_image, mask, reconstructed_image, h=28, w=28):
    bsize = original_image.size(0)
    original_image = original_image.view(bsize, 1, h, w).expand(-1, 3, -1, -1)
    mask = mask.view(bsize, 1, 28, 28).expand(-1, 3, -1, -1)
    masked_image = torch.zeros_like(original_image)
    masked_image[:, 2] = 1
    masked_image[mask == 1] = 0
    masked_image = masked_image + mask * original_image
    masked_image = torch.min(masked_image, torch.ones_like(masked_image))
    reconstructed_image = reconstructed_image.view(-1, 1, 28, 28).expand(-1, 3, h, w)
    stacked = torch.cat([original_image, masked_image, reconstructed_image], dim=0)

    grid = make_grid(stacked, nrow=bsize)
    return np.transpose(grid.detach().numpy(), (1, 2, 0))


def get_sample_images(batch, h, w, context_encoder, context_to_dist, decoder, n_pixels, n_samples, save=False,
                      save_file=""):
    # make grid
    device = batch.device
    grid = make_mesh_grid(h, w).view(1, h * w, 2).expand(batch.size(0), -1, -1).to(device)
    context_full = context_encoder(torch.cat([batch, grid], dim=-1))

    # sample random indices to keep
    mask = random_mask(batch.size(0), n_pixels, total_pixels=784).to(device)
    context_masked = (context_full * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
    z_context = torch.cat(
        [sample_z(context_to_dist(context_masked)).unsqueeze(1).expand(-1, h * w, -1) for i in range(n_samples)],
        dim=0)
    decoded_images = decoder(
        torch.cat([z_context, grid.expand(n_samples, -1, -1, -1).view(-1, h * w, 2)], dim=-1))
    output_grid = display_images(original_image=batch, mask=mask, reconstructed_image=decoded_images)
    if not save:
        plt.imshow(output_grid)
        plt.show()
    else:
        plt.imsave(save_file, output_grid)


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

    h, w = 28, 28

    batch, _ = next(iter(test_loader))
    batch = batch.view(batch.size(0), -1, 1).to(device)  # bsize * 784 *1

    get_sample_images(batch, h, w, context_encoder, context_to_dist, decoder, args.n_pixels, args.n_samples, save=False)


parser = ArgumentParser()
parser.add_argument("--resume_file", type=str, default="models/NP_model_epoch_5000.pt")
parser.add_argument("--bsize", type=int, default=10)
parser.add_argument("--n_pixels", type=int, default=100)
parser.add_argument("--n_samples", type=int, default=5, help="number of samples per context point")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
