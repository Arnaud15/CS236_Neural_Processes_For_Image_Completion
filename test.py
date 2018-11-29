from utils import *
from argparse import ArgumentParser
from models import *
from torchvision import datasets, transforms


def random_mask(bsize, n_pixels_to_keep, total_pixels=784):
    a = np.array([np.random.choice(total_pixels, n_pixels_to_keep, replace=False) for _ in range(bsize)])
    mask = np.zeros((bsize, total_pixels))
    for i in range(bsize):
        mask[i, a[i, :]] = 1
    return torch.tensor(mask).float()


def upper_half_mask(bsize, n_rows=14, h=28, w=28):
    mask = np.zeros((bsize, h, w))
    mask[:, :n_rows, :] = 1
    return torch.tensor(mask).float().view(bsize, -1)


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


def get_sample_images(batch, h, w, context_encoder, context_to_dist, decoder, n_pixels, n_samples, mask=None,
                      save=False, save_file=""):
    # make grid
    device = batch.device
    grid = make_mesh_grid(h, w).view(1, h * w, 2).expand(batch.size(0), -1, -1).to(device)
    context_full = context_encoder(torch.cat([batch, grid], dim=-1))

    # sample random indices to keep
    if mask is None:
        mask = random_mask(batch.size(0), n_pixels, total_pixels=784).to(device)
    else:
        mask = mask.to(device)
    context_masked = (context_full * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
    z_context = torch.cat(
        [sample_z(context_to_dist(context_masked)).unsqueeze(1).expand(-1, h * w, -1) for i in range(n_samples)],
        dim=0)
    decoded_images = decoder(
        torch.cat([z_context, grid.expand(n_samples, -1, -1, -1).view(-1, h * w, 2)], dim=-1))
    output_grid = display_images(original_image=batch, mask=mask, reconstructed_image=decoded_images)
    if save:
        plt.imsave(save_file, output_grid)
    return output_grid


def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > .5).float())
        ])),
        batch_size=args.bsize, shuffle=False)

    context_encoder = ContextEncoder()
    context_to_dist = ContextToLatentDistribution()
    decoder = Decoder()

    if args.resume_file is not None:
        load_models(args.resume_file, context_encoder, context_to_dist, decoder)

    context_encoder = context_encoder.to(device)
    context_to_dist = context_to_dist.to(device)
    decoder = decoder.to(device)

    h, w = 28, 28
    grid = make_mesh_grid(h, w).view(1, h * w, 2).expand(args.bsize, -1, -1).to(device)
    # autoregressive
    if args.test_mode == "sequential":
        nll_test = 0
        for i, (image, _) in enumerate(test_loader):
            # if i > 10:
            #     break
            image = image.view(-1, h * w, 1).to(device)
            mask = torch.zeros(args.bsize, h * w).to(device)
            context_full = context_encoder(torch.cat([image, grid], dim=-1))
            nll_image = 0
            for k in range(0, h * w):
                mask[:, :k] = 1

                context_masked = (context_full * mask.unsqueeze(-1)).sum(dim=1) / (1e-8 + mask.sum(dim=1, keepdim=True))
                z_context = sample_z(context_to_dist(context_masked))
                decoded_pixel = decoder(torch.cat([z_context, grid[:, k]], dim=-1))
                # import pdb;
                # pdb.set_trace()

                y_pred = decoded_pixel
                y_true = image[:, k]
                nll_pixel = F.binary_cross_entropy(y_pred, y_true, reduction='sum').item()
                nll_image += nll_pixel
            nll_test += nll_image
            # print("nll_pixel {:.2f}".format(nll_pixel))
            print("batch {}/{}".format(i, len(test_loader)))
        nll_test /= (len(test_loader) * args.bsize)
        print("NLL TEST SEQUENTIAL {}".format(nll_test))
    if args.test_mode == "random":
        nll_test = 0
        for i, (image, _) in enumerate(test_loader):
            # if i > 10:
            #     break
            image = image.view(-1, h * w, 1).to(device)
            mask = torch.zeros(args.bsize, h * w).to(device)
            context_full = context_encoder(torch.cat([image, grid], dim=-1))
            nll_image = 0
            random_order = np.random.permutation(h * w)
            for k in random_order:
                context_masked = (context_full * mask.unsqueeze(-1)).sum(dim=1) / (1e-8 + mask.sum(dim=1, keepdim=True))
                z_context = sample_z(context_to_dist(context_masked))
                decoded_pixel = decoder(torch.cat([z_context, grid[:, k]], dim=-1))
                # import pdb;
                # pdb.set_trace()

                y_pred = decoded_pixel
                y_true = image[:, k]
                nll_pixel = F.binary_cross_entropy(y_pred, y_true, reduction='sum').item()
                nll_image += nll_pixel
                mask[:, k] = 1
            nll_test += nll_image
            # print("nll_pixel {:.2f}".format(nll_pixel))
            print("batch {}/{}".format(i, len(test_loader)))
        nll_test /= (len(test_loader) * args.bsize)
        print("NLL TEST RANDOM {}".format(nll_test))

    if args.test_mode == "highest_var":
        # assert args.bsize == 1
        nll_test = 0
        for i, (image, _) in enumerate(test_loader):
            # if i > 10:
            #     break
            image = image.view(-1, h * w, 1).to(device)
            mask = torch.zeros(args.bsize, h * w).to(device)
            context_full = context_encoder(torch.cat([image, grid], dim=-1))
            nll_image = 0
            for k in range(h * w):
                context_masked = (context_full * mask.unsqueeze(-1)).sum(dim=1) / (1e-8 + mask.sum(dim=1, keepdim=True))
                z_context = sample_z(context_to_dist(context_masked))

                decoded_pixel = decoder(torch.cat([z_context.unsqueeze(1).expand(-1, h * w, -1), grid], dim=-1))
                # print("decoded pixel max {}".format(decoded_pixel.max()))
                # import pdb;
                # pdb.set_trace()
                a = decoded_pixel * (1 - decoded_pixel) * (1 - mask.unsqueeze(-1))

                _, next_pixel = torch.max(a, dim=1)

                y_pred = decoded_pixel[:, next_pixel]
                y_true = image[:, next_pixel]
                # import pdb;
                # pdb.set_trace()
                nll_pixel = F.binary_cross_entropy(y_pred, y_true, reduction='sum').item()
                nll_image += nll_pixel

                mask[:, next_pixel] = 1
            nll_test += nll_image
            # print("nll_pixel {:.2f}".format(nll_pixel))
            print("batch {}/{} nll {}".format(i, len(test_loader), nll_image))
        nll_test /= (len(test_loader) * args.bsize)
        print("NLL TEST highest var {}".format(nll_test))


parser = ArgumentParser()
parser.add_argument("--resume_file", type=str, default="models_saved/NP_model_epoch_5000.pt")
parser.add_argument("--bsize", type=int, default=10)
parser.add_argument("--n_pixels", type=int, default=100)
parser.add_argument("--n_samples", type=int, default=3, help="number of samples per context point")
parser.add_argument("--mask_type", type=str, choices=["random", "upper"], default="random")
parser.add_argument("--test_mode", type=str, choices=["sequential", "random", "highest_var"],
                    default="highest_var")
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
