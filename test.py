from utils import *
from argparse import ArgumentParser
from models import *
from torchvision import datasets, transforms


def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dataset = datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > .5).float())
    ]))
    if args.quick:
        dataset = torch.utils.data.Subset(dataset, [i for i in range(400)])
    test_loader = torch.utils.data.DataLoader(
        dataset,
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
    if args.autoregressive:
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

                    context_masked = (context_full * mask.unsqueeze(-1)).sum(dim=1) / (
                            1e-8 + mask.sum(dim=1, keepdim=True))
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
                    context_masked = (context_full * mask.unsqueeze(-1)).sum(dim=1) / (
                            1e-8 + mask.sum(dim=1, keepdim=True))
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
                    context_masked = (context_full * mask.unsqueeze(-1)).sum(dim=1) / (
                            1e-8 + mask.sum(dim=1, keepdim=True))
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
    else:
        n_context_max = 60
        if args.test_mode == "sequential":

            # assert args.bsize == 1
            nll_test = np.zeros(n_context_max)
            for i, (image, _) in enumerate(test_loader):
                print("batch {}/{}".format(i, len(test_loader)))
                # if i > 10:
                #     break
                image = image.view(-1, h * w, 1).to(device)
                mask = torch.zeros(args.bsize, h * w).to(device)
                context_full = context_encoder(torch.cat([image, grid], dim=-1))
                nll_image = 0
                for k in range(n_context_max):

                    context_masked = (context_full * mask.unsqueeze(-1)).sum(dim=1) / (
                            1e-8 + mask.sum(dim=1, keepdim=True))
                    z_context = sample_z(context_to_dist(context_masked))

                    decoded_image = decoder(torch.cat([z_context.unsqueeze(1).expand(-1, h * w, -1), grid], dim=-1))
                    if k > 0:
                        nll_test[k] = nll_test[k] + F.binary_cross_entropy(decoded_image, image.view(-1, h * w, 1),
                                                                           reduction='sum').item()

                    next_pixel = k
                    mask[:, next_pixel] = 1

            nll_test = nll_test / (len(test_loader) * args.bsize*h*w)
            p_test = np.exp(-nll_test)
            np.save('p_test_sequential.npy', p_test)
            plt.plot(p_test[1:])
            plt.show()

        elif args.test_mode == "random":
            # assert args.bsize == 1
            nll_test = np.zeros(n_context_max)
            for i, (image, _) in enumerate(test_loader):
                print("batch {}/{}".format(i, len(test_loader)))
                # if i > 10:
                #     break
                image = image.view(-1, h * w, 1).to(device)
                mask = torch.zeros(args.bsize, h * w).to(device)
                context_full = context_encoder(torch.cat([image, grid], dim=-1))
                nll_image = 0
                for k,p in enumerate(np.random.choice(h * w, n_context_max, replace=False)):

                    context_masked = (context_full * mask.unsqueeze(-1)).sum(dim=1) / (
                            1e-8 + mask.sum(dim=1, keepdim=True))
                    z_context = sample_z(context_to_dist(context_masked))

                    decoded_image = decoder(torch.cat([z_context.unsqueeze(1).expand(-1, h * w, -1), grid], dim=-1))
                    if k > 0:
                        nll_test[k] = nll_test[k] + F.binary_cross_entropy(decoded_image, image.view(-1, h * w, 1),
                                                                           reduction='sum').item()

                    next_pixel = p
                    mask[:, next_pixel] = 1

            nll_test = nll_test / (len(test_loader) * args.bsize*h*w)
            p_test = np.exp(-nll_test)
            np.save('p_test_random.npy', p_test)
            plt.plot(p_test[1:])
            plt.show()


parser = ArgumentParser()
parser.add_argument("--resume_file", type=str, default="models/NP_model_epoch_490.pt")
parser.add_argument("--bsize", type=int, default=10)
parser.add_argument("--n_pixels", type=int, default=100)
parser.add_argument("--autoregressive", type=int, default=1)
parser.add_argument("--test_mode", type=str, choices=["sequential", "random", "highest_var"],
                    default="sequential")
parser.add_argument("--quick", type=int, default=0)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
