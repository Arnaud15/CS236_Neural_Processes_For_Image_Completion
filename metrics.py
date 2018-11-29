import torch
import torch.nn.functional as F
import pytorch_ssim

def psnr(generated, original):
    # takes batches of images.
    batch_size = generated.size(0)
    return -10*torch.log(F.mse_loss(generated.view(batch_size, -1), original.view(batch_size, -1), reduction="none")).mean()

def cosine_similarity(generated, original):
    # takes batches of images
    batch_size = generated.size(0)
    return F.cosine_similarity(generated.view(batch_size, -1), original.view(batch_size, -1), dim=1).mean()

def ssim(generated, original):
    batch_size = generated.size(0)
    return pytorch_ssim.ssim(generated.view(batch_size, 1, 28, 28), original).mean()
