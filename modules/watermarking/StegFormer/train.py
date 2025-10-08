import torch
import torch.nn as nn
import torch.optim
import math
import numpy as np
#from critic import *
from torch.utils.tensorboard import SummaryWriter
from thop import profile
from datasets import Celeba_hq
from utils import reverse_message_image
from model import StegFormer
import os
import timm
import timm.scheduler
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torchvision.utils as tv_utils
import random

# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed); np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# loss function
class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

class Restrict_Loss(nn.Module):
    """Restrict loss using L2 loss function"""

    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, X):
        count1 = int(torch.sum(X > 1).item())
        count0 = int(torch.sum(X < 0).item())
        if count1 == 0: count1 = 1
        if count0 == 0: count0 = 1
        
        one = torch.ones_like(X)
        zero = torch.zeros_like(X)
        X_one = torch.where(X <= 1, 1, X)    # 对超过 1 的值施加惩罚
        X_zero = torch.where(X >= 0, 0, X)  # 对小于 0 的值施加惩罚
        diff_one = X_one-one
        diff_zero = zero-X_zero
        loss = torch.sum(0.5*(diff_one**2))/count1 + torch.sum(0.5*(diff_zero**2))/count0
        return loss
    
def ssim(img1, img2):
    s = cal_ssim(img1, img2)
    return s.cpu().detach().numpy()

def psnr(cover, generated):
    psnr = cal_psnr(generated, cover)
    return psnr.cpu().detach().numpy()

def get_message_accuracy(
    msg: torch.Tensor,
    deco_msg: torch.Tensor,
    bpp: int = 1
) -> float:
    """
    Calculates the pixel accuracy of decoded message images.

    Args:
        msg (torch.Tensor): The original message image tensor.
        deco_msg (torch.Tensor): The decoded message image tensor.

    Returns:
        float: The pixel accuracy of the decoded messages.

    Raises:
        TypeError: If the inputs are not torch.Tensors.
    """
    if not isinstance(msg, torch.Tensor) or not isinstance(deco_msg, torch.Tensor):
        raise TypeError("Inputs msg and deco_msg must be torch.Tensors.")
    
    # Ensure tensors are on the CPU for comparison
    if 'cuda' in str(deco_msg.device):
        deco_msg = deco_msg.cpu()
        msg = msg.cpu()

    #print("Deco msg min/max:", deco_msg.min().item(), deco_msg.max().item())
    #print("Original msg min/max:", msg.min().item(), msg.max().item())
    # get the original message from the image
    msg = reverse_message_image(msg, bpp=bpp)
    deco_msg = reverse_message_image(deco_msg, bpp=bpp)
    

    # Round the tensors to the nearest integer to handle potential floating-point errors
    original_rounded = torch.round(msg)
    decoded_rounded = torch.round(deco_msg)

    #print("Original rounded min/max:", original_rounded.min().item(), original_rounded.max().item())
    #print("Decoded rounded min/max:", decoded_rounded.min().item(), decoded_rounded.max().item())
    
    # Perform a direct element-wise comparison to find the number of matching pixels
    correct_predictions = torch.sum(original_rounded == decoded_rounded).item()
    total_elements = original_rounded.numel()
    
    # Calculate the pixel accuracy
    pixel_acc = correct_predictions / total_elements

    return pixel_acc

def compute_image_score(cover, generated):
    
    p = psnr(cover, generated)
    s = ssim(cover, generated)
    
    return p, s

def test(test_on_face_loader):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        sum_psnr = []
        sum_ssim = []
        sum_acc = []
        for i_batch, (cover, secret) in tqdm(enumerate(test_on_face_loader), 
                                             desc=" Test batches ..", leave=False, position=2):
            cover = cover.to(args.device)
            secret = secret.to(args.device)

            # encode
            msg = torch.cat([cover, secret], 1)
            encode_img = encoder(msg)

            # normalizing
            if args.norm_train == 'clamp':
                encode_img_c = torch.clamp(encode_img, 0, 1)
            else:
                encode_img_c = encode_img

            # decode
            decode_img = decoder(encode_img_c)

            # compute psnr, ssim and accuracy
            batch_acc = get_message_accuracy(secret, decode_img)
            batch_psnr, batch_ssim = compute_image_score(cover, encode_img_c)

            sum_acc.append(batch_acc)
            sum_psnr.append(batch_psnr)
            sum_ssim.append(batch_ssim)

        avg_acc = np.mean(sum_acc)
        avg_psnr = np.mean(sum_psnr)
        avg_ssim = np.mean(sum_ssim)
            
    encoder.train()
    decoder.train()

    return avg_acc, avg_psnr, avg_ssim

def save_model(save_path, iterations, tag='psnr'):
    # Save the best model
    decoder_checkpoint = {
        'iteration': iterations + 1,
        'state_dict': decoder.state_dict(),
        'optimizer': decoder_optimizer.state_dict(),
        }
                    
    for key in decoder_checkpoint['state_dict'].keys():
        decoder_checkpoint['state_dict'][key] = decoder_checkpoint['state_dict'][key].to(torch.device('cpu'))
    torch.save(decoder_checkpoint, f'{save_path}/best_{tag}_decoder.pth.tar')

    encoder_checkpoint = {
        'iteration': iterations + 1,
        'state_dict': encoder.state_dict(),
        'optimizer': encoder_optimizer.state_dict(),
        }
                    
    for key in encoder_checkpoint['state_dict'].keys():
        encoder_checkpoint['state_dict'][key] = encoder_checkpoint['state_dict'][key].to(torch.device('cpu'))
    torch.save(encoder_checkpoint, f'{save_path}/best_{tag}_encoder.pth.tar')

    #print(f"💾 Model saved with tag '{tag}' at iteration {iterations + 1}.")

def load_checkpoint_if_available(encoder, decoder, encoder_optimizer, decoder_optimizer, 
                                save_path, device, tag='psnr'):
    """
    Loads the checkpoint if it exists and resumes training.

    Parameters:
        encoder, decoder: model instances
        encoder_optimizer, decoder_optimizer: optimizer instances
        encoder_scheduler, decoder_scheduler: learning rate schedulers
        save_path: base path where checkpoints are stored
        tag: can be 'psnr', 'ssim', 'last' or 'acc' (to select which checkpoint to load)
    Returns:
        start_iteration: iteration number from which to resume
    """
    start_iteration = 0

    resume_encoder_path = os.path.join(save_path, f'best_{tag}_encoder.pth.tar')
    resume_decoder_path = os.path.join(save_path, f'best_{tag}_decoder.pth.tar')

    if os.path.exists(resume_encoder_path) and os.path.exists(resume_decoder_path):
        print(f"🔄 Restoring checkpoint from '{tag.upper()}'...")

        encoder_ckpt = torch.load(resume_encoder_path, map_location=device)
        decoder_ckpt = torch.load(resume_decoder_path, map_location=device)

        encoder.load_state_dict(encoder_ckpt['state_dict'])
        decoder.load_state_dict(decoder_ckpt['state_dict'])

        encoder_optimizer.load_state_dict(encoder_ckpt['optimizer'])
        decoder_optimizer.load_state_dict(decoder_ckpt['optimizer'])

        start_iteration = encoder_ckpt.get('iteration', start_iteration)

        print(f"✅ Checkpoint loaded. Iteration: {start_iteration}")
    else:
        print("ℹ️ No checkpoint founded. Starting from scratch.")

    return start_iteration

## PARAMETERS
parser = argparse.ArgumentParser(description='Params for stf model')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--bpp', type=int, default=1)
parser.add_argument('--iterations', type=int, default=600000)
parser.add_argument('--lr', type=float, default=2e-4)   
parser.add_argument('--warm_up_iteration', type=int, default=10000)
parser.add_argument('--warm_up_lr_init', type=float, default=5e-6)
parser.add_argument('--model_path', type=str, default='/app/runs/')
parser.add_argument('--model_ext', type=str, default='baseline')
parser.add_argument('--use_model', type=str, default='StegFormer-B')
parser.add_argument('--data_path', type=str, default='/app/facial_data/')
parser.add_argument('--dataset', type=str, default='celeba_hq')
parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--norm_train', type=str, default='clamp')
parser.add_argument('--output_act', type=str, default=None)
parser.add_argument('--secret_channels', type=int, default=1)
parser.add_argument('--image_size_train', type=int, default=256)
parser.add_argument('--tag', type=str, default='last', help="can be 'psnr', 'ssim', 'last' or 'acc'")

args = parser.parse_args()

# Paths
model_name = f'{args.use_model}_{args.model_ext}'
save_path = os.path.join(args.model_path, f'{args.secret_channels}_{args.bpp}_{args.norm_train}_{model_name}', args.dataset,'model')
isExist = os.path.exists(save_path)
if not isExist:
    os.makedirs(save_path, exist_ok=True)

log_path = os.path.join(args.model_path, f'{args.secret_channels}_{args.bpp}_{args.norm_train}_{model_name}', args.dataset,'logs')
isExist = os.path.exists(log_path)
if not isExist:
    os.makedirs(log_path, exist_ok=True)
    
log_img_path = os.path.join(args.model_path, f'{args.secret_channels}_{args.bpp}_{args.norm_train}_{model_name}', args.dataset,'samples')
isExist = os.path.exists(log_img_path)
if not isExist:
    os.makedirs(log_img_path, exist_ok=True)

train_path = os.path.join(args.data_path,args.dataset,'train/real')
test_path = os.path.join(args.data_path,args.dataset,'val/Real')

print('Training on dataset:', args.dataset)
print('Training data path:', train_path)

# tensorboard writer
writer = SummaryWriter(log_path)

# StegFormer initiate
if args.use_model == 'StegFormer-S':
    encoder = StegFormer(img_resolution=args.image_size_train, 
                         input_dim=(3 + args.secret_channels), cnn_emb_dim=8, output_dim=3,
                         drop_key=False, patch_size=2, window_size=8, output_act=args.output_act, depth=[1, 1, 1, 1, 2, 1, 1, 1, 1], depth_tr=[2, 2, 2, 2, 2, 2, 2, 2])
    decoder = StegFormer(img_resolution=args.image_size_train, 
                         input_dim=3, cnn_emb_dim=8, output_dim=args.secret_channels,
                         drop_key=False, patch_size=2, window_size=8, output_act=args.output_act, depth=[1, 1, 1, 1, 2, 1, 1, 1, 1], depth_tr=[2, 2, 2, 2, 2, 2, 2, 2])
if args.use_model == 'StegFormer-B':
    encoder = StegFormer(img_resolution=args.image_size_train, 
                         input_dim=(3 + args.secret_channels), cnn_emb_dim=16, output_dim=3)
    decoder = StegFormer(img_resolution=args.image_size_train, input_dim=3, cnn_emb_dim=16, output_dim=args.secret_channels)
if args.use_model == 'StegFormer-L':
    encoder = StegFormer(img_resolution=args.image_size_train, 
                         input_dim=(3 + args.secret_channels), cnn_emb_dim=32, output_dim=3, depth=[2, 2, 2, 2, 2, 2, 2, 2, 2])
    decoder = StegFormer(img_resolution=args.image_size_train, input_dim=3, cnn_emb_dim=32, output_dim=args.secret_channels, depth=[2, 2, 2, 2, 2, 2, 2, 2, 2])

encoder.to(args.device)
decoder.to(args.device)
encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr)
decoder_optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr)
# Create a separate CosineLRScheduler for each optimizer
encoder_scheduler = timm.scheduler.CosineLRScheduler(
    optimizer=encoder_optimizer,
    t_initial=args.iterations,
    lr_min=0,
    warmup_t=args.warm_up_iteration,
    warmup_lr_init=args.warm_up_lr_init
)
decoder_scheduler = timm.scheduler.CosineLRScheduler(
    optimizer=decoder_optimizer,
    t_initial=args.iterations,
    lr_min=0,
    warmup_t=args.warm_up_iteration,
    warmup_lr_init=args.warm_up_lr_init
)

# numbers of the parameter
with torch.no_grad():
    #test_encoder_input = torch.randn(1, (args.num_secret+1)*3, args.image_size_train, args.image_size_train).to(args.device)
    test_encoder_input = torch.randn(1, 3+args.secret_channels, args.image_size_train, args.image_size_train).to(args.device)
    test_decoder_input = torch.randn(1, 3, args.image_size_train, args.image_size_train).to(args.device)
    encoder_mac, encoder_params = profile(encoder, inputs=(test_encoder_input,))
    decoder_mac, decoder_params = profile(decoder, inputs=(test_decoder_input,))
    print("thop result:encoder FLOPs="+str(encoder_mac*2)+",encoder params="+str(encoder_params))
    print("thop result:decoder FLOPs="+str(decoder_mac*2)+",decoder params="+str(decoder_params))

# loss function
conceal_loss_function = L1_Charbonnier_loss().to(args.device)
reveal_loss_function = L1_Charbonnier_loss().to(args.device)
restrict_loss_funtion = Restrict_Loss().to(args.device)

# Metrics
cal_psnr = PeakSignalNoiseRatio().to(args.device)
cal_ssim = StructuralSimilarityIndexMeasure().to(args.device)

# dataset and dataloader
train_on_face_loader = DataLoader(
    Celeba_hq(im_size=(args.image_size_train, args.image_size_train), 
            bpp=args.bpp,
            path=train_path),
    batch_size=args.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=8,
    drop_last=True
)

test_on_face_loader = DataLoader(
    Celeba_hq(im_size=(args.image_size_train, args.image_size_train),   
            bpp=args.bpp,
            path=test_path), 
    batch_size=2,
    shuffle=False,  
    pin_memory=True,
    num_workers=2,
    drop_last=True
)

# flags 
save_log_interval = 1000
save_image_interval = 1000
save_model_interval = 1000
test_interval = 5000

# training
max_psnr = 0
max_ssim = 0
max_acc = 0
sum_loss = []

max_iter = args.iterations
with tqdm(total=max_iter, desc="Iterations ..", position=0) as pbar_while:

    # load model if available
    ##########################
    
    start_iteration  = load_checkpoint_if_available(
        encoder=encoder,
        decoder=decoder,
        encoder_optimizer=encoder_optimizer,
        decoder_optimizer=decoder_optimizer,
        save_path=save_path,
        device=args.device,
        tag=args.tag  # decide what is better to restore: 'psnr', 'ssim' 'last' or 'acc'
    )
    pbar_while.update(start_iteration)
    
    # if resuming, step the schedulers
    encoder_scheduler.step_update(start_iteration)
    decoder_scheduler.step_update(start_iteration)

    while start_iteration <= max_iter:
        for i_batch, (cover, secret) in tqdm(enumerate(train_on_face_loader), 
                                                desc="Batches ..", leave=False, position=1):
            #print('cover range:', cover.min().item(), cover.max().item())
            #print('secret range:', secret.min().item(), secret.max().item())
            if start_iteration > max_iter: break

            start_iteration += 1
            cover = cover.to(args.device)
            secret = secret.to(args.device)

            # encode
            msg = torch.cat([cover, secret], 1)
            encode_img = encoder(msg)

            # normalizing
            if args.norm_train == 'clamp':
                encode_img_c = torch.clamp(encode_img, 0, 1)
            else:
                encode_img_c = encode_img

            # decode
            decode_img = decoder(encode_img_c)

            #print('encode_img range:', encode_img.min().item(), encode_img.max().item())
            #print('decode_img range:', decode_img.min().item(), decode_img.max().item())

            # loss
            conceal_loss = conceal_loss_function(cover.cuda(), encode_img.cuda())
            reveal_loss = reveal_loss_function(secret.cuda(), decode_img.cuda())

            total_loss = None
            if args.norm_train:
                restrict_loss = restrict_loss_funtion(encode_img.cuda())
                total_loss = conceal_loss + reveal_loss + restrict_loss
            else:
                total_loss = conceal_loss + reveal_loss
            sum_loss.append(total_loss.item())

            # backward
            total_loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            encoder_optimizer.zero_grad(set_to_none=True)
            decoder_optimizer.zero_grad(set_to_none=True)
            encoder_scheduler.step_update(start_iteration)
            decoder_scheduler.step_update(start_iteration)

            with torch.no_grad(): 
                
                if (start_iteration) % save_log_interval == 0:
                    encoder.eval(); decoder.eval() 
                    # compute psnr, ssim and accuracy
                    train_message_acc = get_message_accuracy(secret, decode_img, bpp=args.bpp)
                    train_psnr, train_ssim = compute_image_score(cover, encode_img_c)

                    # Log train metrics
                    writer.add_scalar('Train Total Loss', np.mean(sum_loss), start_iteration)
                    writer.add_scalar('Train Message Acc', train_message_acc, start_iteration)
                    writer.add_scalar('Train PSNR', train_psnr, start_iteration)
                    writer.add_scalar('Train SSIM', train_ssim, start_iteration)
                    sum_loss = []
                    encoder.train(); decoder.train()
                
                if (start_iteration) % test_interval == 0 and (start_iteration) > save_model_interval :
                    # do tests
                    #print("Test begin:")
                    avg_acc, avg_psnr, avg_ssim = test(test_on_face_loader)
                    writer.add_scalar('Test Message Acc', avg_acc, start_iteration)
                    writer.add_scalar('Test PSNR', avg_psnr, start_iteration)
                    writer.add_scalar('Test SSIM', avg_ssim, start_iteration)
                    #print("Test end.")

                    if (start_iteration) > save_model_interval:
                        encoder.eval(); decoder.eval() 
                        # save last model
                        save_model(save_path, start_iteration, tag='last')
                        # save best psnr model
                        if avg_psnr > max_psnr:
                            max_psnr = avg_psnr
                            save_model(save_path, start_iteration, tag='psnr')
                            #print(f"New best PSNR: {max_psnr:.4f}. Model saved.")
                        
                        if avg_ssim > max_ssim:
                            max_ssim = avg_ssim
                            save_model(save_path, start_iteration, tag='ssim')
                            #print(f"New best SSIM: {max_ssim:.4f}. Model saved.")

                        if avg_acc > max_acc:
                            max_acc = avg_acc
                            save_model(save_path, start_iteration, tag='acc')
                            #print(f"New best ACC: {max_acc:.4f}. Model saved.")

                        encoder.train(); decoder.train()

                if (start_iteration) % save_image_interval == 0:
                    # save image
                    save_images = encode_img.cpu()
                    tv_utils.save_image(save_images, f'{log_img_path}/{str(start_iteration).zfill(6)}.png', 
                                        nrow=args.batch_size, normalize=True, range=(0, 1))
                    save_images = cover.cpu()
                    tv_utils.save_image(save_images, f'{log_img_path}/{str(start_iteration).zfill(6)}_cover.png', 
                                        nrow=args.batch_size, normalize=True, range=(0, 1)) 
                
        pbar_while.update(i_batch)
writer.close()

