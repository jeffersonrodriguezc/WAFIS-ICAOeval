import os
import numpy as np
import torch
from PIL import Image, ImageOps
import torch
import matplotlib.pyplot as plt
from model import StegFormer

def show_tensor_image(tensor_img, title=None, figsize=(5,5)):
    """
    Displays a torch tensor image loaded with torchvision.io.read_image.
    
    Args:
        tensor_img (torch.Tensor): Tensor of shape (C, H, W) with values in [0, 255].
        title (str, optional): Optional title for visualization.
        figsize (tuple): Matplotlib figure size.
    """
    if not isinstance(tensor_img, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor from read_image().")

    if tensor_img.dim() != 3:
        raise ValueError(f"Tensor must have shape (C, H, W). Got: {tensor_img.shape}")

    # Ensure CPU and uint8
    img = tensor_img.detach().cpu()

    # Convert to PIL
    pil_img = Image.fromarray(img.permute(1, 2, 0).numpy())

    # Plot
    plt.figure(figsize=figsize)
    plt.imshow(pil_img)
    plt.axis("off")

    if title is not None:
        plt.title(title)

    plt.show()

def ssim(img1, img2, cal_ssim):
    s = cal_ssim(img1, img2)
    return s.cpu().detach().numpy()

def psnr(cover, generated, cal_psnr):
    psnr = cal_psnr(generated, cover)
    return psnr.cpu().detach().numpy()

def reverse_message_image(message_image, bpp):
    """
    Reverses the process to extract the original message array from a mapped image.
    """
    if bpp == 1:
        msg_range_high = 2  # Range [0, 1]
        mapping_factor = 255
    elif bpp == 2:
        msg_range_high = 2  # Range [0, 1]
        mapping_factor = 255
    elif bpp == 3:
        msg_range_high = 2  # Range [0, 1]
        mapping_factor = 255
    elif bpp == 4:
        msg_range_high = 16  # Range [0, 15]
        mapping_factor = 256 / msg_range_high
    elif bpp == 6:
        msg_range_high = 4  # Range [0, 3]
        mapping_factor = 256 / msg_range_high
    elif bpp == 8:
        msg_range_high = 16  # Range [0, 15]
        mapping_factor = 256 / msg_range_high
    else:
        raise ValueError(f"Unsupported bpp: {bpp}")
    
    x = message_image

    x255 = x * 255.0
    msg_img = x255 / mapping_factor
    symbols = torch.clamp(msg_img, 0, msg_range_high - 1)

    return symbols, x255

def _sanitize_state_dict(sd: dict) -> dict:
    # 1) elimina claves de contadores de FLOPs/params
    drop = [k for k in sd.keys() if 'total_ops' in k or 'total_params' in k]
    for k in drop:
        sd.pop(k, None)
    # 2) quita 'module.' si se entrenó con DataParallel
    if any(k.startswith('module.') for k in sd.keys()):
        sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    return sd

def load_weights(encoder, decoder, save_path, tag='psnr'):
    enc_path = os.path.join(save_path, f'best_{tag}_encoder.pth.tar')
    dec_path = os.path.join(save_path, f'best_{tag}_decoder.pth.tar')
    assert os.path.exists(enc_path) and os.path.exists(dec_path), \
        f"Checkpoints not found for tag '{tag}' in {save_path}"

    print(f"🔄 Restoring checkpoint from '{tag.upper()}'...")

    enc_ckpt = torch.load(enc_path, map_location='cpu')
    dec_ckpt = torch.load(dec_path, map_location='cpu')

    enc_sd = _sanitize_state_dict(enc_ckpt['state_dict'])
    dec_sd = _sanitize_state_dict(dec_ckpt['state_dict'])

    # strict=False para ignorar cualquier resto inocuo
    missing_e, unexpected_e = encoder.load_state_dict(enc_sd, strict=False)
    missing_d, unexpected_d = decoder.load_state_dict(dec_sd, strict=False)

    if unexpected_e or unexpected_d:
        print("⚠️ Ignored unexpected keys:", unexpected_e, unexpected_d)
    if missing_e or missing_d:
        print("⚠️ Missing keys:", missing_e, missing_d)

    print(f"✅ Loaded weights from {save_path} with tag '{tag}'")

def build_models(image_size, secret_channels):
    # Mirror train.py variants
    print("Using StegFormer-B model with default parameters")
    encoder = StegFormer(img_resolution=image_size, 
                        input_dim=(3 + secret_channels), cnn_emb_dim=16, output_dim=3)
    decoder = StegFormer(img_resolution=image_size, input_dim=3, cnn_emb_dim=16, output_dim=secret_channels)

    return encoder, decoder

def show_pil_image(pil_img, title=None, figsize=(5,5)):
    """
    Displays a PIL image using matplotlib.
    
    Args:
        pil_img (PIL.Image.Image): Image already in PIL format.
        title (str, optional): Title of the plot.
        figsize (tuple): Size of the displayed figure.
    """
    if not isinstance(pil_img, Image.Image):
        raise TypeError("Input must be a PIL.Image.Image object")
    
    plt.figure(figsize=figsize)
    plt.imshow(pil_img)
    plt.axis("off")
    
    if title is not None:
        plt.title(title)
    
    plt.show()

def read_image_pil(path, im_size):
    img = Image.open(path).convert('RGB')
    img_cover = ImageOps.fit(img, im_size)
    img_cover = img_cover

    return img_cover

def visualize_message_image(message):
    """
    Visualize a message tensor as one or multiple grayscale images.
    
    Args:
        message (torch.Tensor): Tensor with shape (C, H, W) or (1, C, H, W).
                               Values are assumed to be in [0, 255] or [0, 1].
    """
    # Ensure tensor on CPU and without gradients
    if not isinstance(message, torch.Tensor):
        raise TypeError("message must be a torch.Tensor")

    msg = message.detach().cpu()

    # Allow (1, C, H, W) or (C, H, W)
    if msg.dim() == 4:
        if msg.shape[0] != 1:
            raise ValueError(f"Expected shape (1, C, H, W) or (C, H, W), got {msg.shape}")
        msg = msg[0]  # remove batch dim

    if msg.dim() != 3:
        raise ValueError(f"Expected 3D tensor (C, H, W), got shape {msg.shape}")

    C, H, W = msg.shape

    # Convert to numpy
    msg_np = msg.numpy()

    # Single channel: just show one image
    if C == 1:
        plt.figure(figsize=(4, 4))
        plt.imshow(msg_np[0], cmap="gray")
        plt.title("Message channel 0")
        plt.axis("off")
        plt.show()
        return

    # Multiple channels: show each as a separate subplot
    cols = min(C, 4)
    rows = (C + cols - 1) // cols

    plt.figure(figsize=(4 * cols, 4 * rows))
    for i in range(C):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(msg_np[i], cmap="gray")
        plt.title(f"Channel {i}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def generate_message_image(im_size = 256, bpp = 1):
    """
    Generates a message array as a NumPy array for a given bpp value,
    based on the specified channels and pixel value ranges.
    """
    m_size_flat = im_size[0] * im_size[1]
    
    if bpp == 1:
        n_channels = 1
        msg_range_high = 2  # Range [0, 1]
        mapping_factor = 255
    elif bpp == 2:
        n_channels = 2
        msg_range_high = 2  # Range [0, 1]
        mapping_factor = 255
    elif bpp == 3:
        n_channels = 3
        msg_range_high = 2  # Range [0, 1]
        mapping_factor = 255
    elif bpp == 4:
        n_channels = 1
        msg_range_high = 16  # Range [0, 15]
        mapping_factor = 256 / msg_range_high
    elif bpp == 6:
        n_channels = 3
        msg_range_high = 4  # Range [0, 3]
        mapping_factor = 256 / msg_range_high
    elif bpp == 8:
        n_channels = 2
        msg_range_high = 16  # Range [0, 15]
        mapping_factor = 256 / msg_range_high
    else:
        raise ValueError(f"Unsupported bpp: {bpp}")
    
    # Generate the values for the message
    total_values = int(m_size_flat * n_channels)
    message_flat = np.random.randint(low=0, high=msg_range_high, size=total_values)
    mapped_message = message_flat * mapping_factor
    
    # Reshape the flat array to the correct image shape
    #message_image = mapped_message.reshape(self.im_size[0], self.im_size[1], n_channels)
    message_image = mapped_message.reshape(n_channels, im_size[0], im_size[1]).astype(np.float32)

    # Convert the NumPy array to a PyTorch tensor
    message_image = torch.from_numpy(message_image).float()

    return message_flat, message_image