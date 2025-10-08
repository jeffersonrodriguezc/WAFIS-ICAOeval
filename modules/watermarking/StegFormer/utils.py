import os
import torch
import numpy as np
import sqlite3
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import torchvision.transforms as transforms

def ssim(img1, img2, cal_ssim):
    s = cal_ssim(img1, img2)
    return s.cpu().detach().numpy()

def psnr(cover, generated, cal_psnr):
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

def compute_image_score(cover, generated, cal_psnr, cal_ssim):
    
    p = psnr(cover, generated, cal_psnr)
    s = ssim(cover, generated, cal_ssim)
    
    return p, s

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

    x255 = torch.clamp(x, 0.0, 1.0) * 255.0
    symbols = torch.round(x255 / mapping_factor).long()
    symbols = torch.clamp(symbols, 0, msg_range_high - 1)

    return symbols

def load_and_preprocess_image(image_path: Path, 
                              im_size: int) -> torch.Tensor:
    """
    Loads an image from path and preprocesses it for model input.
    Args:
        image_path (Path): Path to the image file.
        im_size (int): Target size for the image (im_size, im_size).
        device_id (int): GPU device ID.
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    # Load and process the cover image
    img = Image.open(image_path).convert('RGB')
    img_cover = ImageOps.fit(img, (im_size,im_size))
    image_tensor = transforms.ToTensor()(img_cover)

    return image_tensor

def get_watermark_from_db(db_path: str, filename: str) -> torch.Tensor:
    """
    Retrieves the binary watermark message for a given filename from the SQLite database.
    Args:
        db_path (str): Path to the SQLite watermark database file.
        filename (str): The filename (image_filename) for which to retrieve the watermark.
    Returns:
        numpy.ndarray: Array of floats representing the watermark message, or None if not found.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT watermark_data FROM watermarks WHERE image_filename = ?", (filename,))
        result = cursor.fetchone()
        if result:
            # Convert the string to a tensor of floats
            watermark_str = result[0]
            # Ensure it's a list of floats 
            message_list = [float(bit) for bit in watermark_str]
            return np.array(message_list, dtype=np.float32)
        else:
            print(f"Watermark not found for filename: {filename} in {db_path}")
            return None
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_mapping_params(bpp: int):
    if bpp == 1:  return 1, 2, 255.0
    if bpp == 2:  return 2, 2, 255.0
    if bpp == 3:  return 3, 2, 255.0
    if bpp == 4:  return 1, 16, 256.0/16.0
    if bpp == 6:  return 3, 4, 256.0/4.0
    if bpp == 8:  return 2, 16, 256.0/16.0
    raise ValueError(f"Unsupported bpp: {bpp}")

def symbols_to_message_image(message: np.ndarray, bpp: int, shape_image: tuple[int, int]) -> torch.Tensor:
    """
    message: (N,) array of integers in [0..msg_range_high-1]
    shape_image: (H,W)
    returns image tensor (C,H,W) in [0,1] with mapped values
    """
    n_channels, _, mapping_factor = get_mapping_params(bpp)
    mapped_message = message * mapping_factor
    message_image = mapped_message.reshape(n_channels, shape_image[0], shape_image[1]).astype(np.float32)
    message_image = torch.from_numpy(message_image).float()
    
    return message_image/255.0

def make_output_writable(path):
    """
    Make the output directory writable by changing permissions recursively.
    """
    for root, dirs, files in os.walk(path):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o777)
        for f in files:
            os.chmod(os.path.join(root, f), 0o666)  
    os.chmod(path, 0o777)
