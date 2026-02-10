import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import torchvision.transforms as transforms

def is_image_file(filename, IMG_EXTENSION):
    return any(filename.endswith(extension) for extension in [IMG_EXTENSION])

def load_and_preprocess_image(image_path: Path, 
                              im_size: int, 
                              image_format: str = 'png') -> torch.Tensor:
    """
    Loads an image from the given path, resizes it to the specified size, and converts it to a PyTorch tensor.
    """
    # Load and process the cover image
    if image_format == 'png':
        img = Image.open(image_path).convert('RGB')
        img_cover = ImageOps.fit(img, (im_size,im_size))
        image_tensor = transforms.ToTensor()(img_cover)

    elif image_format == 'npy':
        arr = np.load(image_path, allow_pickle=True)  
        img_cover = np.transpose(arr, (2, 0, 1))  # H,W,C -> C,H,W 
        # avoid normalization again (remember that npy are saved in [0,1])
        image_tensor = torch.from_numpy(img_cover)
    else:
        raise ValueError(f"Unsupported image format: {image_format}")

    return image_tensor 

def alignment(images):
  return F.interpolate(
            images,
            size=(112, 112),
            mode="bilinear",
            align_corners=False,
            antialias=True
        )

def l2_norm(input,axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

def tensor2img(var):
    # var: 3 x 256 x 256 --> 256 x 256 x 3
    var = var.cpu().detach().numpy().transpose([1,2,0])
    #var = ((var+1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))

def linf_project(delta, epsilon):
    """
    Project delta onto the L-infinity ball of radius epsilon.
        This is done by clamping each element of delta to be within [-epsilon, epsilon].
        If an element of delta is greater than epsilon, it will be set to epsilon.
        If an element of delta is less than -epsilon, it will be set to -epsilon.
    """
    return torch.clamp(delta, -epsilon, epsilon)

def pgd_step_linf(delta, grad, step_size):
    """
    Perform a PGD step using the sign of the gradient (for the L-infinity).
    """
    # We use the sign of the gradient: if the gradient is positive, we go up; if it's negative, we go down.
    # Since we want to MINIMIZE the loss, we subtract the sign.
    return delta - step_size * torch.sign(grad)

def l2_project(delta, epsilon):
    """
    Project delta onto the L2 ball of radius epsilon.
        This is done by scaling delta if its L2 norm exceeds epsilon.
        If the L2 norm of delta is less than or equal to epsilon, it is returned unchanged.
    """
    delta_norm = torch.norm(delta.view(delta.size(0), -1), dim=1, keepdim=True)
    factor = torch.clamp(epsilon / (delta_norm + 1e-12), max=1.0)
    return delta * factor

def pgd_step(delta, grad, step_size):
    """
    Perform a PGD step using the normalized gradient. This is used for the L2 case.
    """
    # 1. Calculate the L2 norm of the gradient for each element in the batch
    # .view(N, -1) flattens the tensor while keeping the batch dimension, to calculate the norm per element.
    grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1, keepdim=True)
    
    # 2. Normalize the gradient
    # We divide the gradient by its norm. Now 'normalized_grad' has length 1.
    # We add 1e-8 to avoid division by zero if the gradient is zero.
    normalized_grad = grad / (grad_norm + 1e-8)
    
    # 3. Update delta by moving in the direction of the negative normalized gradient, scaled by the step size.
    # We subtract the gradient to MINIMIZE the loss function.
    return delta - step_size * normalized_grad    

