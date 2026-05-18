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
        #image_tensor = transforms.ToTensor()(img_cover)
        image_tensor = torch.from_numpy(np.array(img_cover)).permute(2, 0, 1).float()

    elif image_format == 'npy':
        img_cover = np.load(image_path, allow_pickle=True)  
        img_cover = np.transpose(img_cover, (2, 0, 1))  # H,W,C -> C,H,W 
        # avoid normalization again (remember that npy are saved in [0,1])
        image_tensor = torch.from_numpy(img_cover)
    else:
        raise ValueError(f"Unsupported image format: {image_format}")

    return image_tensor

def alignment(images, size=(112, 112)):
  return F.interpolate(
            images,
            size=size,
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
    delta_norm = torch.norm(delta.view(delta.size(0), -1), dim=1).view(-1, 1, 1, 1)
    factor = torch.clamp(epsilon / (delta_norm + 1e-12), max=1.0)
    return delta * factor

def pgd_step(delta, grad, step_size):
    """
    Perform a PGD step using the normalized gradient. This is used for the L2 case.
    """
    # 1. Calculate the L2 norm of the gradient for each element in the batch
    # .view(N, -1) flattens the tensor while keeping the batch dimension, to calculate the norm per element.
    grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1, 1)
    
    # 2. Normalize the gradient
    # We divide the gradient by its norm. Now 'normalized_grad' has length 1.
    # We add 1e-8 to avoid division by zero if the gradient is zero.
    normalized_grad = grad / (grad_norm + 1e-8)
    
    # 3. Update delta by moving in the direction of the negative normalized gradient, scaled by the step size.
    # We subtract the gradient to MINIMIZE the loss function.
    return delta - step_size * normalized_grad    



def pgd_step_linf_masked(delta, grad, step_size, mask):
    """
    Perform a PGD step using the sign of the gradient (for the L-infinity) with mask.
    """
    grad_filtered = grad * mask
    # We use the sign of the gradient: if the gradient is positive, we go up; if it's negative, we go down.
    # Since we want to MINIMIZE the loss, we subtract the sign.
    return delta - step_size * torch.sign(grad_filtered)

def generate_background_mask(model, input_tensor, target_class=15):
    """
    Generates a binary mask to isolate a specific object (e.g., a person)
    and remove the background using a pre-trained segmentation model.

    Args:
        model: Pre-trained PyTorch segmentation model (in .eval() mode).
        input_tensor: Preprocessed image tensor [1, 3, H, W].
        target_class: Integer ID of the class to keep (default 15 for 'person' in Pascal VOC).

    Returns:
        binary_mask: Float tensor [1, 1, H, W] where 1.0 is the object and 0.0 is the background.
    """
    # 1. Disable gradient calculation for inference
    with torch.no_grad():
        # model(input_tensor) returns a dictionary; we take the 'out' key
        # output shape: [1, num_classes, H, W]
        output = model(input_tensor)['out']

    # 2. Get the class with the highest probability for each pixel
    # argmax(1) reduces the channel dimension
    # prediction shape: [1, H, W]
    prediction = output.argmax(1)

    # 3. Create a binary mask for the target class
    # Result is a boolean tensor, converted to float (0.0 or 1.0)
    binary_mask = (prediction == target_class).float()

    # 4. Add a channel dimension to make it [1, 1, H, W]
    # This allows for easy element-wise multiplication with the original image
    binary_mask = binary_mask.unsqueeze(1)

    return binary_mask

def compute_sobel_edges_mask(img_wm, threshold=0.5, invert = False):
    """
    Computes the spatial gradient magnitude (edges/high frequencies)
    of a batch of images using the Sobel operator.

    Args:
        img_wm: Image tensor [B, C, H, W] with values in [0, 1].
        threshold: Threshold for edge detection.
        invert: If True, returns the inverted binary mask.
    Returns:
        magnitude: Tensor [B, 1, H, W] with edge intensity.
    """
    # 1. Convert to grayscale if the image has 3 channels (RGB)
    # This makes it easier to find global edges instead of processing per channel
    if img_wm.size(1) == 3:
        # Standard luminosity weights (Rec. 601) to convert RGB to Grayscale
        weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1).to(img_wm.device)
        img_gray = torch.sum(img_wm * weights, dim=1, keepdim=True) # Output: [B, 1, H, W]
    else:
        img_gray = img_wm

    # 2. Define Sobel kernels for X and Y axes
    kernel_x = torch.tensor([[-1.,  0.,  1.],
                             [-2.,  0.,  2.],
                             [-1.,  0.,  1.]]).view(1, 1, 3, 3).to(img_wm.device)

    kernel_y = torch.tensor([[-1., -2., -1.],
                             [ 0.,  0.,  0.],
                             [ 1.,  2.,  1.]]).view(1, 1, 3, 3).to(img_wm.device)

    # 3. Apply filters via 2D convolution
    # Use padding=1 so the resulting mask has the exact same size (H, W)
    grad_x = F.conv2d(img_gray, kernel_x, padding=1)
    grad_y = F.conv2d(img_gray, kernel_y, padding=1)

    # 4. Calculate the combined gradient magnitude
    # A small epsilon (1e-6) is added to avoid numerical instability (derivative of sqrt at 0)
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

    binary_mask = get_binary_mask(magnitude, threshold=threshold)

    if invert:
        return 1 - binary_mask
    else:
        return binary_mask

def get_binary_mask(magnitude, threshold=0.2):
    # Return True where edges exist otherwise returns False
    mask = (magnitude > threshold).float()
    return mask

def generate_face_box_mask(box, img_shape, device):
    """
    Builds a binary mask from MTCNN bounding box output.
    MTCNN box format: [x1, y1, x2, y2] in pixel coordinates.
    Returns: [B, C, H, W] mask — 1=attack (smooth face region), 0=skip
    """
    B, C, H, W = img_shape
    mask = torch.zeros(B, 1, H, W, device=device)
    
    for b in range(B):
        x1, y1, x2, y2 = box[b]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(W, int(x2))
        y2 = min(H, int(y2))
        mask[b, :, y1:y2, x1:x2] = 1.0

    return mask


