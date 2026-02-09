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
    Proyecta la perturbación delta dentro de una bola L-infinito de radio epsilon.
    Asegura que ningún cambio individual supere el umbral.
    """
    return torch.clamp(delta, -epsilon, epsilon)

def pgd_step_linf(delta, grad, step_size):
    """
    Paso PGD usando el signo del gradiente (típico de L-infinito).
    """
    # Usamos el signo del gradiente: si el gradiente es positivo, subimos; si es negativo, bajamos.
    # Como queremos MINIMIZAR la pérdida, restamos el signo.
    return delta - step_size * torch.sign(grad)

def l2_project(delta, epsilon):
    delta_norm = torch.norm(delta.view(delta.size(0), -1), dim=1, keepdim=True)
    factor = torch.clamp(epsilon / (delta_norm + 1e-12), max=1.0)
    return delta * factor

def pgd_step(delta, grad, step_size):
    """
    Realiza un paso de actualización PGD normalizado en L2.
    """
    # 1. Calcular la norma L2 del gradiente para cada elemento del batch
    # .view(N, -1) aplana el tensor manteniendo el batch, para calcular la norma por imagen
    grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1, keepdim=True)
    
    # 2. Normalizar el gradiente
    # Dividimos el gradiente por su norma. Ahora 'normalized_grad' tiene longitud 1.
    # Sumamos 1e-8 para evitar división por cero si el gradiente es nulo.
    normalized_grad = grad / (grad_norm + 1e-8)
    
    # 3. Actualizar delta (Gradient DESCENT)
    # RESTAMOS el gradiente para MINIMIZAR la función de pérdida.
    return delta - step_size * normalized_grad    

