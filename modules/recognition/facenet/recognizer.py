# modules/recognition/FaceNet/facenet_recognizer.py
import torch
from typing import List, Optional, Tuple, Union
from facenet_pytorch import InceptionResnetV1, MTCNN, fixed_image_standardization
 # InceptionResnetV1 as a facenet backbone
from PIL import Image
from torchvision import transforms
from facenet_pytorch.models import mtcnn as mtcnn_mod
import numpy as np
import os 
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor as tv_to_tensor
import torchvision

# utils_viz_mtcnn.py
from pathlib import Path
from typing import Union
import numpy as np
import torch
from PIL import Image
import cv2
from skimage import transform as trans

arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

def compare_crops_interactive(
    imgs, boxes_list, landmarks_list,
    image_size=160, max_show=4,
    save_path="./output/debug_compare_crops.png"
):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    if imgs.dim() == 3:
        imgs = imgs.unsqueeze(0)
    
    B = min(imgs.shape[0], max_show)
    fig, axes = plt.subplots(B, 3, figsize=(9, 3 * B))
    if B == 1:
        axes = axes[None, :]
    
    for i in range(B):
        # Original crop (numpy/cv2 path)
        crop_orig = _extract_face_float(
            imgs[i], boxes_list[i], image_size=image_size
        )
        
        # Differentiable crop (grid_sample path)
        crop_diff = _extract_face_differentiable_batch(
            imgs[i:i+1], [boxes_list[i]], [landmarks_list[i]], image_size=image_size
        ).squeeze(0)
        
        # Input
        img_np = imgs[i].detach().cpu().permute(1, 2, 0).numpy()
        axes[i, 0].imshow(np.clip(img_np, 0, 1))
        axes[i, 0].set_title(f"Input [{i}]")
        axes[i, 0].axis("off")
        
        # Original crop
        co_np = crop_orig.detach().cpu().permute(1, 2, 0).numpy()
        if co_np.max() > 1.0:
            co_np = co_np / 255.0
        axes[i, 1].imshow(np.clip(co_np, 0, 1))
        axes[i, 1].set_title(f"Original (cv2)")
        axes[i, 1].axis("off")
        
        # Differentiable crop
        cd_np = crop_diff.detach().cpu().permute(1, 2, 0).numpy()
        axes[i, 2].imshow(np.clip(cd_np, 0, 1))
        axes[i, 2].set_title(f"Differentiable")
        axes[i, 2].axis("off")
    
    fig.suptitle("Original vs Differentiable Crop", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[debug] saved → {save_path}")

def show_input_vs_crop_interactive(
    original, face_crop, title="Input vs Crop", max_show=8, 
    save_path="./output/debug_crop.png"
):
    import matplotlib
    matplotlib.use('Agg')  # backend sin display
    import matplotlib.pyplot as plt
    
    if original.dim() == 3:
        original = original.unsqueeze(0)
    if face_crop.dim() == 3:
        face_crop = face_crop.unsqueeze(0)
    
    B = min(original.shape[0], max_show)
    
    fig, axes = plt.subplots(B, 2, figsize=(6, 3 * B))
    if B == 1:
        axes = axes[None, :]
    
    for i in range(B):
        img_np = original[i].detach().cpu().permute(1, 2, 0).numpy()
        axes[i, 0].imshow(np.clip(img_np, 0, 1))
        axes[i, 0].set_title(f"Input [{i}]")
        axes[i, 0].axis("off")
        
        crop_np = face_crop[i].detach().cpu().permute(1, 2, 0).numpy()
        #crop_np = crop_np * 255.0
        #crop_np = crop_np.astype(np.uint8)
        axes[i, 1].imshow(np.clip(crop_np, 0, 1))
        axes[i, 1].set_title(f"Crop [{i}]")
        axes[i, 1].axis("off")
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[debug] saved → {save_path}")

def _to_hwc_uint8_for_viz(x: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
    """Convierte PIL/np/tensor a PIL RGB para visualizar (uint8), exprimiendo dims=1 si existen."""
    if isinstance(x, Image.Image):
        return x.convert("RGB")

    # -> numpy
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    elif not isinstance(x, np.ndarray):
        raise TypeError(f"Unsupported type: {type(x)}")

    # squeeze todas las dims==1 (maneja (1,1,H,W,3), (1,H,W,3), etc.)
    x = np.squeeze(x)

    # Si viene CHW, pásalo a HWC
    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[-1] != 3:
        x = np.transpose(x, (1, 2, 0))  # CHW -> HWC

    # Tras squeeze, esperamos HWC
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 after squeeze/transpose, got {x.shape}")

    # rango a [0,255] y uint8 SOLO para viz
    x = x.astype(np.float32)
    x = np.clip(x, 0, 255)
    if x.max() <= 1.0 + 1e-6:
        x = x * 255.0
    return Image.fromarray(x.astype(np.uint8), mode="RGB")

def show_input_vs_mtcnn_output(original: Union[Image.Image, np.ndarray, torch.Tensor],
                               face_tensor: torch.Tensor,
                               tag: str = "viz",
                               out_dir: str = "./output") -> str:
    """
    Visualiza lado a lado:
      - original (PIL/np/tensor)  -> convertido solo para visualizar (uint8)
      - face_tensor (salida de MTCNN) -> [3,S,S] o [1,3,S,S], en [0..1] o [0..255]
    Guarda un PNG y devuelve la ruta.
    """
    # add tag to the output directory
    Path(os.path.join(out_dir, tag)).mkdir(parents=True, exist_ok=True)

    # original -> PIL
    original_pil = _to_hwc_uint8_for_viz(original)

    # face -> PIL
    face = face_tensor.detach().cpu()
    if face.dim() == 4:
        face = face.squeeze(0)  # [3,S,S]
    if face.dim() == 3 and face.shape[0] in (1, 3):
        face_np = face.permute(1, 2, 0).numpy()  # HWC
    else:
        raise ValueError(f"Unexpected face tensor shape: {tuple(face.shape)}")

    face_np = np.clip(face_np, 0, 255).astype(np.float32)
    if face_np.max() <= 1.0 + 1e-6:
        face_np = face_np * 255.0
    face_pil = Image.fromarray(face_np.astype(np.uint8), mode="RGB")

    # mismo tamaño
    original_pil = original_pil.resize(face_pil.size)

    # concatenar
    side = Image.new("RGB", (face_pil.width * 2, face_pil.height))
    side.paste(original_pil, (0, 0))
    side.paste(face_pil, (face_pil.width, 0))

    # generate a unique id for the image
    id = np.random.randint(0, 1e6)
    out_path = f"{out_dir}/{tag}/{id}.png"
    side.save(out_path)
    #print(f"[viz] saved -> {out_path}")
    return out_path

def show_input_vs_mtcnn_output_old(original: Union[Image.Image, np.ndarray, torch.Tensor],
                               face_tensor: torch.Tensor,
                               tag: str = "viz",
                               out_dir: str = "./output") -> str:
    """
    Visualiza lado a lado:
      - original (PIL/np/tensor)  -> convertido solo para visualizar (uint8)
      - face_tensor (salida de MTCNN) -> [3,S,S] o [1,3,S,S], en [0..1] o [0..255]
    Guarda un PNG y devuelve la ruta.
    """
    # add tag to the output directory
    Path(os.path.join(out_dir, tag)).mkdir(parents=True, exist_ok=True)

    # 1) Original -> PIL uint8 SOLO para visualización
    if isinstance(original, torch.Tensor):
        x = original.detach().cpu()
        if x.dim() == 4 and x.shape[0] == 1:  # [1,H,W,3] o [1,3,H,W]
            x = x.squeeze(0)
        if x.dim() == 3 and x.shape[0] in (1, 3) and (x.shape[-1] != 3):
            x = x.permute(1, 2, 0)  # CHW -> HWC
        original_np = x.numpy()
    elif isinstance(original, Image.Image):
        original_np = np.array(original.convert("RGB"))
    else:  # numpy
        original_np = original

    original_np = original_np.astype(np.float32)
    original_np = np.clip(original_np, 0, 255)
    if original_np.max() <= 1.0 + 1e-6:  # si vino ya en 0..1
        original_np = (original_np * 255.0)
    original_pil = Image.fromarray(original_np.astype(np.uint8))

    # 2) face_tensor -> PIL uint8
    face = face_tensor.detach().cpu()
    if face.dim() == 4:
        face = face.squeeze(0)         # [3,S,S]
    if face.dim() != 3 or face.shape[0] not in (1, 3):
        raise ValueError(f"Unexpected face tensor shape: {tuple(face.shape)}")
    face_np = face.permute(1, 2, 0).numpy()  # HWC
    face_np = np.clip(face_np, 0, 255)
    if face_np.max() <= 1.0 + 1e-6:          # si vino en 0..1
        face_np = (face_np * 255.0)
    face_pil = Image.fromarray(face_np.astype(np.uint8))

    # 3) Alinear tamaños (redimensiono el original al tamaño del crop)
    original_pil = original_pil.resize(face_pil.size)

    # 4) Concatenar lado a lado
    side = Image.new("RGB", (face_pil.width * 2, face_pil.height))
    side.paste(original_pil, (0, 0))
    side.paste(face_pil, (face_pil.width, 0))
    # generate a unique id for the image
    id = np.random.randint(0, 1e6)
    out_path = f"{out_dir}/{tag}/{id}.png"
    side.save(out_path)
    #print(f"[viz] saved -> {out_path}")
    return out_path

def estimate_norm(lmk, image_size=160):
    assert lmk.shape == (5, 2)
    assert image_size%160==0 

    lmk = np.array(lmk, dtype=np.float64)

    ratio = float(image_size) / 112.0 # based on arcface
    dst = (arcface_dst * ratio).astype(np.float64)

    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M

def norm_crop_differentiable(img_tensor, landmark, image_size=112):
    """
    Differentiable version of norm_crop.
    """
    M = estimate_norm(landmark, image_size)  # forward: lmk → arcface_dst
    
    # cv2.warpAffine invierte M internamente — replicamos eso
    M_3x3 = np.vstack([M, [0, 0, 1]])
    M_inv = np.linalg.inv(M_3x3)[:2, :]  # inverse: arcface_dst → lmk
    
    C, H, W = img_tensor.shape
    device = img_tensor.device
    
    # Build output pixel grid
    ys = torch.arange(image_size, dtype=torch.float32, device=device)
    xs = torch.arange(image_size, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    ones = torch.ones_like(grid_x)
    dst_coords = torch.stack([grid_x, grid_y, ones], dim=-1)  # [H_out, W_out, 3]
    
    # Apply M_inv: src_pixel = M_inv @ dst_pixel
    M_inv_t = torch.tensor(M_inv, dtype=torch.float32, device=device)
    src_coords = torch.einsum('ij,hwj->hwi', M_inv_t, dst_coords)  # [H_out, W_out, 2]
    
    # Normalize to [-1, 1] for grid_sample
    src_coords[..., 0] = 2.0 * src_coords[..., 0] / (W - 1) - 1.0
    src_coords[..., 1] = 2.0 * src_coords[..., 1] / (H - 1) - 1.0
    
    grid = src_coords.unsqueeze(0)  # [1, H_out, W_out, 2]
    warped = F.grid_sample(img_tensor.unsqueeze(0), grid, 
                           mode='bilinear', padding_mode='zeros', 
                           align_corners=True)
    
    return warped.squeeze(0)

def crop_resize_differentiable(img_tensor, box, image_size):
    """
    Differentiable version of crop_resize.
    Square crop centered on face + resize via grid_sample.
    
    Args:
        img_tensor: [C, H, W] torch tensor (may carry grad)
        box: list/array [x1, y1, x2, y2] - constant from MTCNN
        image_size: output size (square)
    
    Returns:
        [C, image_size, image_size] torch tensor preserving gradient graph
    """
    C, H, W = img_tensor.shape
    
    # --- Replicate EXACT same crop geometry as original ---
    x1, y1, x2, y2 = map(int, box)
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    s = max(w, h)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0

    x0 = int(round(cx - s / 2.0))
    y0 = int(round(cy - s / 2.0))

    # Shift window to stay inside image (keeps square)
    x0 = min(max(0, x0), max(0, W - s))
    y0 = min(max(0, y0), max(0, H - s))
    
    # --- Compute theta: maps dst [-1,1] → src [-1,1] ---
    # dst pixel 0         → src pixel x0       → src_norm = 2*x0/(W-1) - 1
    # dst pixel size-1    → src pixel x0+s-1   → src_norm = 2*(x0+s-1)/(W-1) - 1
    # Linear map: src_norm = a * dst_norm + b
    
    a_x = (s - 1.0) / (W - 1.0)
    b_x = (2.0 * x0 + s - 1.0) / (W - 1.0) - 1.0
    a_y = (s - 1.0) / (H - 1.0)
    b_y = (2.0 * y0 + s - 1.0) / (H - 1.0) - 1.0
    
    theta = torch.tensor([
        [a_x,  0,   b_x],
        [0,    a_y, b_y]
    ], dtype=torch.float32, device=img_tensor.device).unsqueeze(0)  # [1, 2, 3]
    
    grid = F.affine_grid(theta, [1, C, image_size, image_size], 
                         align_corners=True)
    cropped = F.grid_sample(img_tensor.unsqueeze(0), grid, 
                            mode='bilinear', padding_mode='zeros', 
                            align_corners=True)
    
    return cropped.squeeze(0)  # [C, image_size, image_size]

def crop_resize(img, box, image_size):
    """
    box: (x1, y1, x2, y2) in pixel coords, x2/y2 exclusive-style is fine too (we resize anyway).
    img: numpy HWC, torch HWC or CHW, or PIL Image
    """

    x1, y1, x2, y2 = map(int, box)
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    s = max(w, h)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0

    # square window [x0, x0+s), [y0, y0+s)
    x0 = int(round(cx - s / 2.0))
    y0 = int(round(cy - s / 2.0))

    if isinstance(img, np.ndarray):
        H, W = img.shape[:2]
    elif isinstance(img, torch.Tensor):
        # accept HWC or CHW
        if img.ndim != 3:
            raise ValueError("torch img must be 3D (HWC or CHW)")
        if img.shape[0] in (1, 3, 4) and img.shape[2] not in (1, 3, 4):
            # CHW
            C, H, W = img.shape
            chw = True
        else:
            # HWC
            H, W, C = img.shape
            chw = False
    else:
        # PIL
        W, H = img.size

    # shift window to stay inside image (keeps square)
    x0 = min(max(0, x0), max(0, W - s))
    y0 = min(max(0, y0), max(0, H - s))
    x1n, y1n = x0 + s, y0 + s

    if isinstance(img, np.ndarray):
        crop = img[y0:y1n, x0:x1n]
        return cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA).copy()

    if isinstance(img, torch.Tensor):
        if chw:
            crop = img[:, y0:y1n, x0:x1n]
        else:
            crop = img[y0:y1n, x0:x1n, :]

        # simplest: use torch.nn.functional.interpolate on float
        if chw:
            crop_f = crop.unsqueeze(0).float()
        else:
            crop_f = crop.permute(2, 0, 1).unsqueeze(0).float()

        out = F.interpolate(crop_f, size=(image_size, image_size), mode="area")
        out = out.squeeze(0)
        if not chw:
            out = out.permute(1, 2, 0)
        return out.byte()

    # PIL
    crop = img.crop((x0, y0, x1n, y1n))
    return crop.resize((image_size, image_size), Image.BILINEAR)

def _extract_face_float(img, box, image_size=160, margin=0, save_path=None):
    """
    Float-safe replacement for extract_face.
    Matches original crop logic + uses INTER_AREA resize for consistency.
    """
    # --- Convert to numpy HWC float32 ---
    if isinstance(img, torch.Tensor):
        if img.dim() == 4:
            img = img.squeeze(0) 
        if img.dim() == 3 and img.shape[0] in (1, 3):
            img = img.permute(1, 2, 0).detach().cpu().numpy()
        elif img.dim() == 3 and img.shape[2] in (1, 3):
            img = img.detach().cpu().numpy()
        else:
            raise ValueError(f"Unsupported tensor shape: {tuple(img.shape)}")
    elif isinstance(img, np.ndarray):
        pass
    else:
        img = np.asarray(img, dtype=np.float32)

    img = img.astype(np.float32)
    h, w = img.shape[:2]

    # --- Margin: replicate EXACT original logic ---
    margin_adj = [
        margin * (box[2] - box[0]) / (image_size - margin),
        margin * (box[3] - box[1]) / (image_size - margin),
    ]
    x1 = int(max(box[0] - margin_adj[0] / 2, 0))
    y1 = int(max(box[1] - margin_adj[1] / 2, 0))
    x2 = int(min(box[2] + margin_adj[0] / 2, w))
    y2 = int(min(box[3] + margin_adj[1] / 2, h))

    box_margin = [x1, y1, x2, y2]
    #face_np = img[y1:y2, x1:x2, :]
    face_np = crop_resize(img, box_margin, image_size)

    if face_np.size == 0:
        raise ValueError(f"Empty face crop with box {box} and margin {margin}. Check the box coordinates and margin size.")
    
    face_t = torch.from_numpy(face_np.copy()).permute(2, 0, 1).float()

    return face_t

def _extract_face_differentiable_batch(imgs, boxes_list, landmarks_list, image_size=160, margin=0):
    """
    Differentiable batch face extraction.
    No numpy conversion — stays in torch preserving gradients.
    
    Args:
        imgs: [B, C, H, W] torch tensor (may carry grad from x_adv)
        boxes_list: list of B np.ndarray (4,) - constants from MTCNN
        landmarks_list: list of B np.ndarray (5,2) or None - constants from MTCNN
        image_size: output face size
        margin: margin around face (default 0)
    
    Returns:
        [B, C, image_size, image_size] torch tensor preserving gradient graph
    """
    B, C, H, W = imgs.shape
    faces = []
    
    for i in range(B):
        img = imgs[i]  # [C, H, W] — preserves grad via indexing
        
        if landmarks_list[i] is not None:
            face = norm_crop_differentiable(img, landmarks_list[i], image_size=image_size)
        else:
            margin_adj = [
                margin * (boxes_list[i][2] - boxes_list[i][0]) / (image_size - margin),
                margin * (boxes_list[i][3] - boxes_list[i][1]) / (image_size - margin),
            ]
            x1 = int(max(boxes_list[i][0] - margin_adj[0] / 2, 0))
            y1 = int(max(boxes_list[i][1] - margin_adj[1] / 2, 0))
            x2 = int(min(boxes_list[i][2] + margin_adj[0] / 2, W))
            y2 = int(min(boxes_list[i][3] + margin_adj[1] / 2, H))
            
            face = crop_resize_differentiable(img, [x1, y1, x2, y2], image_size)
        
        faces.append(face)
    
    return torch.stack(faces, dim=0)  # [B, C, image_size, image_size]

def _extract_face_float_v2(img, box, image_size=160, margin=0, save_path=None):
    """
    Float-safe replacement for extract_face.
    Matches original crop logic + uses symmetric pad/crop to reach image_size
    WITHOUT rescaling pixels (preserves original pixel values).
    """
    # --- Convert to numpy HWC float32 ---
    if isinstance(img, torch.Tensor):
        if img.dim() == 3 and img.shape[0] in (1, 3):
            img = img.permute(1, 2, 0).cpu().numpy()
        elif img.dim() == 3 and img.shape[2] in (1, 3):
            img = img.cpu().numpy()
        else:
            raise ValueError(f"Unsupported tensor shape: {tuple(img.shape)}")
    elif isinstance(img, np.ndarray):
        pass
    else:
        img = np.asarray(img, dtype=np.float32)

    img = img.astype(np.float32)
    h, w = img.shape[:2]

    # --- Margin: replicate EXACT original logic ---
    margin_adj = [
        margin * (box[2] - box[0]) / (image_size - margin),
        margin * (box[3] - box[1]) / (image_size - margin),
    ]
    x1 = int(max(box[0] - margin_adj[0] / 2, 0))
    y1 = int(max(box[1] - margin_adj[1] / 2, 0))
    x2 = int(min(box[2] + margin_adj[0] / 2, w))
    y2 = int(min(box[3] + margin_adj[1] / 2, h))

    face_np_old = img[y1:y2, x1:x2, :]

    if face_np_old.size == 0:
        raise ValueError(
            f"Empty face crop with box {box} and margin {margin}. "
            "Check the box coordinates and margin size."
        )

    # --- Symmetric pad or crop to reach image_size x image_size ---
    
    fh, fw = face_np_old.shape[:2]

    # Compute new crop window in the original image
    diff_h = image_size - fh
    diff_w = image_size - fw

    before_h = diff_h // 2
    after_h  = diff_h - before_h
    before_w = diff_w // 2
    after_w  = diff_w - before_w

    new_y1 = y1 - before_h
    new_y2 = y2 + after_h
    new_x1 = x1 - before_w
    new_x2 = x2 + after_w

    # Validate bounds before touching anything
    if new_y1 < 0 or new_y2 > h or new_x1 < 0 or new_x2 > w:
        raise ValueError(
            f"Cannot expand face crop to {image_size}x{image_size}: "
            f"requested y=[{new_y1}:{new_y2}] x=[{new_x1}:{new_x2}] "
            f"exceeds image bounds [0:{h}] x [0:{w}]."
        )

    face_np = img[new_y1:new_y2, new_x1:new_x2, :]

    assert face_np.shape[:2] == (image_size, image_size), (
        f"Shape mismatch after pad/crop: got {face_np.shape[:2]}, "
        f"expected ({image_size}, {image_size})"
    )

    face_t = torch.from_numpy(face_np.copy()).permute(2, 0, 1).float()
    return face_t

def _PIL_numpy_to_tensor(img_any, to_CHW: bool = False) -> torch.Tensor:
    """
    Convert input image to float32 tensor WHC in [0,255] WITHOUT quantizing.
    - If PIL: converts to numpy float (already 8-bit source, but we don't re-quantize)
    - If numpy HWC: just wraps
    """
    if isinstance(img_any, Image.Image):
        # PNG path: source is 8-bit, but we keep it float afterwards
        arr = np.array(img_any.convert('RGB'), dtype=np.float32)  # H,W,3 in [0,255]
        if to_CHW:
            arr = np.transpose(arr, (2, 0, 1))[np.newaxis]  # 1,C,H,W
        ten = torch.from_numpy(arr).float()      
        return ten

    if isinstance(img_any, np.ndarray):
        arr = img_any
        arr = arr.astype(np.float32)   
        if to_CHW:
            arr = np.transpose(arr, (2, 0, 1))[np.newaxis]  # 1,C,H,W
        ten = torch.from_numpy(arr).float()       # H,W,3
        return ten

    raise TypeError(f"Unsupported type: {type(img_any)}")

def preprocess_for_facenet(img_any, to_CHW: bool = False, TARGET: tuple = (160, 160), device: str = 'cpu') -> torch.Tensor:
    """
    Returns a tensor [1,3,S,S] ready for FaceNet:
    - Float32 in [0,1] after fixed_image_standardization
    """
    # check if the img_any is a Pil image or a numpy array, and convert to tensor
    if isinstance(img_any, (Image.Image, np.ndarray)):
        # 1) To CHW float [0,255]
        img_any = _PIL_numpy_to_tensor(img_any, to_CHW=to_CHW)  # [1,3,H,W], float32
        #print('preprocess_for_facenet - after to_tensor, before standardization: size:', chw.shape, 'dtype:', chw.dtype, 'min:', chw.min().item(), 'max:', chw.max().item())
        #chw = chw.unsqueeze(0)  # [1,3,H,W]
    
    # 2) standardize (FaceNet expects fixed_image_standardization)   
    if img_any.max() <= 1.0:
        img_any = img_any * 255.0 # fixed image standardization waits for [0,255] input

    #print('range before standardization:', img_any.min().item(), img_any.max().item())
    #print("shape before standardization:", img_any.shape)
    
    chw_std = fixed_image_standardization(img_any)  # [3,S,S], float32
    #print('preprocess_for_facenet - after standardization: size:', chw_std.shape, 'dtype:', chw_std.dtype, 'min:', chw_std.min().item(), 'max:', chw_std.max().item())
    # 4) resize
    #if chw_std.shape[-2:] != torch.Size(list(TARGET)):
    #    chw_std = F.interpolate(
    #        chw_std,
    #        size=TARGET,
    #        mode='bilinear',
    #        align_corners=False
    #    )       
    return chw_std.to(device)  # ready for FaceNet

class FaceNetRecognizer:
    """
    This service extracts facial embeddings using a pre-trained FaceNet model.
    """
    IMG_SIZE = 160  # facenet native resolution

    def __init__(self, device: str = 'cpu', image_format: str = 'png', use_mtcnn: bool = True,
                 save_images_path: Union[str, Path] = None):

        self.device = torch.device(device)
        print(f"Initializing FaceNetRecognizer on device: {self.device}")
        # model
        self.model = InceptionResnetV1(pretrained='vggface2', classify=False).eval().to(self.device)
        # path to save visualization images
        self.save_images_path = save_images_path
        # for online and offline tests
        self.image_format = image_format
        self.use_mtcnn = use_mtcnn
        # --- MTCNN (same config as FaceNet service, output size = 160) ---
        self.mtcnn = None
        if use_mtcnn:
            self.mtcnn = MTCNN(
                image_size=self.IMG_SIZE,  # crop directly to facenet input size
                margin=0,
                keep_all=False,            
                post_process=False,         # output tensor in [0, 255]
                device=self.device
            )

    # --------------------------------------------------------------------- #
    # Box detection (runs MTCNN detection only, no embedding)
    # --------------------------------------------------------------------- #
    def detect_box(self, img) -> Optional[np.ndarray]:
        """
        Run MTCNN face detection and return the bounding box.
        For npy inputs, converts to uint8 PIL for detection only
        (the box coordinates are what we need, not the pixel values).
        
        Returns:
            np.ndarray of shape (4,) with [x1, y1, x2, y2] or None.
        """
        if self.mtcnn is None:
            return None
        
        if img.max() > 1.0:
            img = img/255.0
        
        # Convert to PIL for detection (MTCNN.detect expects PIL or uint8)
        if isinstance(img, Image.Image):
            detect_img = img
        if isinstance(img, torch.Tensor):
            detect_img = torchvision.transforms.ToPILImage()(img.squeeze(0))
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")
        
        boxes, _ = self.mtcnn.detect(detect_img)
        
        if boxes is not None and len(boxes) > 0:
            return boxes[0]  # first (most prominent) face
        return None    

    def detect_boxes_batch(self, imgs) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
        """
        Run MTCNN face detection on a batch of images.
        
        Args:
            imgs: torch.Tensor of shape [B, C, H, W] normalized [0, 1] or [0, 255]
        
        Returns:
            Tuple of (boxes_list, landmarks_list) where each is a list of B elements.
            Each element is np.ndarray or None if no face detected.
            boxes: shape (4,) with [x1, y1, x2, y2]
            landmarks: shape (5, 2) with 5 facial keypoints
        """
        if self.mtcnn is None:
            return [None] * imgs.shape[0], [None] * imgs.shape[0]
        
        B = imgs.shape[0]
        
        # Normalize if needed because our standard is always a 0-1 range
        if imgs.max() > 1.0:
            imgs = imgs / 255.0
        
        # Convert batch to list of PIL images
        pil_imgs = [torchvision.transforms.ToPILImage()(imgs[i]) for i in range(B)]
        
        # Batch detection
        boxes_batch, _, landmarks_batch = self.mtcnn.detect(pil_imgs, landmarks=True)
        
        # Process results: extract first (most prominent) face per image
        boxes_list = []
        landmarks_list = []
        
        for i in range(B):
            if boxes_batch[i] is not None and len(boxes_batch[i]) > 0:
                boxes_list.append(boxes_batch[i][0])  # first face box
                landmarks_list.append(landmarks_batch[i][0])  # first face landmarks
            else:
                boxes_list.append(None)
                landmarks_list.append(None)
        
        return boxes_list, landmarks_list
    
    # --------------------------------------------------------------------- #
    # Embed with precomputed box (float-safe, no uint8 quantization)
    # --------------------------------------------------------------------- #
    def _embed_with_box(self, img, box: np.ndarray, debug_img: bool = False, origin: str = "original") -> torch.Tensor:
        """
        Crop the face using a precomputed bounding box via _extract_face_float
        (preserves float32 precision), then run through the ArcFace backbone.
        """
        face_tensor = _extract_face_float(img, box, image_size=self.IMG_SIZE, margin=0)
        #print(f"Box reuse path: extracted face tensor shape: {face_tensor.shape}, dtype: {face_tensor.dtype}, min: {face_tensor.min().item()}, max: {face_tensor.max().item()}")
        
        #print(f"debug_img: {debug_img}, save_images_path: {self.save_images_path}, origin: {origin}")
        if debug_img and self.save_images_path is not None:
            #print(f"[debug_img] Box reuse path: visualizing original vs MTCNN crop for {origin} image")
            tag = f'PIL_box_reuse-{origin}' if isinstance(img, Image.Image) else f'NPY_box_reuse-{origin}'
            show_input_vs_mtcnn_output(original=img, face_tensor=face_tensor, 
                                       tag=tag, out_dir=self.save_images_path)
        
        tensor = face_tensor.unsqueeze(0).to(self.device)
        #print(f"Box reuse path: tensor shape before preprocess_for_facenet: {tensor.shape}, dtype: {tensor.dtype}, min: {tensor.min().item()}, max: {tensor.max().item()}")
        tensor = preprocess_for_facenet(tensor, to_CHW=False, TARGET=(self.IMG_SIZE, self.IMG_SIZE))
        #print(f"Box reuse path: tensor shape after preprocess_for_facenet: {tensor.shape}, dtype: {tensor.dtype}, min: {tensor.min().item()}, max: {tensor.max().item()}")
        
        with torch.no_grad():
            embedding = self.model(tensor)
        
        return embedding.squeeze(0)  
    
    def embed_batch_with_boxes(self, imgs, boxes_list, landmarks_list, requires_grad=True):
        """
        Batch embedding with optional gradient flow.
        
        Args:
            imgs: [B, C, H, W] torch tensor
            boxes_list: list of B boxes
            landmarks_list: list of B landmarks
            requires_grad: If True, preserves gradients (for PGD loop)
        
        Returns: [B, 512] embeddings
        """
        # Crop diferenciable 
        faces = _extract_face_differentiable_batch(
            imgs, boxes_list, landmarks_list, image_size=self.IMG_SIZE, margin=0
        )  # [B, C, H, W] #[None,None] to test without landmarks
        
        # debug
        #show_input_vs_crop_interactive(imgs, faces, title="face verification batch crop")

        # Preprocess
        faces = preprocess_for_facenet(faces, device=self.device, 
                                    TARGET=(self.IMG_SIZE, self.IMG_SIZE))
        
        # Forward con o sin gradientes
        if requires_grad:
            embeddings = self.model(faces)  # SIN torch.no_grad()
        else:
            with torch.no_grad():
                embeddings = self.model(faces)
        
        return embeddings  # [B, 512]    

    # --------------------------------------------------------------------- #
    # get_embedding_and_box: detect + embed, return both
    # --------------------------------------------------------------------- #
    def get_embedding_and_box(self, img, debug_img: bool = False, origin: str = "original") -> Tuple[Optional[torch.Tensor], Optional[np.ndarray]]:
        """
        Detect face with MTCNN, compute embedding, and return both the
        embedding and the bounding box for later reuse on watermarked images.
        
        Returns:
            (embedding, box) — embedding is 512-d tensor, box is np.ndarray(4,)
        """
        if not self.use_mtcnn:
            # No MTCNN: just embed, no box
            emb = self.get_embedding(img)
            return emb, None
        
        # Detect box
        box = self.detect_box(img)
        if box is None:
            raise ValueError("MTCNN failed to detect a face in the image.")
        
        # Now run get embedding with the precomputed box (float-safe)
        embedding = self.get_embedding(img, debug_img=debug_img, precomputed_box=box, origin=origin)
        
        return embedding.squeeze(0), box      
    
    # --------------------------------------------------------------------- #
    # get_embedding: original method, now with optional precomputed_box
    # --------------------------------------------------------------------- #
    def get_embedding(self, img, debug_img: bool = False, 
                      precomputed_box: Optional[np.ndarray] = None,
                      origin: str = "original"):
        """
        Return a 512-d embedding.
        
        If precomputed_box is provided and use_mtcnn=True, the box is used
        to crop the face directly (float-safe), bypassing MTCNN detection
        and its internal uint8 quantization.
        """
        # --- Box reuse path: float-safe crop ---
        if precomputed_box is not None and self.use_mtcnn:
            return self._embed_with_box(img, precomputed_box, debug_img=debug_img, origin=origin)
        
        tensor = preprocess_for_facenet(img, to_CHW=True, 
                                            TARGET=(self.IMG_SIZE, self.IMG_SIZE))
                 
        with torch.no_grad():
            embedding = self.model(tensor)
 
        return embedding.squeeze(0) 

    
    def get_distance(self, emb1: torch.Tensor, emb2: torch.Tensor, metric: str) -> float:
        """
        Calculate the distance between two facial embeddings.
        
        Args:
            emb1: First embedding tensor.
            emb2: Second embedding tensor.
            metric: Distance metric to use ('euclidean' or 'cosine').
        
        Returns:
            Distance as a float.
        """
        if emb1.shape != emb2.shape:
            raise ValueError("Embeddings must have the same shape.")
        
        # Calculate the distance
        emb1_norm = emb1 / emb1.norm(p=2, dim=0, keepdim=True)
        emb2_norm = emb2 / emb2.norm(p=2, dim=0, keepdim=True)
        if metric == 'cosine':
            # Cosine distance
            #cosine_similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0), dim=0)
            cosine_similarity = torch.dot(emb1_norm, emb2_norm).item()
            distance = 1 - cosine_similarity
        elif metric == 'euclidean':
            # Euclidean distance
            distance = torch.norm(emb1_norm - emb2_norm).item()
        else:
            raise ValueError("Unsupported metric. Use 'euclidean' or 'cosine'.")
        
        return distance