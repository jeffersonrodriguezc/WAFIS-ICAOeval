# modules/recognition/FaceNet/facenet_recognizer.py
import torch
from typing import Optional, Tuple, Union
from facenet_pytorch import InceptionResnetV1, MTCNN, fixed_image_standardization
 # InceptionResnetV1 as a facenet backbone
from PIL import Image
from torchvision import transforms
from facenet_pytorch.models import mtcnn as mtcnn_mod
import numpy as np
import os 
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor as tv_to_tensor

# utils_viz_mtcnn.py
from pathlib import Path
from typing import Union
import numpy as np
import torch
from PIL import Image
import cv2

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

def _extract_face_float_old(img, box, image_size=160, margin=0, save_path=None):
    """
    Float-safe replacement for extract_face.
    Matches original crop logic + uses INTER_AREA resize for consistency.
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

    face_np = img[y1:y2, x1:x2, :]

    if face_np.size == 0:
        face_np = np.zeros((image_size, image_size, 3), dtype=np.float32)
        face_t = torch.from_numpy(face_np).permute(2, 0, 1)
        raise ValueError(f"Empty face crop with box {box} and margin {margin}. Check the box coordinates and margin size.")
    else:
        # INTER_AREA to match original cv2.resize behavior
        face_np = cv2.resize(
            face_np,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA
        )
        face_t = torch.from_numpy(face_np.copy()).permute(2, 0, 1).float()

    return face_t

def _extract_face_float(img, box, image_size=160, margin=0, save_path=None):
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

    face_np = img[y1:y2, x1:x2, :]

    if face_np.size == 0:
        raise ValueError(
            f"Empty face crop with box {box} and margin {margin}. "
            "Check the box coordinates and margin size."
        )

    # --- Symmetric pad or crop to reach image_size x image_size ---
    fh, fw = face_np.shape[:2]

    def _pad_or_crop_axis(arr, current, target, axis, img_full, offset):
        """
        Expand (using original image pixels) or crop symmetrically along one axis.
        axis: 0 = height (y), 1 = width (x)
        offset: y1 or x1 (position of the crop in the full image)
        """
        diff = target - current
        if diff == 0:
            return arr

        if diff > 0:
            # Need to expand: pull pixels from the original image
            before = diff // 2
            after  = diff - before
            if axis == 0:
                new_start = offset - before
                new_end   = offset + current + after
                if new_start < 0 or new_end > img_full.shape[0]:
                    raise ValueError(
                        f"Cannot expand face crop along axis {axis}: "
                        f"requested [{new_start}:{new_end}] exceeds image bounds [0:{img_full.shape[0]}]."
                    )
                return img_full[new_start:new_end, :, :]
            else:
                new_start = offset - before
                new_end   = offset + current + after
                if new_start < 0 or new_end > img_full.shape[1]:
                    raise ValueError(
                        f"Cannot expand face crop along axis {axis}: "
                        f"requested [{new_start}:{new_end}] exceeds image bounds [0:{img_full.shape[1]}]."
                    )
                return img_full[:, new_start:new_end, :]
        else:
            # Need to crop: remove symmetrically from both sides
            remove = -diff
            before = remove // 2
            after  = remove - before
            if axis == 0:
                return arr[before:current - after, :, :]
            else:
                return arr[:, before:current - after, :]

    # Apply along height, then width
    face_np = _pad_or_crop_axis(face_np, fh, image_size, axis=0, img_full=img, offset=y1)
    face_np = _pad_or_crop_axis(face_np, fw, image_size, axis=1, img_full=img, offset=x1)

    assert face_np.shape[:2] == (image_size, image_size), (
        f"Shape mismatch after pad/crop: got {face_np.shape[:2]}, expected ({image_size}, {image_size})"
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

def preprocess_for_facenet(img_any, to_CHW: bool = False, TARGET: tuple = (160, 160)) -> torch.Tensor:
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
    chw_std = fixed_image_standardization(img_any)  # [3,S,S], float32
    #print('preprocess_for_facenet - after standardization: size:', chw_std.shape, 'dtype:', chw_std.dtype, 'min:', chw_std.min().item(), 'max:', chw_std.max().item())
    # 4) resize
    if chw_std.shape[-2:] != torch.Size(list(TARGET)):
        chw_std = F.interpolate(
            chw_std,
            size=TARGET,
            mode='bilinear',
            align_corners=False
        )       
    return chw_std  # ready for FaceNet

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
        
        # Convert to PIL for detection (MTCNN.detect expects PIL or uint8)
        if isinstance(img, Image.Image):
            detect_img = img
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")
        
        boxes, _ = self.mtcnn.detect(detect_img)
        
        if boxes is not None and len(boxes) > 0:
            return boxes[0]  # first (most prominent) face
        return None    
    
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