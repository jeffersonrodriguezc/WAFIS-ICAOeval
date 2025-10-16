# modules/recognition/FaceNet/facenet_recognizer.py
import torch
from typing import Union
from facenet_pytorch import InceptionResnetV1, MTCNN, fixed_image_standardization
 # InceptionResnetV1 as a facenet backbone
from PIL import Image
from torchvision import transforms
from facenet_pytorch.models import mtcnn as mtcnn_mod
import numpy as np
import os 
import torch.nn.functional as F

# utils_viz_mtcnn.py
from pathlib import Path
from typing import Union
import numpy as np
import torch
from PIL import Image

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
        raise ValueError(f"Unexpected face shape: {tuple(face.shape)}")

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

def _extract_face_float(img, box, image_size=160, margin=0, save_path=None):
    """
    Replacement for facenet_pytorch.models.utils.detect_face.extract_face
    Accepts float32 arrays/tensors without quantizing to uint8.
    - img: np.ndarray HWC float32 in [0,255], or torch.Tensor CHW/HWC float32 in [0,255]
    - box: [x1, y1, x2, y2]
    Returns: torch.Tensor [3, image_size, image_size] float32 in [0,255]
    """
    # to numpy HWC float32
    if isinstance(img, torch.Tensor):
        if img.dim() == 3 and img.shape[0] in (1, 3):      # CHW
            img = img.permute(1, 2, 0).cpu().numpy()
        elif img.dim() == 3 and img.shape[2] in (1, 3):    # HWC
            img = img.cpu().numpy()
        else:
            raise ValueError(f"Unsupported tensor shape: {tuple(img.shape)}")
    elif isinstance(img, np.ndarray):
        pass  # already fine
    else:
        # likely PIL.Image – fallback (esto cuantiza, evítalo en la ruta npy)
        img = np.asarray(img, dtype=np.float32)

    img = img.astype(np.float32)
    h, w = img.shape[:2]

    x1, y1, x2, y2 = [float(b) for b in box]
    if isinstance(margin, int):
        mx = my = margin
    else:
        mx, my = margin

    x1 = max(0.0, x1 - mx / 2.0)
    y1 = max(0.0, y1 - my / 2.0)
    x2 = min(w,   x2 + mx / 2.0)
    y2 = min(h,   y2 + my / 2.0)

    x1i, y1i, x2i, y2i = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    face_np = img[y1i:y2i, x1i:x2i, :]  # H',W',C

    if face_np.size == 0:
        face_np = np.zeros((image_size, image_size, 3), dtype=np.float32)
        face_t = torch.from_numpy(face_np).permute(2, 0, 1)  # C,H,W
    else:
        face_t = torch.from_numpy(face_np).permute(2, 0, 1).unsqueeze(0).float()  # 1,C,H,W
        face_t = F.interpolate(face_t, size=(image_size, image_size),
                               mode='bilinear', align_corners=False)
        face_t = face_t.squeeze(0)  # C,H,W

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
            arr = np.transpose(arr, (2, 0, 1))  # C,H,W
        ten = torch.from_numpy(arr).float()      # H,W,3
        return ten

    if isinstance(img_any, np.ndarray):
        arr = img_any
        arr = arr.astype(np.float32)   
        if to_CHW:
            arr = np.transpose(arr, (2, 0, 1)) # keep decimals
        ten = torch.from_numpy(arr).float()       # H,W,3
        return ten

    raise TypeError(f"Unsupported type: {type(img_any)}")

def preprocess_for_facenet(img_any, to_CHW: bool = False) -> torch.Tensor:
    """
    Returns a tensor [1,3,S,S] ready for FaceNet:
    - Float32 in [0,1] after fixed_image_standardization
    """
    # 1) To CHW float [0,255]
    chw = _PIL_numpy_to_tensor(img_any, to_CHW=to_CHW)  # [1,3,H,W], float32
    #chw = chw.unsqueeze(0)  # [1,3,H,W]
    # 2) standardize (FaceNet expects fixed_image_standardization)   
    chw_std = fixed_image_standardization(chw)  # [3,S,S], float32
        
    return chw_std  # ready for FaceNet

class FaceNetRecognizer:
    """
    This service extracts facial embeddings using a pre-trained FaceNet model.
    """
    def __init__(self, device: str = 'cpu', image_format: str = 'png', use_mtcnn: bool = True,
                 save_images_path: Union[str, Path] = None):

        self.device = torch.device(device)
        print(f"Initializing FaceNetRecognizer on device: {self.device}")
        # model
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        # path to save visualization images
        self.save_images_path = save_images_path
        # for online and offline tests
        self.image_format = image_format
        self.use_mtcnn = use_mtcnn
        # MTCNN for face detection and alignment in the wild
        if use_mtcnn: 
            if image_format == 'png':
                self.mtcnn = MTCNN(keep_all=False, device=self.device)
            else:
                mtcnn_mod.extract_face = _extract_face_float
                self.mtcnn = mtcnn_mod.MTCNN(keep_all=False, device=self.device)

    def get_embedding(self, image, debug_img: bool = False) -> Union[torch.Tensor, None]:
        """
        Calculate the facial embedding for a given image input.
        Args:
            image: PIL Image (Image.Image), or an array NumPy (np.ndarray),
        
        Returns:
            Pytorch Tensor with the facial embedding, or None if the input is invalid.
        """
        if self.use_mtcnn:
            
            # Detect and align the face using MTCNN  
            if isinstance(image, Image.Image): # png
                image_face = self.mtcnn(image)
                if debug_img and self.save_images_path is not None:
                    show_input_vs_mtcnn_output(original=image, face_tensor=image_face, tag='PIL',
                                           out_dir=self.save_images_path)
            
            else: # npy
                image = image[None, ...] # Add batch dimension
                image_face = self.mtcnn(image)
                # return the first face
                image_face = image_face[0] if image_face is not None else None
                if debug_img and self.save_images_path is not None:
                    show_input_vs_mtcnn_output(original=image, face_tensor=image_face, tag='NPY',
                                           out_dir=self.save_images_path)

            if image_face is None:
                print("No face detected in the image.")
                return None
            
            image_face = image_face.unsqueeze(0).to(self.device)  # Add batch dimension [1,3,S,S]
            
        else:
            image_face = preprocess_for_facenet(image, to_CHW=True)
            # This no apply face detection nither alignment nither cropping
            # Only channel permutation and standardization
            image_face = image_face.unsqueeze(0).to(self.device)  # Add batch dimension [1,3,S,S]

        # Calculate the embedding using the FaceNet model
        with torch.no_grad():
            embedding = self.model(image_face)
        
        # The embedding is a 512-dimensional vector
        return embedding.squeeze(0) # Delete the batch dimension for a single image embedding

    
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
        if metric == 'cosine':
            # Cosine distance
            emb1_norm = emb1 / emb1.norm(p=2, dim=0, keepdim=True)
            emb2_norm = emb2 / emb2.norm(p=2, dim=0, keepdim=True)
            #cosine_similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0), dim=0)
            cosine_similarity = torch.dot(emb1_norm, emb2_norm).item()
            distance = 1 - cosine_similarity
        elif metric == 'euclidean':
            # Euclidean distance
            distance = torch.norm(emb1 - emb2).item()
        else:
            raise ValueError("Unsupported metric. Use 'euclidean' or 'cosine'.")
        
        return distance