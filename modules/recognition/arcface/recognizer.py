import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from facenet_pytorch import MTCNN
from facenet_pytorch.models import mtcnn as mtcnn_mod

from typing import Union
from PIL import Image

# ---------------------------------------------------------------------------
# IResNet backbone — inlined, no dependency on cloned repo
# Matches arcface_torch/backbones/iresnet.py exactly
# ---------------------------------------------------------------------------

from torch import nn

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

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

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

def preprocess_for_arcface(
    array,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Resize to 112×112 and normalise to [-1, 1] for ArcFace inference.
    Official pipeline: div(255).sub(0.5).div(0.5)

    Accepts:
        - PIL.Image.Image              any mode  → output (1, 3, 112, 112)
        - np.ndarray  HWC  float32/uint8  [0, 255]  → output (1, 3, 112, 112)
        - np.ndarray  BHWC float32/uint8  [0, 255]  → output (B, 3, 112, 112)
        - torch.Tensor (3, H, W)    float32 [0,255] → output (1, 3, 112, 112)
        - torch.Tensor (B, 3, H, W) float32 [0,255] → output (B, 3, 112, 112)

    Returns:
        torch.Tensor (B, 3, 112, 112) in [-1, 1], always with batch dimension
    """
    TARGET = (112, 112)

    if isinstance(array, Image.Image):
        # 1) PIL → numpy HWC float32 [0, 255]
        arr = np.array(array.convert('RGB'), dtype=np.float32)  # (H, W, 3)

        # 2) normalise
        arr = arr / 255.0
        arr = (arr - 0.5) / 0.5                     # [-1, 1]

        # 3) HWC → BCHW
        arr = arr.transpose(2, 0, 1)[np.newaxis]    # (1, 3, H, W)
        tensor = torch.from_numpy(arr)

        # 4) resize
        if tensor.shape[-2:] != torch.Size(list(TARGET)):
            tensor = F.interpolate(
                tensor,
                size=TARGET,
                mode='bilinear',
                align_corners=False
            )                                       # (1, 3, 112, 112)

    elif isinstance(array, np.ndarray):
        arr = array.astype(np.float32)

        # 1) ensure batch dimension: HWC → BHWC
        if arr.ndim == 3:
            arr = arr[np.newaxis]                   # (1, H, W, C)
        elif arr.ndim != 4:
            raise ValueError(f"numpy input must be HWC or BHWC, got shape: {array.shape}")

        # 2) normalise
        arr = arr / 255.0
        arr = (arr - 0.5) / 0.5                     # [-1, 1]

        # 3) BHWC → BCHW → torch tensor
        arr = arr.transpose(0, 3, 1, 2)             # (B, 3, H, W)
        tensor = torch.from_numpy(arr)

        # 4) resize
        if tensor.shape[-2:] != torch.Size(list(TARGET)):
            tensor = F.interpolate(
                tensor,
                size=TARGET,
                mode='bilinear',
                align_corners=False
            )                                       # (B, 3, 112, 112)

    elif isinstance(array, torch.Tensor):
        tensor = array.float()

        # 1) ensure batch dimension: CHW → BCHW
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)            # (1, 3, H, W)
        elif tensor.ndim != 4:
            raise ValueError(f"tensor input must be CHW or BCHW, got shape: {array.shape}")

        # 2) normalise
        tensor = tensor / 255.0
        tensor = (tensor - 0.5) / 0.5              # [-1, 1]

    else:
        raise TypeError(
            f"Unsupported type: {type(array)}. Use PIL.Image, np.ndarray or torch.Tensor"
        )

    return tensor.to(device)
    
class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('IBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in IBasicBlock")
        self.bn1   = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2   = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3   = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

class IResNet(nn.Module):
    fc_scale = 7 * 7

    def __init__(self, block, layers, dropout=0, num_features=512,
                 zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, fp16=False):
        super().__init__()
        self.fp16      = fp16
        self.inplanes  = 64
        self.dilation  = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups     = groups
        self.base_width = width_per_group
        self.conv1  = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu  = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn2      = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout  = nn.Dropout(p=dropout, inplace=True)
        self.fc       = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample        = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )
        layers = [block(self.inplanes, planes, stride, downsample,
                        self.groups, self.base_width, previous_dilation)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = x.float()
        x = self.fc(x)
        x = self.features(x)
        return x

def iresnet50(**kwargs):
    return IResNet(IBasicBlock, [3, 4, 14, 3], **kwargs)

def iresnet100(**kwargs):
    return IResNet(IBasicBlock, [3, 13, 30, 3], **kwargs)

MODELS = {'r50': iresnet50, 'r100': iresnet100}

# ---------------------------------------------------------------------------
# ArcFaceRecognizer
# ---------------------------------------------------------------------------
class ArcFaceRecognizer:
    """
    ArcFace embedding extractor with the same public interface as
    FaceNetRecognizer (get_embedding / get_distance).

    MTCNN behaviour
    ---------------
    When use_mtcnn=True the MTCNN from facenet_pytorch is used identically
    to how FaceNet uses it: it detects and crops the face region from the
    full image before passing it to the backbone.

    MTCNN (facenet_pytorch, post_process=True) returns a float32 tensor in
    [-1, 1]. ArcFace expects exactly the same normalisation range
    (div/255 → sub/0.5 → div/0.5), so the tensor is fed directly to the
    backbone without any additional normalisation step.

    When use_mtcnn=False the image is resized to 112×112 and normalised via
    _to_tensor(), replicating the official arcface_torch/inference.py pipeline.

    Args
    ----
    weight_path      : path to pretrained backbone .pth (state_dict)
    network          : 'r50' (iresnet50) | 'r100' (iresnet100)
    device           : 'cpu' | 'cuda' | 'cuda:0' …
    image_format     : 'png' (PIL Image) | 'npy' (float32 ndarray)
    use_mtcnn        : detect + crop face with MTCNN before embedding
    save_images_path : optional directory to save debug crops
    """

    IMG_SIZE = 112  # ArcFace native resolution

    def __init__(self, weight_path: str, network: str = 'r50',
                 device: str = 'cpu', image_format: str = 'png',
                 use_mtcnn: bool = False, save_images_path=None):

        self.device       = torch.device(device)
        self.image_format = image_format
        self.use_mtcnn    = use_mtcnn
        self.save_images_path = Path(save_images_path) if save_images_path else None

        if network not in MODELS:
            raise ValueError(f"network must be one of {list(MODELS.keys())}, got '{network}'")

        # --- backbone ---
        self.net = MODELS[network](num_features=512)
        state = torch.load(weight_path, map_location=self.device)
        # arcface_torch saves plain state_dicts; handle wrapped checkpoints too
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        self.net.load_state_dict(state)
        self.net.to(self.device)
        self.net.eval()

        # --- MTCNN (same config as FaceNet service, output size = 112) ---
        self.mtcnn = None
        if use_mtcnn:
            if image_format == 'png':
                self.mtcnn = MTCNN(
                    image_size=self.IMG_SIZE,  # crop directly to ArcFace input size
                    margin=0,
                    keep_all=False,            # most prominent face only
                    post_process=False,         # output tensor in [-1, 1]
                    device=self.device
                )
            else:
                mtcnn_mod.extract_face = _extract_face_float
                self.mtcnn = MTCNN(
                    image_size=self.IMG_SIZE,  # crop directly to ArcFace input size
                    margin=0,
                    keep_all=False,            # most prominent face only
                    post_process=False,         # output tensor in [-1, 1]
                    device=self.device
                )

    def get_embedding(self, img, debug_img: bool = False):
        """
        Return a 512-d L2-normalised embedding. 
        """
        if self.use_mtcnn:

            if isinstance(img, Image.Image): # png
                # --- MTCNN path (identical behaviour to FaceNet service) ---
                img_tensor = self.mtcnn(img)        # (3, 112, 112) in [0, 255]
                if debug_img and self.save_images_path is not None:
                    show_input_vs_mtcnn_output(original=img, face_tensor=img_tensor, tag='PIL',
                                        out_dir=self.save_images_path)
                
            else:
                img = img[None, ...] # Add batch dimension
                img_tensor = self.mtcnn(img)
                # return the first face
                img_tensor = img_tensor[0] if img_tensor is not None else None
                if debug_img and self.save_images_path is not None:
                    show_input_vs_mtcnn_output(original=torch.from_numpy(img), face_tensor=img_tensor, tag='NPY',
                                        out_dir=self.save_images_path)
                
            if img_tensor is None:
                print("No face detected in the image.")
                return None
                
            tensor = img_tensor.unsqueeze(0).to(self.device)
            # normalisation to [-1,1]
            tensor = preprocess_for_arcface(tensor, device=self.device)

        else:
            tensor = preprocess_for_arcface(img, device=self.device)  # (3,112,112) in [-1, 1]
                 
        # compute the embedding    
        with torch.no_grad():
            embedding = self.net(tensor)

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