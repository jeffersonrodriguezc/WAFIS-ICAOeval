import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from criteria.lpips.lpips import LPIPS

class AdvLoss(nn.Module):
    def __init__(self, adv_weight, device, mode='evasion'):
        """
        Args:
            adv_weight (float): Weight for the adversarial loss component in the total loss.
            device (str): 'cuda' or 'cpu'.
            mode (str): 'evasion' (Untargeted) or 'targeted'.
        """
        super(AdvLoss, self).__init__()
        self.adv_weight = adv_weight
        self.device = device
        self.mode = mode
        
        # Cosine loss is used for both evasion and targeted, but with different target labels.
        # For evasion, we want to minimize similarity (target = -1). 
        # For targeted, we want to maximize similarity (target = 1).
        self.cosine_loss = nn.CosineEmbeddingLoss()

    def forward(self, v_gen, v_ref):
        """
        Args:
            v_gen (Tensor): The facial vector extracted from the generated/attacked image. 
                            Must have active gradients (requires_grad=True in the previous flow).
            v_ref (Tensor): The reference facial vector (original or target).
        """
        # Get the batch size (N)
        batch_size = v_gen.size(0)
        
        loss_val = 0.0
        
        if self.mode == 'evasion':
            # OBJECTIVE: EVASION (Untargeted)
            # Target = -1 -> Forces the vectors to be opposite (minimizes similarity)
            target_label = -1 * torch.ones(batch_size).to(self.device)
            loss_val = self.cosine_loss(v_gen, v_ref, target_label)
            
        elif self.mode == 'targeted':
            # OBJECTIVE: TARGETED (Targeted)
            # Target = 1 -> Forces the vectors to be the same (maximizes similarity)
            target_label = torch.ones(batch_size).to(self.device)
            loss_val = self.cosine_loss(v_gen, v_ref, target_label)
        
        else:
            raise ValueError(f"Mode {self.mode} not supported")

        return loss_val * self.adv_weight

class RecLoss(nn.Module):
    def __init__(self, rec_weight, loss_mode, device, mse_weight=1.0, lpips_weight=1.0):
        """
        Args:
            rec_weight (float): Global weight for the reconstruction loss.
            loss_mode (str): 'l2', 'lpips' or 'combined'.
            device (str): 'cuda' or 'cpu'.
            mse_weight (float): Internal weight for the MSE (L2) term.
            lpips_weight (float): Internal weight for the LPIPS term.
        """
        super(RecLoss, self).__init__()
        self.rec_weight = rec_weight
        self.mode = loss_mode
        self.device = device
        
        # Internal weights
        self.mse_weight = mse_weight
        self.lpips_weight = lpips_weight

        # Initialize the specific loss components.
        self.criterion_mse = None
        self.criterion_lpips = None

        if self.mode in ['l2', 'combined']:
            self.criterion_mse = nn.MSELoss().to(device)
            
        if self.mode in ['lpips', 'combined']:
            # net_type='alex' is fast and standard for perceptual similarity
            self.criterion_lpips = LPIPS(net_type='alex').to(device).eval() 

        if self.mode not in ['l2', 'lpips', 'combined']:
             raise ValueError(f'Unexpected Loss Mode {loss_mode}')            

    def forward(self, img_input, img_output):
        loss_components = 0.0
        
        # 1. Component MSE (L2) --> weighted by mse_weight
        if self.mode in ['l2', 'combined']:
            # this is used to preserve the watermark and the visual similarity, so we want to minimize it.
            mse_val = self.criterion_mse(img_input, img_output)
            loss_components += (self.mse_weight * mse_val)

        # 2. Component LPIPS --> weighted by lpips_weight
        if self.mode in ['lpips', 'combined']:
            # LPIPS expects inputs in the range [-1, 1], so we normalize them.
            img_input_norm = img_input * 2 - 1
            img_output_norm = img_output * 2 - 1

            # Safety: ensure values are within the valid range due to potential floating point errors
            img_input_norm_c = torch.clamp(img_input_norm, -1, 1)
            img_output_norm_c = torch.clamp(img_output_norm, -1, 1)
            lpips_val = self.criterion_lpips(img_input_norm_c, img_output_norm_c).mean()
            loss_components += (self.lpips_weight * lpips_val)

        # Return the weighted sum of internal components multiplied by the global weight
        return loss_components * self.rec_weight
    
class FreqLoss(nn.Module):
    def __init__(
        self,
        freq_weight=1.0,
        loss_mode="l2",
        band="all",
        device="cuda",
        gamma=1.0,
        low_thr=0.25,
        high_thr=0.65,
        low_weight=0.0,
        mid_weight=0.5,
        high_weight=1.0,
        normalize_by_mask_area=False,
        eps=1e-8,
    ):
        """
        Frequency-domain preservation loss.

        This loss penalizes the spectral energy/magnitude of the effective
        adversarial perturbation:

            delta = img_input - img_target

        where typically:

            img_input  = x_adv
            img_target = imgs_wm

        Args:
            freq_weight (float):
                Global weight for the frequency loss.

            loss_mode (str):
                'l1' or 'l2'.

                'l1':
                    mean(w_map * abs(FFT(delta)))

                'l2':
                    mean(w_map * abs(FFT(delta)) ** 2)

            band (str):
                Which frequency region to penalize.

                'all':
                    Uses a continuous radial map:
                    center = 0, borders = 1.

                'low':
                    Penalizes low frequencies.

                'mid':
                    Penalizes middle frequencies.

                'high':
                    Penalizes high frequencies.

                'custom':
                    Uses low_weight, mid_weight and high_weight to build
                    a weighted band map.

            device (str):
                'cuda' or 'cpu'.

            gamma (float):
                Controls the sharpness of the radial weights.

                gamma = 1.0 -> linear
                gamma > 1.0 -> stronger penalty near extreme high frequencies
                gamma < 1.0 -> smoother penalty from mid frequencies

            low_thr (float):
                Radius threshold for low frequencies.
                Values are normalized in [0, 1].

            high_thr (float):
                Radius threshold for high frequencies.
                Frequencies above this are considered high.

            low_weight, mid_weight, high_weight (float):
                Used only when band='custom'.

            normalize_by_mask_area (bool):
                If True, divides the loss by the average mask area.
                Useful for facial mask changes size between images.

            eps (float):
                Small constant for numerical stability.
        """
        super(FreqLoss, self).__init__()

        self.freq_weight = freq_weight
        self.loss_mode = loss_mode
        self.band = band
        self.device = device
        self.gamma = gamma

        self.low_thr = low_thr
        self.high_thr = high_thr

        self.low_weight = low_weight
        self.mid_weight = mid_weight
        self.high_weight = high_weight

        self.normalize_by_mask_area = normalize_by_mask_area
        self.eps = eps

        valid_modes = ["l1", "l2"]
        valid_bands = ["all", "low", "mid", "high", "custom"]

        if self.loss_mode not in valid_modes:
            raise ValueError(f"Unexpected loss_mode '{loss_mode}'. Use one of {valid_modes}.")

        if self.band not in valid_bands:
            raise ValueError(f"Unexpected band '{band}'. Use one of {valid_bands}.")

    def _radial_coordinates(self, H, W, device):
        """
        Creates normalized radial coordinates in [0, 1].

        After fftshift:
            center -> low frequencies
            borders -> high frequencies
        """
        cy, cx = H // 2, W // 2

        y = torch.arange(H, device=device).float() - cy
        x = torch.arange(W, device=device).float() - cx

        yy, xx = torch.meshgrid(y, x, indexing="ij")

        r = torch.sqrt(yy ** 2 + xx ** 2)
        r = r / (r.max() + self.eps)

        return r

    def _build_weight_map(self, H, W, device):
        """
        Builds the frequency weight map.

        Returns:
            w_map: shape [1, 1, H, W], ready to broadcast over [B, C, H, W].
        """
        r = self._radial_coordinates(H, W, device)

        if self.band == "all":
            # Basic idea:
            # center = 0, borders = 1
            w_map = r ** self.gamma

        elif self.band == "low":
            # Penalize only low frequencies.
            # Strongest at the center, zero after low_thr.
            w_map = torch.clamp((self.low_thr - r) / (self.low_thr + self.eps), 0.0, 1.0)
            w_map = w_map ** self.gamma

        elif self.band == "mid":
            # Penalize middle frequencies.
            # Creates a smooth band between low_thr and high_thr.
            mid_center = (self.low_thr + self.high_thr) / 2.0
            mid_width = (self.high_thr - self.low_thr) / 2.0

            w_map = 1.0 - torch.abs(r - mid_center) / (mid_width + self.eps)
            w_map = torch.clamp(w_map, 0.0, 1.0)
            w_map = w_map ** self.gamma

        elif self.band == "high":
            # Penalize only high frequencies.
            # Zero below high_thr, then smoothly increases to 1.
            w_map = torch.clamp((r - self.high_thr) / (1.0 - self.high_thr + self.eps), 0.0, 1.0)
            w_map = w_map ** self.gamma

        elif self.band == "custom":
            # Three explicit bands:
            # low:  r < low_thr
            # mid:  low_thr <= r < high_thr
            # high: r >= high_thr
            low_mask = (r < self.low_thr).float()
            mid_mask = ((r >= self.low_thr) & (r < self.high_thr)).float()
            high_mask = (r >= self.high_thr).float()

            w_map = (
                self.low_weight * low_mask
                + self.mid_weight * mid_mask
                + self.high_weight * high_mask
            )

        else:
            raise ValueError(f"Unexpected band '{self.band}'.")

        return w_map[None, None, :, :]

    def forward(self, img_input, img_target, mask=None):
        """
        Args:
            img_input:
                Usually x_adv. Shape [B, C, H, W].

            img_target:
                Usually imgs_wm. Shape [B, C, H, W].

            mask:
                Optional spatial mask. Shape can be [B, 1, H, W] or [B, C, H, W].
                This is only used for normalization if normalize_by_mask_area=True.
                The perturbation itself should already be reflected in:
                    delta = img_input - img_target

        Returns:
            Frequency loss multiplied by freq_weight.
        """
        B, C, H, W = img_input.shape

        # Effective perturbation.
        # This is better than using delta_img directly because img_input may have been clamped.
        delta = img_input - img_target

        # FFT of the effective perturbation.
        delta_freq = torch.fft.fftshift(
            torch.fft.fft2(delta.float(), norm="ortho"),
            dim=(-2, -1)
        )

        magnitude = torch.abs(delta_freq)

        w_map = self._build_weight_map(H, W, img_input.device)

        if self.loss_mode == "l1":
            freq_val = torch.mean(w_map * magnitude)

        elif self.loss_mode == "l2":
            freq_val = torch.mean(w_map * (magnitude ** 2))

        else:
            raise ValueError(f"Unexpected loss_mode '{self.loss_mode}'.")

        if self.normalize_by_mask_area and mask is not None:
            mask_area = mask.float().mean().clamp_min(self.eps)
            freq_val = freq_val / mask_area

        return freq_val * self.freq_weight