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