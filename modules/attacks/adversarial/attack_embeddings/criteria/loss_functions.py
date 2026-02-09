import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from criteria.lpips.lpips import LPIPS

class AdvLoss(nn.Module):
    def __init__(self, adv_weight, device, mode='evasion'):
        """
        Args:
            adv_weight (float): Peso global del ataque adversarial.
            device (str): 'cuda' o 'cpu'.
            mode (str): 'evasion' (Untargeted) o 'targeted'.
        """
        super(AdvLoss, self).__init__()
        self.adv_weight = adv_weight
        self.device = device
        self.mode = mode
        
        # Coseno Similitud: mide el ángulo entre dos vectores
        self.cosine_loss = nn.CosineEmbeddingLoss()

    def forward(self, v_gen, v_ref):
        """
        Args:
            v_gen (Tensor): El vector facial extraído de la imagen generada/atacada. 
                            Debe tener gradientes activos (requires_grad=True en el flujo anterior).
            v_ref (Tensor): El vector facial de referencia (original u objetivo).
        """
        
        # Obtenemos el tamaño del batch (N)
        batch_size = v_gen.size(0)
        
        loss_val = 0.0
        
        if self.mode == 'evasion':
            # OBJETIVO: EVASIÓN (Untargeted)
            # Target = -1 -> Fuerza a los vectores a ser opuestos (minimiza similitud)
            target_label = -1 * torch.ones(batch_size).to(self.device)
            loss_val = self.cosine_loss(v_gen, v_ref, target_label)
            
        elif self.mode == 'targeted':
            # OBJETIVO: SUPLANTACIÓN (Targeted)
            # Target = 1 -> Fuerza a los vectores a ser iguales (maximiza similitud)
            target_label = torch.ones(batch_size).to(self.device)
            loss_val = self.cosine_loss(v_gen, v_ref, target_label)
        
        else:
            raise ValueError(f"Mode {self.mode} not supported")

        return loss_val * self.adv_weight

class RecLoss(nn.Module):
    def __init__(self, rec_weight, loss_mode, device, mse_weight=1.0, lpips_weight=1.0):
        """
        Args:
            rec_weight (float): Peso global de la reconstrucción (frente al ataque adversarial).
            loss_mode (str): 'l2', 'lpips' o 'combined'.
            device (str): 'cuda' o 'cpu'.
            mse_weight (float): Peso interno para el término MSE (L2).
            lpips_weight (float): Peso interno para el término LPIPS.
        """
        super(RecLoss, self).__init__()
        self.rec_weight = rec_weight
        self.mode = loss_mode
        self.device = device
        
        # Pesos internos
        self.mse_weight = mse_weight
        self.lpips_weight = lpips_weight

        # Inicialización de criterios bajo demanda
        self.criterion_mse = None
        self.criterion_lpips = None

        if self.mode in ['l2', 'combined']:
            self.criterion_mse = nn.MSELoss().to(device)
            
        if self.mode in ['lpips', 'combined']:
            # net_type='alex' es rápido y estándar para similitud perceptual
            self.criterion_lpips = LPIPS(net_type='alex').to(device).eval() 

        if self.mode not in ['l2', 'lpips', 'combined']:
             raise ValueError(f'Unexpected Loss Mode {loss_mode}')            

    def forward(self, img_input, img_output):
        loss_components = 0.0
        
        # 1. Componente MSE (L2) -> Ponderado por mse_weight
        if self.mode in ['l2', 'combined']:
            mse_val = self.criterion_mse(img_input, img_output)
            loss_components += (self.mse_weight * mse_val)

        # 2. Componente LPIPS -> Ponderado por lpips_weight
        if self.mode in ['lpips', 'combined']:
            img_input_norm = img_input * 2 - 1
            img_output_norm = img_output * 2 - 1

            # Seguridad: asegurar que no nos pasamos de rango por decimales
            img_input_norm_c = torch.clamp(img_input_norm, -1, 1)
            img_output_norm_c = torch.clamp(img_output_norm, -1, 1)
            lpips_val = self.criterion_lpips(img_input_norm_c, img_output_norm_c).mean()
            loss_components += (self.lpips_weight * lpips_val)

        # Retornamos la suma ponderada interna multiplicada por el peso global
        return loss_components * self.rec_weight