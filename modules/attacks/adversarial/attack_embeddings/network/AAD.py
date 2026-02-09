import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def weight_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

    if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)


class AADLayer(nn.Module):
    def __init__(self, c_h, c_att, c_id=256):
        super(AADLayer, self).__init__()

        self.norm = nn.InstanceNorm2d(c_h, affine=False)
        self.conv_h = nn.Conv2d(c_h, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(c_att, c_h, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(c_att, c_h, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(c_id, c_h)
        self.fc2 = nn.Linear(c_id, c_h)
        
    def forward(self, h_in, z_att, z_id):
        h_bar = self.norm(h_in)
        M = self.sigmoid(self.conv_h(h_bar))

        gamma_attr = self.conv1(z_att)
        beta_attr = self.conv2(z_att)
        A = A = gamma_attr * h_bar + beta_attr

        gamma_id = self.fc1(z_id).unsqueeze(-1).unsqueeze(-1).expand_as(h_in)
        beta_id = self.fc2(z_id).unsqueeze(-1).unsqueeze(-1).expand_as(h_in)
        I = gamma_id * h_bar + beta_id

        h_out = (1 - M) * A + M * I
        return h_out



class AADResBlock(nn.Module):
    def __init__(self, c_in, c_out, c_att, c_id=256):
        super(AADResBlock, self).__init__()
        self.c_in = c_in
        self.c_out = c_out

        # Sub-Block1
        self.aad1 = AADLayer(c_in, c_att, c_id)
        self.conv1 = nn.Conv2d(c_in, c_in, kernel_size=3, stride=1, padding=1)
        
        # Sub-Block2
        self.aad2 = AADLayer(c_in, c_att, c_id)
        self.conv2 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1)

        if c_in != c_out:
            self.aad3 = AADLayer(c_in, c_att, c_id)
            self.conv3 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1)
        
        self.activation = nn.ReLU(inplace=True)

    def forward(self, h_in, z_att, z_id):
        x = self.aad1(h_in, z_att, z_id)
        x = self.activation(x)
        x = self.conv1(x)

        x = self.aad2(x, z_att, z_id)
        x = self.activation(x)
        x = self.conv2(x)

        h_mid = h_in
        if self.c_in != self.c_out:
            h_mid = self.aad3(h_in, z_att, z_id)
            h_mid = self.activation(h_mid)
            h_mid = self.conv3(h_mid)
        
        h_out = x + h_mid
        return h_out



class AADGenerator(nn.Module):
    def __init__(self, c_id=256):
        super(AADGenerator, self).__init__()
        self.convt = nn.ConvTranspose2d(c_id, 1024, kernel_size=2, stride=1, padding=0)

        self.AADResBlk1 = AADResBlock(c_in=1024, c_out=1024, c_att=1024, c_id=c_id)
        self.AADResBlk2 = AADResBlock(1024, 1024, 2048, c_id)
        self.AADResBlk3 = AADResBlock(1024, 1024, 1024, c_id)
        self.AADResBlk4 = AADResBlock(1024, 512, 512, c_id)
        self.AADResBlk5 = AADResBlock(512, 256, 256, c_id)
        self.AADResBlk6 = AADResBlock(256, 128, 128, c_id)
        self.AADResBlk7 = AADResBlock(128, 64, 64, c_id)
        self.AADResBlk8 = AADResBlock(64, 3, 64, c_id)

        self.apply(weight_init)
        self.output_function = nn.Sigmoid()

    def forward(self, inputs):
    #def forward(self, z_att, z_id):
        z_att, z_id = inputs
        # m0: bs x 1024 x 2 x 2
        m = self.convt(z_id.unsqueeze(-1).unsqueeze(-1))
        
        # m1: bs x 1024 x 4 x 4
        m = F.interpolate(self.AADResBlk1(m, z_att[0], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        
        # m2: bs x 1024 x 8 x 8
        m = F.interpolate(self.AADResBlk2(m, z_att[1], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        
        # m3: bs x 1024 x 16 x 16
        m = F.interpolate(self.AADResBlk3(m, z_att[2], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        
        # m4: bs x 512 x 32 x 32
        m = F.interpolate(self.AADResBlk4(m, z_att[3], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        
        # m5: bs x 256 x 64 x 64
        m = F.interpolate(self.AADResBlk5(m, z_att[4], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        
        # m6: bs x 128 x 128 x 128
        m = F.interpolate(self.AADResBlk6(m, z_att[5], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        
        # m7: bs x 64 x 256 x 256
        m = F.interpolate(self.AADResBlk7(m, z_att[6], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        
        # y: bs x 3 x 256 x 256
        y = self.AADResBlk8(m, z_att[7], z_id)

        return self.output_function(y)

class FusionModule(nn.Module):
    def __init__(self, input_channels=6):
        """
        Input: Concatenación de Img Original (3ch) + Img Adversarial (3ch) = 6 canales.
        Output: Máscara de 1 canal (valores 0 a 1).
        """
        super(FusionModule, self).__init__()
        self.net = nn.Sequential(
            # Capa 1: Analizar características locales
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Capa 2: Refinar la forma de la máscara
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Capa 3: Comprimir a 1 canal
            nn.Conv2d(32, 1, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_org, img_adv, weight_mask=None):
        # Concatenar inputs: (Batch, 6, H, W)
        x = torch.cat([img_org, img_adv], dim=1)
        
        # Generar logits y normalizar a [0, 1]
        mask = self.sigmoid(self.net(x))
        if weight_mask is not None:
            mask = mask * weight_mask
        
        # Mezcla diferenciable:
        # Mask = 1 -> Usa Adversarial (Identidad falsa)
        # Mask = 0 -> Usa Original (Marca de agua intacta)
        out = mask * img_adv + (1 - mask) * img_org
        
        return out, mask

def get_spatial_weights(h, w, device, sigma_scale=0.4):
    """
    Genera una matriz de pesos espaciales con forma (1, 1, H, W) para broadcasting.
    - Centro (Cara) -> Valores bajos (~0.0). 'Barato' enmascarar aquí.
    - Bordes (Fondo) -> Valores altos (~1.0). 'Caro' enmascarar aquí.
    """
    # 1. Crear grid de coordenadas [-1, 1]
    x = torch.linspace(-1, 1, w, device=device)
    y = torch.linspace(-1, 1, h, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij') # Forma (H, W)
    
    # 2. Calcular distancia al cuadrado desde el centro (0,0)
    d2 = grid_x**2 + grid_y**2
    
    # 3. Gaussiana invertida
    # sigma_scale controla el radio del "hueco" central. 
    # Menor sigma = hueco más pequeño y concentrado.
    gaussian = torch.exp(-d2 / (2 * sigma_scale**2))
    
    # Invertimos: 1.0 en bordes, 0.0 en centro
    weight_map = 1.0 - gaussian
    
    # 4. AÑADIR DIMENSIONES DE BATCH Y CANAL -> (1, 1, H, W)
    # Esto es lo que permite multiplicar por un batch de tamaño N automáticamente
    return weight_map.view(1, 1, h, w)

def get_gaussian_kernel(kernel_size=5, sigma=1.0, channels=3):
    # Crear kernel gaussiano para separar frecuencias
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1) / 2.
    variance = sigma**2.
    gaussian_kernel = (1./(2.*math.pi*variance)) * torch.exp( -torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance) )
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    return gaussian_kernel.repeat(channels, 1, 1, 1)

def frequency_injection(img_gen, img_org, kernel, device):
    # 1. Calcular el padding dinámicamente según el tamaño del kernel
    # Si kernel=3 -> pad=1. Si kernel=5 -> pad=2.
    k_size = kernel.shape[2]
    pad = k_size // 2 
    
    # 2. Aplicar convolución con el padding correcto
    # Nota: groups=3 asume que la imagen tiene 3 canales (RGB)
    low_real = F.conv2d(img_org, kernel.to(device), padding=pad, groups=3)
    high_real = img_org - low_real
    
    low_gen = F.conv2d(img_gen, kernel.to(device), padding=pad, groups=3)
    
    # 3. Fusionar
    return low_gen + high_real
