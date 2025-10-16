"""
Author: Gao Yu
Company: Bosch Research / Asia Pacific
Date: 2024-08-03
Description: util functions for stegaformer
Mofified by: Jefferson Rodríguez & Gemini - University of Cagliari - 2025-06-25
"""
import os
import sqlite3
from glob import glob
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils import data

class VGGLoss(nn.Module):
    """
    Part of pre-trained VGG16.
    See for instance https://arxiv.org/abs/1603.08155

    block_no：how many blocks do we need; layer_within_block：which layer within the block do we need
    """

    def __init__(self, block_no: int, layer_within_block: int, use_batch_norm_vgg: bool):
        super(VGGLoss, self).__init__()
        if use_batch_norm_vgg:
            vgg16 = torchvision.models.vgg16_bn(pretrained=True)
        else:
            vgg16 = torchvision.models.vgg16(pretrained=True)
        curr_block = 1
        curr_layer = 1
        layers = []
        for layer in vgg16.features.children():
            layers.append(layer)
            if curr_block == block_no and curr_layer == layer_within_block:
                break
            if isinstance(layer, nn.MaxPool2d):
                curr_block += 1
                curr_layer = 1
            else:
                curr_layer += 1
        self.vgg_loss = nn.Sequential(*layers)

    def forward(self, img):
        return self.vgg_loss(img)

def infinite_sampler(n: int):
    """
    Yields an infinite sequence of random indices from 0 to n-1.

    Args:
        n (int): The number of samples.

    Yields:
        int: Random index from 0 to n-1.
    """
    if n <= 0:
        raise ValueError("[ERROR] El dataset está vacío. No se puede crear un sampler con 0 elementos.")

    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0

class InfiniteSamplerWrapper(data.Sampler):
    """
    A data sampler that wraps around another sampler to yield an infinite sequence of samples.
    
    Args:
        data_source (Dataset): The dataset to sample from.
    """
    
    def __init__(self, data_source: data.Dataset):
        self.num_samples = len(data_source)

    def __iter__(self):
        """
        Returns an iterator that yields an infinite sequence of samples.
        
        Returns:
            Iterator[int]: An iterator over random indices.
        """
        return iter(infinite_sampler(self.num_samples))

    def __len__(self) -> int:
        """
        Returns a large number to simulate an infinite length.
        
        Returns:
            int: A large integer value.
        """
        return 2 ** 31
        
def rgb_to_yuv(image: torch.Tensor) -> torch.Tensor:
    """
    Convert an RGB image to YUV color space.

    Args:
        image (torch.Tensor): Input image tensor with shape (*, 3, H, W).

    Returns:
        torch.Tensor: Image tensor in YUV color space with shape (*, 3, H, W).

    Raises:
        TypeError: If the input is not a torch.Tensor.
        ValueError: If the input tensor does not have at least 3 dimensions or the third dimension is not 3.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    u: torch.Tensor = -0.147 * r - 0.289 * g + 0.436 * b
    v: torch.Tensor = 0.615 * r - 0.515 * g - 0.100 * b

    yuv_image: torch.Tensor = torch.stack([y, u, v], dim=-3)

    return yuv_image

def get_message_accuracy(
    msg: torch.Tensor,
    deco_msg: torch.Tensor,
    msg_num: int
) -> float:
    """
    Calculate the bit accuracy of decoded messages.

    Args:
        msg (torch.Tensor): The original message tensor.
        deco_msg (torch.Tensor): The decoded message tensor.
        msg_num (int): The number of the message segments.

    Returns:
        float: The bit accuracy of the decoded messages.

    Raises:
        TypeError: If the inputs are not torch.Tensors or msg_num is not an int.
        ValueError: If the msg_num is less than 1.
    """
    if not isinstance(msg, torch.Tensor) or not isinstance(deco_msg, torch.Tensor):
        raise TypeError("Inputs msg and deco_msg must be torch.Tensors.")
    
    if not isinstance(msg_num, int):
        raise TypeError("Input msg_num must be an int.")
    
    if msg_num < 1:
        raise ValueError("msg_num must be at least 1.")
    
    if 'cuda' in str(deco_msg.device):
        deco_msg = deco_msg.cpu()
        msg = msg.cpu()

    if msg_num == 1:
        deco_msg = torch.round(deco_msg)
        correct_pred = torch.sum((deco_msg - msg) == 0, dim=1)
        bit_acc = torch.sum(correct_pred).numpy() / deco_msg.numel()
    else:
        bit_acc = 0.0
        for i in range(msg_num):
            cur_deco_msg = torch.round(deco_msg[:, i, :])
            correct_pred = torch.sum((cur_deco_msg - msg[:, i, :]) == 0, dim=1)
            #print(cur_deco_msg)
            bit_acc += torch.sum(correct_pred).numpy() / cur_deco_msg.numel()
        bit_acc /= msg_num

    return bit_acc

def get_watermark_from_db(db_path: str, filename: str) -> torch.Tensor:
    """
    Retrieves the binary watermark message for a given filename from the SQLite database.
    Args:
        db_path (str): Path to the SQLite watermark database file.
        filename (str): The filename (image_filename) for which to retrieve the watermark.
    Returns:
        torch.Tensor: A tensor representing the binary watermark message (0s and 1s),
                      or None if not found.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT watermark_data FROM watermarks WHERE image_filename = ?", (filename,))
        result = cursor.fetchone()
        if result:
            # Convert the string to a tensor of floats
            watermark_str = result[0]
            # Ensure it's a list of floats 
            if ',' in watermark_str:
                message_list = [float(bit) for bit in watermark_str.split(',')]
            else:   
                message_list = [float(bit) for bit in watermark_str]
            return np.array(message_list, dtype=np.float32)
        else:
            print(f"Watermark not found for filename: {filename} in {db_path}")
            return None
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None
    finally:
        if conn:
            conn.close()

def load_and_preprocess_image(image_path: Path, 
                              im_size: int, 
                              img_norm: bool, 
                              device_id: int,
                              image_format: str = 'png') -> torch.Tensor:
    """
    Loads an image from path and preprocesses it for model input.
    Args:
        image_path (Path): Path to the image file.
        im_size (int): Target size for the image (im_size, im_size).
        device_id (int): GPU device ID.
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    # Load and process the cover image
    if image_format == 'png':
        img = Image.open(image_path).convert('RGB')
        img_cover = ImageOps.fit(img, (im_size,im_size))
        if img_norm:
            img_cover = np.array(img_cover, dtype=np.float32) / 255.0
        else:
            img_cover = np.array(img_cover, dtype=np.float32)

    elif image_format == 'npy':
        img_cover = np.load(image_path)
        #img_array = np.clip(img_array, 0, 255)
        if img_norm:
            img_cover = img_cover / 255.0       
    else:
        raise ValueError(f"Unsupported image format: {image_format}")
    
    image_tensor = transforms.ToTensor()(img_cover)

    return image_tensor.unsqueeze(0).cuda(device_id) # Add batch dimension and move to GPU

class MIMData_inference(Dataset):
    """
    A custom dataset class for handling images and secret messages for inference.
    """
    def __init__(
        self,
        data_path: str,
        db_path: str, # Path to the SQLite database containing watermarks
        num_message: int = 16,
        message_size: int = 64*64,
        image_size: tuple = (256, 256),
        dataset: str = 'facelab_london',
        roi: str = 'fit',
        msg_r: int = 1,
        img_norm: bool = False
    ):
        self.data_path = data_path
        self.db_path = db_path
        self.m_size = message_size
        self.m_num = num_message
        self.im_size = image_size
        self.roi = roi
        self.msg_range = msg_r
        self.image_norm = img_norm

        assert dataset in ['facelab_london', 'CFD', 'ONOT', 'LFW'], "Invalid DataSet. only support ['facelab_london', 'CFD', 'ONOT', 'LFW']."
        assert roi in ['fit', 'crop'], "Invalid Roi Selection. only support ['fit', 'crop']."

        if dataset == 'facelab_london':
            self.files_list = glob(os.path.join(self.data_path, '*.jpg'))
        elif dataset == 'CFD':
            self.files_list = glob(os.path.join(self.data_path, '*.jpg'))
        elif dataset == 'ONOT':
            self.files_list = glob(os.path.join(self.data_path, '*.png'))
        elif dataset == 'LFW':
            self.files_list = glob(os.path.join(self.data_path, '*.jpg'))

        if not self.files_list:
            raise RuntimeError(f"No image files found in {self.data_path} for dataset {dataset} with specified extension.")
            
        self.to_tensor = transforms.ToTensor()

        # Establish SQLite database connection.
        # This connection will be persistent throughout the dataset's life.
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def __getitem__(self, idx: int) -> tuple:
        img_cover_path = self.files_list[idx]
        # Extract just the filename (e.g., '001_1_000000.png') to query the database.
        filename = os.path.basename(img_cover_path) 

        # Load and process the cover image
        img = Image.open(img_cover_path).convert('RGB')
        if self.roi == 'fit':
            img_cover = ImageOps.fit(img, self.im_size)
        elif self.roi == 'crop':
            width, height = img.size   # Get dimensions
            left = (width - self.im_size[0]) / 2
            top = (height - self.im_size[1]) / 2
            right = (width + self.im_size[0]) / 2
            bottom = (height + self.im_size[1]) / 2
            # Crop the center of the image
            img_cover = img.crop((left, top, right, bottom))

        if self.image_norm:
            img_cover = np.array(img_cover, dtype=np.float32) / 255.0
        else:
            img_cover = np.array(img_cover, dtype=np.float32)

        img_cover = self.to_tensor(img_cover)

        # --- Load the message (watermark) from the database ---
        watermark_str = None
        messages = None # Initialize messages to None or a default zero array
        
        try:
            # Execute SQL query to get the watermark data for the current filename
            self.cursor.execute("SELECT watermark_data FROM watermarks WHERE image_filename = ?", (filename,))
            result = self.cursor.fetchone() # Fetch the result

            if result:
                watermark_str = result[0] # The watermark string is the first element of the result tuple
            else:
                # Fallback if watermark is not found for a given filename
                print(f"Warning: Watermark not found for {filename} in DB. Returning a zero watermark.")
                # Create a zero-filled watermark of the expected shape (m_num, m_size)
                messages = np.zeros((self.m_num, self.m_size), dtype=np.float32)
        except sqlite3.Error as e:
            # Handle any SQLite database errors during fetching
            print(f"Database error while fetching watermark for {filename}: {e}. Returning a zero watermark.")
            messages = np.zeros((self.m_num, self.m_size), dtype=np.float32) # Fallback to zeros

        if watermark_str:
            # Convert the watermark string ('0101...') to a NumPy array of floats (0.0 or 1.0)
            # for bpp > 3 the watermark is stored as comma-separated values
            if self.msg_range > 1:
                messages_flat = np.array([float(bit) for bit in watermark_str.split(',')]).astype(np.float32)
            else:
                messages_flat = np.array(list(watermark_str)).astype(np.float32)
            
            # Debug: Print the loaded watermark for verification
            #print(filename)
            #print(f"Loaded watermark for {filename}: {messages_flat}")
            # Reshape the flat array to the expected (num_message, message_size) format (e.g., 4096, 16)
            messages = messages_flat.reshape((self.m_num, self.m_size))

            # Apply normalization to messages if img_norm is true, mirroring MIMData's behavior
            # Original MIMData: `messages[n, :] = message / (self.msg_range + 1)` if image_norm is True
            if self.image_norm:
                messages = messages / (self.msg_range + 1) # Normalizes 0 to 0 and 1 to 0.5 (if msg_range=1)

        # Convert the NumPy message array to a PyTorch tensor (float type)
        messages = torch.from_numpy(messages).float()
        
        return img_cover, messages, filename
    
    def __len__(self) -> int:
        return len(self.files_list)

class MIMData(Dataset):
    """
    A custom dataset class for handling images and secret messages.

    Args:
        data_path (str): Path to the dataset.
        num_message (int, optional): Number of messages. Defaults to 16.
        message_size (int, optional): Size of each message. Defaults to 64*64.
        image_size (tuple, optional): Size of the images. Defaults to (256, 256).
        dataset (str, optional): Dataset type ('coco' or 'div2k'). Defaults to 'coco'.
            # Adapted to support CelebA-HQ (celeba_hq) dataset as well.
        roi (str, optional): Region of interest ('fit' or 'crop'). Defaults to 'fit'.
        msg_r (int, optional): Message range. Defaults to 1.
        img_norm (bool, optional): Whether to normalize images. Defaults to False.
    """
    def __init__(
        self,
        data_path: str,
        num_message: int = 16,
        message_size: int = 64*64,
        image_size: tuple = (256, 256),
        dataset: str = 'celeba_hq',
        roi: str = 'fit',
        msg_r: int = 1,
        img_norm: bool = False
    ):
        self.data_path = data_path
        self.m_size = message_size
        self.m_num = num_message
        self.im_size = image_size
        self.roi = roi
        self.msg_range = msg_r
        self.image_norm = img_norm
        
        assert dataset in ['coco', 'div2k', 'celeba_hq'], "Invalid DataSet. only support ['coco', 'div2k', 'celeba_hq']."
        assert roi in ['fit', 'crop'], "Invalid Roi Selection. only support ['fit', 'crop']."
        
        if dataset == 'div2k':
            self.files_list = glob(os.path.join(self.data_path, '*.png'))
        elif dataset == 'coco':
            self.files_list = glob(os.path.join(self.data_path, '*.jpg'))
        elif dataset == 'celeba_hq':
            self.files_list = glob(os.path.join(self.data_path, '*.png'))
            
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, idx: int) -> tuple:
        img_cover_path = self.files_list[idx]

        # Load and process the cover image
        img = Image.open(img_cover_path).convert('RGB')
        if self.roi == 'fit':
            img_cover = ImageOps.fit(img, self.im_size)
        elif self.roi == 'crop':
            width, height = img.size   # Get dimensions
            left = (width - self.im_size[0]) / 2
            top = (height - self.im_size[1]) / 2
            right = (width + self.im_size[0]) / 2
            bottom = (height + self.im_size[1]) / 2
            # Crop the center of the image
            img_cover = img.crop((left, top, right, bottom))

        if self.image_norm:
            img_cover = np.array(img_cover, dtype=np.float32) / 255.0
        else:
            img_cover = np.array(img_cover, dtype=np.float32)

        img_cover = self.to_tensor(img_cover)
        
        # Generate secret messages
        messages = np.zeros((self.m_num, self.m_size))
        for n in range(self.m_num):
            message = np.random.randint(low=0, high=self.msg_range + 1, size=self.m_size)
            if self.image_norm:
                messages[n, :] = message / (self.msg_range + 1)
            else:
                messages[n, :] = message
            
        messages = torch.from_numpy(messages).float()
        
        return img_cover, messages

    def __len__(self) -> int:
        return len(self.files_list)

if __name__ == '__main__':

    # --- Configuration for MIMData (training) ---
    print("\nTesting MIMData (Training)...")
    train_data_path = os.path.join('/app/facial_data','celeba_hq','train/real')
    msg_range = 1  # message r (range) should be 1, 3, 7 for binary, 1, 2, 3-bit per message unit
    try:
        train_dataset = MIMData(
            data_path=train_data_path,
            num_message=64*64,
            message_size=16,
            image_size=(256, 256),
            dataset='celeba_hq', 
            msg_r=msg_range,
            img_norm=False,
            roi='fit'
        )
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        train_iterator = iter(train_loader)
        img_train, msg_train = next(train_iterator)

        print(f"MIMData: Dataset size: {len(train_dataset)}")
        print(f"MIMData: Image batch shape: {img_train.shape}")
        print(f"MIMData: Message batch shape: {msg_train.shape}")
        print(f"MIMData: Sample message: {msg_train[0].flatten()}")
    except RuntimeError as e:
        print(f"Error testing MIMData: {e}")
    except StopIteration:
        print("MIMData DataLoader is empty.")
    except Exception as e:
        print(f"An unexpected error occurred with MIMData: {e}")

    img2, msg_train2 = next(train_iterator)
    # check the message accuracy
    acc = get_message_accuracy(msg_train, msg_train, msg_num=64*64)
    print(f"Two identity message acc = {acc}")
    acc = get_message_accuracy(msg_train, msg_train2, msg_num=64*64)
    print(f"Random guess of two message in range{msg_range}, acc = {acc}")

    # --- Configuration for MIMData_Inference (default for a quick test) ---
    print("\nTesting MIMData_Inference (Inference)...")
    inference_data_path = os.path.join('/app/facial_data','facelab_london', 'processed', 'test')
    watermark_db_path = os.path.join('/app/facial_data','facelab_london', 'processed','watermarks', 'watermarks_BBP_1_65536_500_facelab_london.db')

    try:
        inference_dataset = MIMData_inference(
            data_path=inference_data_path,
            db_path=watermark_db_path,
            num_message=4096,  
            message_size=16,   
            image_size=(256, 256),
            dataset='facelab_london',
            img_norm=False,
            msg_r=1,  # message range 1--> 0 or 1
            roi='fit'
        )
        inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False)
        inference_iterator = iter(inference_loader)
        img_inf, msg_inf, _ = next(inference_iterator)

        print(f"MIMData_Inference: Dataset size: {len(inference_dataset)}")
        print(f"MIMData_Inference: Image batch shape: {img_inf.shape}")
        print(f"MIMData_Inference: Message batch shape: {msg_inf.shape}")
        print(f"MIMData_Inference: Sample message: {msg_inf[0].flatten()}")
    except RuntimeError as e:
        print(f"Error testing MIMData_Inference: {e}")
    except StopIteration:
        print("MIMData_Inference DataLoader is empty.")
    except sqlite3.Error as e:
        print(f"Database error with MIMData_Inference: {e}")
    except Exception as e:
        print(f"An unexpected error occurred with MIMData_Inference: {e}")

    img2_inf, msg_inf2, _ = next(inference_iterator)
    # check the message accuracy
    acc = get_message_accuracy(msg_inf, msg_inf, msg_num=64*64)
    print(f"Two identity message acc = {acc}")
    acc = get_message_accuracy(msg_inf, msg_inf2, msg_num=64*64)
    print(f"Random guess of two message in range{msg_range}, acc = {acc}")
    