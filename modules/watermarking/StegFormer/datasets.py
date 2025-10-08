import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from natsort import natsorted
from PIL import Image, ImageOps
import numpy as np
import torchvision.transforms as transforms
from glob import glob
import sqlite3
from torchvision.transforms.functional import to_pil_image
from utils import load_and_preprocess_image, get_mapping_params, symbols_to_message_image, get_message_accuracy, compute_image_score

def inference(inference_loader, encoder, decoder, device, 
            bpp, norm_train, output_save_dir, cal_psnr, cal_ssim):
    
    os.makedirs(output_save_dir, exist_ok=True)
    acc_list = []
    psnr_list = []
    ssim_list = []

    with torch.no_grad():
        for i, (images, messages, filenames) in tqdm(enumerate(inference_loader), desc="Batches ..."):
            messages = messages.cuda(device)
            images = images.cuda(device)

            msg = torch.cat([images, messages], 1)
            encode_img = encoder(msg)

            # normalizing
            if norm_train == 'clamp':
                encode_img_c = torch.clamp(encode_img, 0, 1)
            else:
                encode_img_c = encode_img

            # decode
            decode_img = decoder(encode_img_c)

            # compute psnr, ssim and accuracy
            acc = get_message_accuracy(messages, decode_img, bpp=bpp)
            psnr, ssim = compute_image_score(images, encode_img_c, cal_psnr, cal_ssim)

            acc_list.append(acc)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

            # store the watermarked images
            for j in range(encode_img_c.shape[0]):
                current_filename = filenames[j]
                save_path_full = output_save_dir / current_filename
                save_path_full = save_path_full.with_suffix(".png")  # store in PNG
                #print(f"Saving watermarked image to {save_path_full}")
                #save_image(enco_images_clamped[j].cpu(), save_path_full, normalize=True, range=(0, 255)) #
                #pil = to_pil_image(encode_img_c[j].to(torch.uint8))  # C,H,W -> PIL espera C,H,W uint8
                pil = to_pil_image(encode_img_c[j].detach().cpu()) 
                #pil = torch.clamp(torch.round(enco_images_clamped[j]), 0, 255).to(torch.uint8)
                #pil = to_pil_image(pil.cpu())  # C,H,W -> PIL espera C,H,W uint8
                pil.save(save_path_full, format="PNG", compress_level=0)
    encoder.train()
    decoder.train()

    return np.mean(acc_list), np.mean(psnr_list), np.mean(ssim_list), np.std(acc_list), np.std(psnr_list), np.std(ssim_list)

class data_inference(Dataset):
    """
    A custom dataset class for handling images and secret messages for inference.
    """
    def __init__(
        self,
        data_path: str,
        db_path: str = None,
        image_size: tuple = (256, 256),
        dataset: str = 'facelab_london',
        bpp = 1,

    ):
        self.data_path = data_path
        self.db_path = db_path
        self.im_size = image_size
        self.bpp = bpp

        assert dataset in ['facelab_london', 'CFD', 'ONOT', 'LFW'], "Invalid DataSet. only support ['facelab_london', 'CFD', 'ONOT']."

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
        img_cover = load_and_preprocess_image(img_cover_path, self.im_size[0])

        # --- Load the message (watermark) from the database ---
        watermark_str = None
        messages = None # Initialize messages to None or a default zero array

        self.n_channels, self.msg_range, self.mapping_factor = get_mapping_params(self.bpp)
        
        try:
            # Execute SQL query to get the watermark data for the current filename
            self.cursor.execute("SELECT watermark_data FROM watermarks WHERE image_filename = ?", (filename,))
            result = self.cursor.fetchone() # Fetch the result

            if result:
                watermark_str = result[0] # The watermark string is the first element of the result tuple
            else:
                # raise and error to stop the process, not returning any watermark
                raise ValueError(f"Watermark not found for filename: {filename} in {self.db_path}")

        except sqlite3.Error as e:
            raise RuntimeError(f"Database error: {e}")

        if watermark_str:
            # Convert the watermark string ('0101...') to a NumPy array of floats (0.0 or 1.0)
            messages_flat = np.array(list(watermark_str)).astype(np.float32)
            messages = symbols_to_message_image(messages_flat, self.bpp, (self.im_size[0], self.im_size[1]))
        
        return img_cover, messages, filename
    
    def __len__(self) -> int:
        return len(self.files_list)

class Celeba_hq(Dataset):
    def __init__(self,
                 im_size=(256, 256), bpp=4,
                 path=None):
        self.transform = T.ToTensor()
        self.files = natsorted(
                sorted(glob(path+"/*."+"png")))
        self.im_size = im_size
        self.bpp = bpp
        
    def generate_message_image(self):
        """
        Generates a message array as a NumPy array for a given bpp value,
        based on the specified channels and pixel value ranges.
        """
        m_size_flat = self.im_size[0] * self.im_size[1]
        
        if self.bpp == 1:
            n_channels = 1
            msg_range_high = 2  # Range [0, 1]
            mapping_factor = 255
        elif self.bpp == 2:
            n_channels = 2
            msg_range_high = 2  # Range [0, 1]
            mapping_factor = 255
        elif self.bpp == 3:
            n_channels = 3
            msg_range_high = 2  # Range [0, 1]
            mapping_factor = 255
        elif self.bpp == 4:
            n_channels = 1
            msg_range_high = 16  # Range [0, 15]
            mapping_factor = 256 / msg_range_high
        elif self.bpp == 6:
            n_channels = 3
            msg_range_high = 4  # Range [0, 3]
            mapping_factor = 256 / msg_range_high
        elif self.bpp == 8:
            n_channels = 2
            msg_range_high = 16  # Range [0, 15]
            mapping_factor = 256 / msg_range_high
        else:
            raise ValueError(f"Unsupported bpp: {self.bpp}")
        
        # Generate the values for the message
        total_values = int(m_size_flat * n_channels)
        message_flat = np.random.randint(low=0, high=msg_range_high, size=total_values)
        mapped_message = message_flat * mapping_factor
        
        # Reshape the flat array to the correct image shape
        #message_image = mapped_message.reshape(self.im_size[0], self.im_size[1], n_channels)
        message_image = mapped_message.reshape(n_channels, self.im_size[0], self.im_size[1]).astype(np.float32)

        # Convert the NumPy array to a PyTorch tensor
        message_image = torch.from_numpy(message_image).float()

        return message_image/255.0 

    def __getitem__(self, index):
        # Get new image
        img = Image.open(self.files[index]).convert('RGB')
        img_cover = ImageOps.fit(img, self.im_size)
        img_cover = self.transform(img_cover)
        img_cover = img_cover#/255.0

        # Generate and process the secret message
        message_img = self.generate_message_image()

        return img_cover, message_img
    
    def __len__(self):
        return len(self.files)
    


