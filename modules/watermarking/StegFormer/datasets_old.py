import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from natsort import natsorted
from PIL import Image
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageOps
import numpy as np
import config
args = config.Args()

transform = T.Compose([
    T.RandomCrop(128),
    T.RandomHorizontalFlip(),
    T.ToTensor()
])

# albumentations
transform_A = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.RandomRotate90(),
    A.HorizontalFlip(),
    A.augmentations.transforms.ChannelShuffle(0.3),
    ToTensorV2()
])

transform_A_valid = A.Compose([
    A.CenterCrop(width=256, height=256),
    ToTensorV2()
])

transform_A_test = A.Compose([
    A.CenterCrop(width=1024, height=1024),
    ToTensorV2()
])

transform_A_test_256 = A.Compose([
    A.PadIfNeeded(min_width=256,min_height=256),
    A.CenterCrop(width=256, height=256),
    ToTensorV2()
])

def reverse_message_image(message_image, bpp):
    """
    Reverses the process to extract the original message array from a mapped image.
    """
    if bpp == 1:
        msg_range_high = 2  # Range [0, 1]
        mapping_factor = 255
    elif bpp == 2:
        msg_range_high = 2  # Range [0, 1]
        mapping_factor = 255
    elif bpp == 3:
        msg_range_high = 2  # Range [0, 1]
        mapping_factor = 255
    elif bpp == 4:
        msg_range_high = 16  # Range [0, 15]
        mapping_factor = 256 / msg_range_high
    elif bpp == 6:
        msg_range_high = 4  # Range [0, 3]
        mapping_factor = 256 / msg_range_high
    elif bpp == 8:
        msg_range_high = 16  # Range [0, 15]
        mapping_factor = 256 / msg_range_high
    else:
        raise ValueError(f"Unsupported bpp: {bpp}")
    
    x = message_image

    x255 = torch.clamp(x, 0.0, 1.0) * 255.0
    symbols = torch.round(x255 / mapping_factor).long()
    symbols = torch.clamp(symbols, 0, msg_range_high - 1)

    return symbols

class Celeba_hq(Dataset):
    def __init__(self,
                 im_size=(256, 256), bpp=4,
                 path=None):
        self.transform = T.ToTensor()
        self.files = natsorted(
                sorted(glob.glob(path+"/*."+"png")))
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

class DIV2K_Dataset(Dataset):
    def __init__(self, transforms_=None, mode='train'):
        self.transform = transforms_
        self.mode = mode
        if mode == 'train':
            self.files = natsorted(
                sorted(glob.glob(DIV2K_path+"/DIV2K_train_HR"+"/*."+"png")))
        else:
            self.files = natsorted(
                sorted(glob.glob(DIV2K_path+"/DIV2K_valid_HR"+"/*."+"png")))

    def __getitem__(self, index):
        img = cv2.imread(self.files[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
        trans_img = self.transform(image=img)
        item = trans_img['image']
        item = item/255.0
        return item

    def __len__(self):
        return len(self.files)

class Flickr2K_Dataset(Dataset):
    def __init__(self, transforms_=None, mode='train'):
        self.transform = transforms_
        self.mode = mode
        self.files = natsorted(
            sorted(glob.glob(Flickr2K_path+"/Flickr2K_HR"+"/*."+"png")))

    def __getitem__(self, index):
        img = cv2.imread(self.files[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
        trans_img = self.transform(image=img)
        item= trans_img['image']
        item=item/255.0
        return item

    def __len__(self):
        return len(self.files)

class COCO_Test_Dataset(Dataset):
    def __init__(self, transforms_=None):
        self.transform = transforms_
        self.files = natsorted(
                sorted(glob.glob("/home/whq135/dataset/COCO2017/test2017"+"/*."+"jpg")))

    def __getitem__(self, index):
        img = cv2.imread(self.files[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
        trans_img = self.transform(image=img)
        item= trans_img['image']
        item=item/255.0
        return item

    def __len__(self):
        return len(self.files)

"""DIV2K_train_cover_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A, mode="train"),
    batch_size=args.single_batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=8,
    drop_last=True
)

DIV2K_train_secret_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A, mode="train"),
    batch_size=args.single_batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=8,
    drop_last=True
)

DIV2K_val_cover_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A_valid, mode="val"),
    batch_size=args.single_batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=2,
    drop_last=True
)

DIV2K_val_secret_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A_valid, mode="val"),
    batch_size=args.single_batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=2,
    drop_last=True
)

DIV2K_test_cover_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A_test, mode="val"),
    batch_size=1,
    shuffle=True,
    pin_memory=True,
    num_workers=1,
    drop_last=True
)

DIV2K_test_secret_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A_test, mode="val"),
    batch_size=1,
    shuffle=False,
    pin_memory=True,
    num_workers=1,
    drop_last=True
)

DIV2K_multi_train_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A, mode="train"),
    batch_size=args.multi_batch_iteration,
    shuffle=True,
    pin_memory=True,
    num_workers=16,
    drop_last=True
)

DIV2K_multi_val_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A_valid, mode="val"),
    batch_size=args.multi_batch_iteration,
    shuffle=True,
    pin_memory=True,
    num_workers=16,
    drop_last=True
)

DIV2K_multi_test_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A_test, mode="val"),
    batch_size=args.test_multi_batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=1,
    drop_last=True
)

COCO_test_multi_loader = DataLoader(
    COCO_Test_Dataset(transforms_=transform_A_test_256),
    batch_size=args.test_multi_batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=1,
    drop_last=True
)

COCO_test_cover_loader = DataLoader(
    COCO_Test_Dataset(transforms_=transform_A_test_256),
    batch_size=1,
    shuffle=True,
    pin_memory=True,
    num_workers=1,
    drop_last=True
)

COCO_test_secret_loader = DataLoader(
    COCO_Test_Dataset(transforms_=transform_A_test_256),
    batch_size=1,
    shuffle=True,
    pin_memory=False,
    num_workers=1,
    drop_last=True
)

Flickr2K_multi_train_loader = DataLoader(
    Flickr2K_Dataset(transforms_=transform_A),
    batch_size=40,
    shuffle=True,
    pin_memory=True,
    num_workers=16,
    drop_last=True
)

Flickr2K_train_cover_loader = DataLoader(
    Flickr2K_Dataset(transforms_=transform_A),
    batch_size=batchsize,
    shuffle=True,
    pin_memory=True,
    num_workers=2,
    drop_last=True
)

Flickr2K_train_secret_loader = DataLoader(
    Flickr2K_Dataset(transforms_=transform_A),
    batch_size=batchsize,
    shuffle=True,
    pin_memory=True,
    num_workers=2,
    drop_last=True
)"""
