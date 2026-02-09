
import os
import numpy as np
import sqlite3
from pathlib import Path
import torch
from torch.utils.data import Dataset

from watermarking.StegFormer.utils import symbols_to_message_image
from utils import is_image_file, load_and_preprocess_image

class WatermarkedDataset(Dataset):
    """
    Dataset for loading watermarked images, their corresponding original images, and the associated watermarks from a database.
    This dataset is designed to be used for training or evaluating, where each sample consists of:
    - img_wm: The watermarked image (as a tensor).
    - img_ori: The original image (as a tensor).
    - watermark: The watermark associated with the image (as a tensor).
    The returned files are filtered by the identities list, always provided.
    """
    def __init__(self,
                data_path: str,
                db_path: str = None,
                db_name: str = 'watermarks_BBP_1_65536_500.db',
                image_size: tuple = (256, 256),
                dataset: str = 'CFD',
                identities: list = None,
                train_dataset: str = 'celeba_hq',
                wm_algorithm: str = 'StegFormer',
                experiment_name: str = '1_1_255_w16_learn_im',
                IMG_EXTENSION: str = 'npy',
                max_images: int = None,
                ori_data_path: str = None,
                bpp: int = 1,
                ):
        self.bpp = bpp
        self.wm_algorithm = wm_algorithm
        self.im_size = image_size
        self.max_images = max_images
        self.identities = identities
        self.IMG_EXTENSION = IMG_EXTENSION

        if dataset == 'facelab_london':
            self.org_ext = '.jpg'
        elif dataset == 'CFD':
            self.org_ext = '.jpg'
        elif dataset == 'ONOT' or dataset == 'ONOT_set1':
            self.org_ext = '.png'
        elif dataset == 'LFW':
            self.org_ext = '.jpg'
        elif dataset == 'SCface':
            self.org_ext = '.jpg'

        # paths to search the images
        # for original images without watermark
        self.ori_data_path = os.path.join(ori_data_path,
                                          dataset,
                                          'processed',
                                          'test')
        # for watermarked images
        self.data_path = os.path.join(data_path, 
                                      wm_algorithm, 
                                      experiment_name,
                                      'inference',
                                      train_dataset,
                                      dataset,
                                      'watermarked_images') 
        # to get the watermarks from the database
        self.db_path = os.path.join(db_path,
                                    dataset,
                                    'processed',
                                    'watermarks',
                                    db_name)
        
                # for attack images

        print("[*]Loading Images from {}".format(self.data_path))
        self.image_paths = sorted(self.select_dataset())
        print('[*]{} images have been loaded'.format(len(self.image_paths)))

        # Establish SQLite database connection.
        # This connection will be persistent throughout the dataset's life.
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def search_watermark_in_db(self, filename: str) -> np.ndarray:
        # --- Load the message (watermark) from the database ---
        watermark_str = None
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
            # Convert the watermark string ('0101...') or ('2,0,2,4,5') to a NumPy array of floats (0.0 or 1.0)
            if ',' in watermark_str:
                messages_flat = np.array([float(bit) for bit in watermark_str.split(',')]).astype(np.float32)
            else:
                messages_flat = np.array(list(watermark_str)).astype(np.float32)
        else:
            raise ValueError(f"Watermark string is empty for filename: {filename} in {self.db_path}")

        return messages_flat
    
    def select_dataset(self):
        image_paths = []
        assert os.path.isdir(self.data_path), '[*]{} is not a valid directory'.format(self.data_path)
        for root, _, fnames in sorted(os.walk(self.data_path)):
            for fname in fnames:
                if is_image_file(fname, self.IMG_EXTENSION) and (fname.split('.')[0] in self.identities):
                    path = os.path.join(root,fname)
                    image_paths.append(path)
        #print("[*]Loaded {} images with mark".format(len(image_paths)))
        
        if self.max_images is not None:
            image_paths = image_paths[:self.max_images]

        return image_paths

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]

        # Load and process the cover image
        img_wm = load_and_preprocess_image(image_path, self.im_size[0], self.IMG_EXTENSION)
        filename = os.path.basename(image_path) 
        #identity = filename.split('_')[0]
        old_ext = '.'+filename.split('.')[-1]
        id_wm = filename.replace(old_ext, self.org_ext)
        real_wm = self.search_watermark_in_db(id_wm) 

        if self.wm_algorithm.lower() == 'stegformer':
            messages = symbols_to_message_image(real_wm, self.bpp, (self.im_size[0], self.im_size[1])) 
        elif self.wm_algorithm.lower() == 'stegaformer':
            pass       

        org_full_path = os.path.join(self.ori_data_path, filename.replace(old_ext, self.org_ext))
        ori_img = load_and_preprocess_image(org_full_path, self.im_size[0], 'png')

        return img_wm, filename, messages, ori_img
    
class FaceAttackedDataset(Dataset):
    def __init__(self,
                data_path: str,
                image_size: tuple = (256, 256),
                dataset: str = 'CFD',
                train_dataset: str = 'celeba_hq',
                wm_algorithm: str = 'StegFormer',
                experiment_name: str = '1_1_255_w16_learn_im',
                IMG_EXTENSION: str = 'npy',
                max_images: int = None,
                ori_data_path: str = None,
                bpp: int = 1,
                set_name: str = None,
                experiment_dir: str = None,
                experiment_name_attack: str = None,
                id_experiment_attack: str = None,
                face_model_attacked: str = None,
                attacked_dataset: str = None,
                ):
        self.bpp = bpp
        self.wm_algorithm = wm_algorithm
        self.im_size = image_size
        self.max_images = max_images
        self.IMG_EXTENSION = IMG_EXTENSION

        if dataset == 'facelab_london':
            self.org_ext = '.jpg'
        elif dataset == 'CFD':
            self.org_ext = '.jpg'
        elif dataset == 'ONOT' or dataset == 'ONOT_set1':
            self.org_ext = '.png'
        elif dataset == 'LFW':
            self.org_ext = '.jpg'
        elif dataset == 'SCface':
            self.org_ext = '.jpg'

        # paths to search the images
        # for templates images witout watermark
        self.template_data_path = os.path.join(ori_data_path,
                                               dataset,
                                               'processed',
                                               'templates')
        # for watermarked images
        self.data_path_wm = os.path.join(data_path, 
                                      wm_algorithm, 
                                      experiment_name,
                                      'inference',
                                      train_dataset,
                                      dataset,
                                      'watermarked_images') 
        
        # for attack images
        self.attack_data_path = os.path.join(experiment_dir,
                                            wm_algorithm,
                                            experiment_name,
                                            train_dataset,
                                            attacked_dataset,
                                            face_model_attacked,
                                            experiment_name_attack,
                                            id_experiment_attack,
                                            'attacked_samples',
                                            set_name)
        #print("[*]Loading Template Images from {}".format(self.template_data_path))
        #print("[*]Loading Watermarked Images from {}".format(self.data_path_wm))
        #print("[*]Loading Attacked Images from {}".format(self.attack_data_path))

        self.template_paths = sorted(self.get_template_paths())
        template_filenames = [os.path.basename(path).split('.')[0] for path in self.template_paths]    
        self.wm_paths = self.get_watermarked_paths(template_filenames)
        self.attack_paths = self.get_attacked_paths(template_filenames)

        print("[*] Loaded in total {} files for the FR dataset evaluation".format(len(self.template_paths))) 

    def get_attacked_paths(self, template_namefiles):
        image_paths = []
        assert os.path.isdir(self.attack_data_path), '[*]{} is not a valid directory'.format(self.attack_data_path)
        for root, _, fnames in sorted(os.walk(self.attack_data_path)):
            for fname in fnames:
                if fname.split('_')[0] in template_namefiles and is_image_file(fname, self.IMG_EXTENSION):
                    path = os.path.join(root,fname)
                    image_paths.append(path)
        return image_paths

    def get_watermarked_paths(self, template_namefiles):
        image_paths = []
        assert os.path.isdir(self.data_path_wm), '[*]{} is not a valid directory'.format(self.data_path_wm)
        for root, _, fnames in sorted(os.walk(self.data_path_wm)):
            for fname in fnames:
                if fname.split('_')[0] in template_namefiles and is_image_file(fname, self.IMG_EXTENSION):
                    path = os.path.join(root,fname)
                    image_paths.append(path)
        return image_paths
    
    def get_template_paths(self):
        template_paths = []
        assert os.path.isdir(self.template_data_path), '[*]{} is not a valid directory'.format(self.template_data_path)
        for root, _, fnames in sorted(os.walk(self.template_data_path)):
            for fname in fnames:
                path = os.path.join(root,fname)
                template_paths.append(path)
        return template_paths

    def __len__(self):
        return len(self.template_paths)
    
    def __getitem__(self, index):
        wm_path = self.wm_paths[index]
        template_path = self.template_paths[index]
        attack_path = self.attack_paths[index]

        # load and process the watermarked image (to get the filename and the identity)
        wm_img = load_and_preprocess_image(wm_path.replace('.png', '.npy'), 
                                             self.im_size[0], self.IMG_EXTENSION)
        
        # get the template image
        template_img = load_and_preprocess_image(template_path, 
                                           self.im_size[0], 'png')
        
        # get the attacked image
        attacked_img = load_and_preprocess_image(attack_path.replace('.png', '.npy'), 
                                           self.im_size[0], self.IMG_EXTENSION)
        
        filename = os.path.basename(template_path).split('.')[0]

        return template_img, wm_img, attacked_img, filename

if __name__ == '__main__':
    dataset = WatermarkedDataset(
        data_path='/app/output/watermarking', 
        db_path='/app/facial_data',
        db_name='watermarks_BBP_1_65536_500.db',
        dataset='CFD',
        identities = None,
        train_dataset='celeba_hq',
        wm_algorithm='stegaformer',
        experiment_name='1_1_255_w16_learn_im',
        image_size=(256, 256),
        IMG_EXTENSION='npy',
        ori_data_path='/app/facial_data',
    )

    for i in range(len(dataset)):
        img_wm, identity, real_wm, ori_img = dataset[i]
        print(f"Image {i}: Identity={identity}, Watermark shape={real_wm.shape}")
        print(f"Image ORG shape: {ori_img.shape}")