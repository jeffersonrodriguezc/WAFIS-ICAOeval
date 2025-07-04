import os
import glob
import sqlite3
import argparse
import numpy as np
from pathlib import Path

def extract_user_id(filename: str) -> str:
    """Extrae la ID de usuario de un nombre de archivo como '001_image_variant.png'."""
    # first part of the filename before the underscore is considered the user ID
    return filename.split('_')[0]

def generate_perturbed_watermark(base_watermark: np.ndarray, user_seed: int, flip_bits_count: int) -> np.ndarray:
    """
    Generate a perturbed version of the base watermark by flipping a specified number of bits based on a user seed.
    """
    watermark_flat = base_watermark.flatten()
    watermark_length = watermark_flat.shape[0]

    rng = np.random.default_rng(user_seed)
    
    # Ensure flip_bits_count does not exceed the watermark length
    flip_bits_count = min(flip_bits_count, watermark_length)

    # Select random indices to flip bits
    indices_to_flip = rng.choice(watermark_length, size=flip_bits_count, replace=False)

    user_watermark_flat = watermark_flat.copy()
    # Flip the bits at the selected indices with XOR operation with 1
    # WARNING this is only for binary watermarks (0s and 1s)
    # If the watermark is not binary, this will not work as expected
    user_watermark_flat[indices_to_flip] = 1 - user_watermark_flat[indices_to_flip]

    return user_watermark_flat

def create_watermark_db(output_filepath: str):
    """Create a SQLite database to store watermarks."""
    # Connect to the SQLite database (it will be created if it doesn't exist)
    conn = sqlite3.connect(output_filepath)
    cursor = conn.cursor()

    # Create a table to store watermarks
    # Store the watermark data as a string of 0s and 1s for simplicity
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS watermarks (
            image_filename TEXT PRIMARY KEY,
            watermark_data TEXT
        )
    ''')
    conn.commit()

    return conn, cursor

def main(
    data_root_path: str,
    dataset_name: str,
    message_n: int, # base lenght of message (4096 or 64*64)
    message_l: int, # message_size (e.g., 16*bpp)
    flip_bits_count: int, # number of bits to flip for user-specific watermark
    bbp: int = 1, # bits per pixel, assuming bpp=1
    overwrite_existing: bool = False
):
    """
    Generate fixed watermarks for inference based on user IDs in the dataset and save them to a SQLite DB.
    The watermarks are generated by perturbing a base global watermark with a user-specific seed.
    The base size of the watermark is defined by message_n and message_l. Was designed for a image size of 256x256 with bpp=1.
    """
    
    test_data_path = os.path.join(data_root_path, dataset_name, 'processed', 'test')
    output_dir = os.path.join(data_root_path, dataset_name, 'processed', 'watermarks')
    
    if not os.path.exists(test_data_path):
        print(f"Error: The test data path does not exist: {test_data_path}")
        return

    # Define the shape and length of the watermark based on message_n and message_l
    BBP = bbp # Assuming message_l=16 y image_size=256x256 -> 16*4096 / (256*256) = 1
    WATERMARK_LENGTH = message_n * (message_l*bbp) # 65536, bbp=1

    output_filename = f"watermarks_BBP_{BBP}_{WATERMARK_LENGTH}_{flip_bits_count}.db"
    output_filepath = os.path.join(output_dir, output_filename)

    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(output_filepath) and not overwrite_existing:
        print(f"Warning the db file '{output_filename}' already exists."
              "Use --overwrite to generate a new file.")
        return

    # Generate a global base watermark with a fixed seed
    # This will be the base watermark that is perturbed for each user
    global_base_rng = np.random.default_rng(seed=42) # Fixed seed for reproducibility
    global_base_watermark = global_base_rng.integers(low=0, high=2, size=WATERMARK_LENGTH)

    # Dicctionary to store user-specific watermarks
    user_watermarks_map = {} # {user_id: np.ndarray(65536)}

    # Obtain all image files in the test data directory
    if dataset_name == 'facelab_london':
        image_files = glob.glob(os.path.join(test_data_path, '*.jpg'))
    elif dataset_name == 'CFD':
        image_files = glob.glob(os.path.join(test_data_path, '*.jpg'))
    elif dataset_name == 'ONOT':
        image_files = glob.glob(os.path.join(test_data_path, '*.png'))
    
    if not image_files:
        print(f"No images found in {test_data_path}.")
        return

    print(f"Was found {len(image_files)} images in {test_data_path}")

    # Create a SQLite database to store the watermarks
    conn, cursor = create_watermark_db(output_filepath)
    # Iterate over each image file to extract user IDs and generate watermarks
    for img_path in sorted(image_files): # Ordered to ensure consistent output (1 image per user_id)
        filename = os.path.basename(img_path)
        user_id = extract_user_id(filename) # example: '001_image_variant.png' -> '001'
        # Convert user_id to an integer seed
        try:
            user_seed = int(user_id)
        except ValueError:
            # Alternatively, use a hash of the user_id string
            user_seed = sum(ord(c) for c in user_id)
        
        # Generate a user-specific watermark by perturbing the global base watermark
        user_specific_watermark = generate_perturbed_watermark(
            global_base_watermark, user_seed, flip_bits_count
        )
        
        watermark_str = ''.join(map(str, user_specific_watermark))
        cursor.execute("INSERT INTO watermarks (image_filename, watermark_data) VALUES (?, ?)", (filename, watermark_str))
        conn.commit()
    conn.close() # Close the database connection

    print(f"Was generated and stored {len(image_files)} user-specific watermarks in the database.")
    print(f"The watermarks were generated based on a global base watermark with a fixed seed of 42 and {flip_bits_count} bits flipped for each user.")
    print(f"Watermarks were saved to the SQLite database: {output_filepath}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate user-specific watermarks for a dataset.')
    parser.add_argument('--data_root', type=str, default='../datasets',
                        help='Root folder containing the dataset (e.g., "datasets").')
    parser.add_argument('--dataset_name', type=str, default='facelab_london',
                        help='Dataset name (e.g., "facelab_london").')
    parser.add_argument('--message_n', type=int, default=4096, # 64*64 used in stegaformer
                        help='Number of values in the message per segment.')
    parser.add_argument('--message_l', type=int, default=16, # 16*bpp, assuming bpp=1
                        help='Lenght of each segment of the message.')
    parser.add_argument('--flip_bits_count', type=int, default=500,
                        help='Number of bits to flip in the watermark for user-specific perturbation.')
    parser.add_argument('--overwrite', action='store_true',
                        help='OVerwrite existing watermark database if it exists.')

    args = parser.parse_args()

    main(
        data_root_path=args.data_root,
        dataset_name=args.dataset_name,
        message_n=args.message_n,
        message_l=args.message_l,
        flip_bits_count=args.flip_bits_count,
        overwrite_existing=args.overwrite
    )