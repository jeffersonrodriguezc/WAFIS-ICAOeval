import os 
import argparse 
import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image, ImageOps

def visualize_watermark_difference(
    exp_name: str,
    dataset: str,
    train_dataset: str,
    model_name: str,
    num_images_to_visualize: int,
    output_viz_dir: str
) -> None:
    """
    Generates visualizations of the difference between original and watermarked images.

    This function loads original and watermarked images, calculates their absolute
    difference (residual), and displays them side-by-side. The residual image
    highlights the areas where the watermark has introduced changes.

    Args:
        exp_name (str): The name of the experiment. This is used to construct the
                        path to the watermarked images and results.
        dataset (str): The name of the dataset. This is used to construct the
                       paths to both original and watermarked images.
        train_dataset (str): The name of the dataset used for training. This is part of the path
                             to the watermarked images.
        model_name (str): The name of the model used for watermarking. This is part of the path
                         to the watermarked images.
        num_images_to_visualize (int): The number of image pairs (original and watermarked)
                                       to process and visualize.
        output_viz_dir (str): The directory path where the generated visualization
                              images will be saved.
    """
    # Construct base paths for input and output directories based on conventions.
    # The 'base_output_dir' is where watermarked images and results are expected to be.
    base_output_dir = f'../experiments/output/watermarking/{model_name}/{exp_name}/inference/{train_dataset}/{dataset}'
  
    # The 'original_images_dir' points to the location of the clean, original images.
    original_images_dir = f'../datasets/{dataset}/processed/test'
    # The 'watermarked_images_dir' points to the directory containing the images
    # after the watermark has been embedded.
    watermarked_images_dir = f'{base_output_dir}/watermarked_images'

    # Create the output directory for visualizations if it does not already exist.
    # 'exist_ok=True' prevents an error if the directory already exists.
    output_viz_dir = os.path.join(output_viz_dir, f'{model_name}', 'differences', f'{exp_name}_{train_dataset}_{dataset}')
    os.makedirs(output_viz_dir, exist_ok=True)

    # Get a sorted list of image files from the watermarked images directory.
    # It filters for common image file extensions. Sorting ensures consistent processing order.
    
    if args.dataset == 'ONOT':
        image_files = sorted([f for f in os.listdir(original_images_dir) if f.endswith(('.png', '.PNG'))])
    else:
        image_files = sorted([f for f in os.listdir(original_images_dir) if f.endswith(('.jpg', '.jpeg'))])

    # Check if any image files were found. If not, print a message and exit the function.
    if not image_files:
        print(f"No images found in {original_images_dir}")
        return

    # Loop through a subset of the found image files up to 'num_images_to_visualize'.
    for i, filename in enumerate(image_files[:num_images_to_visualize]):
        try:
            # Construct full paths for the current original and watermarked images.
            original_img_path = os.path.join(original_images_dir, filename)
            ext = filename.split('.')[-1]
            if args.dataset == 'ONOT':
                # ONOT uses png images in test set
                watermarked_img_path = os.path.join(watermarked_images_dir, filename)
            else:
                watermarked_img_path = os.path.join(watermarked_images_dir, filename.replace(ext, 'png'))

            # Open the images using Pillow and convert them to RGB format.
            # Converting to RGB ensures consistent channel structure for calculations.
            original_img = Image.open(original_img_path).convert('RGB')
            original_img = ImageOps.fit(original_img, (256, 256))
            watermarked_img = Image.open(watermarked_img_path).convert('RGB')
            
            # Convert PIL Image objects to NumPy arrays.
            # This allows for efficient pixel-wise mathematical operations.
            original_np = np.array(original_img)
            watermarked_np = np.array(watermarked_img)
            #print('max original:', original_np.max(), 'min:', original_np.min())
            #print('max watermarked:', watermarked_np.max(), 'min:', watermarked_np.min())

            # Calculate the residual image by taking the absolute difference
            # between the original and watermarked image arrays.
            # This highlights where pixel values have changed due to watermarking.
            residual_np = np.abs(original_np.astype(float) - watermarked_np.astype(float))

            # Normalize the residual image for better visual representation.
            # This scales pixel values to a 0-1 range, making differences more apparent.
            if residual_np.max() > residual_np.min(): # Avoid division by zero if all values are identical
                residual_normalized = (residual_np - residual_np.min()) / (residual_np.max() - residual_np.min())
            else:
                residual_normalized = np.zeros_like(residual_np) # If no difference, set to black

            magnitude_of_change = np.sum(np.abs(original_np.astype(float) - watermarked_np.astype(float)), axis=-1)
            # normalize the magnitude of change to a 0-1 range for visualization
            if magnitude_of_change.max() > magnitude_of_change.min():
                magnitude_normalized = (magnitude_of_change - magnitude_of_change.min()) / (magnitude_of_change.max() - magnitude_of_change.min())
            else:
                magnitude_normalized = np.zeros_like(magnitude_of_change, dtype=float)


            # Create a new Matplotlib figure with a specified size.
            plt.figure(figsize=(24, 6))

            # Subplot 1: Original Image
            plt.subplot(1, 4, 1) # 1 row, 4 columns, 1st subplot
            plt.imshow(original_img) # Display the original image
            plt.title('Original Image') # Set title
            plt.axis('off') # Hide axes for cleaner image display

            # Subplot 2: Watermarked Image
            plt.subplot(1, 4, 2) # 1 row, 4 columns, 2nd subplot
            plt.imshow(watermarked_img) # Display the watermarked image
            plt.title('Watermarked Image') # Set title
            plt.axis('off') # Hide axes

            # Subplot 3: Residual (Difference) Image
            plt.subplot(1, 4, 3) # 1 row, 4 columns, 3rd subplot
            plt.imshow(residual_normalized) # Display the normalized residual image
            plt.title('Residual (Difference)') # Set title
            plt.axis('off') # Hide axes

            # Subplot 4: Heatmap of Change Magnitude 
            plt.subplot(1, 4, 4) # 1 row, 4 columns, 4th subplot
            plt.imshow(magnitude_normalized, cmap='Reds') # 'hot' o 'Reds'
            #plt.colorbar(label='Magnitude of Change (Normalized)', orientation='horizontal') 
            plt.title('Heatmap of Change Magnitude')
            plt.axis('off')

            # Set a super title for the entire figure, including experiment and dataset details.
            plt.suptitle(f'Experiment: {exp_name}, Dataset: {dataset}, Model: {model_name}', fontsize=16)

            # Construct the full path to save the generated comparison image.
            save_path = os.path.join(output_viz_dir, f'comparison_{i+1}_{filename}')
            # Save the figure to the specified path. 'bbox_inches='tight'' ensures no extra whitespace.
            plt.savefig(save_path, bbox_inches='tight')
            plt.close() # Close the figure to free up memory.
            print(f"Comparison saved for {filename} to {save_path}")

        except FileNotFoundError as fnfe:
            # Handle cases where an expected image file is not found.
            print(f"File not found for visualization: {fnfe}")
        except Exception as e:
            # Catch any other unexpected errors during image processing.
            print(f"Error processing {filename}: {e}")

# This block ensures that the code inside it only runs when the script is executed directly,
# not when it's imported as a module.
if __name__ == '__main__':
    # Initialize the ArgumentParser to handle command-line arguments.
    parser = argparse.ArgumentParser(description="Generate watermark difference visualizations.")
    
    # Define command-line arguments with their types, default values, and help messages.
    parser.add_argument('--exp_name', type=str, default='1_2_255_w16_learn_im',
                        help='Name of the experiment to visualize.')
    parser.add_argument('--dataset', type=str, default='facelab_london',
                        choices=['facelab_london', 'CFD', 'ONOT', 'LFW'],
                        help='Dataset name.')
    parser.add_argument('--train_dataset', type=str, default='celeba_hq',
                        choices=['celeba_hq', 'coco'],
                        help='Dataset used for training.')
    parser.add_argument('--model_name', type=str, default='stegaformer',
                        help='Name of the model used for watermarking.')
    parser.add_argument('--num_images', type=int, default=5,
                        help='Number of images to visualize.')
    parser.add_argument('--output_dir', type=str, default='../evaluation/visualizations',
                        help='Directory to save the visualizations.')
    
    # Parse the arguments provided by the user.
    args = parser.parse_args()

    # Call the main visualization function with the parsed arguments.
    visualize_watermark_difference(
        exp_name=args.exp_name,
        dataset=args.dataset,
        train_dataset=args.train_dataset,
        model_name=args.model_name,
        num_images_to_visualize=args.num_images,
        output_viz_dir=args.output_dir
    )

#python .\visualize_watermark_difference.py --exp_name 1_1_255_w16_learn_im --dataset CFD