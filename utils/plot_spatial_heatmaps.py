import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from matplotlib.colors import TwoSlopeNorm
 
 
def compute_bpp(model_name: str, exp_name: str) -> int:
    """Determine bpp based on experiment name and model."""
    splits = exp_name.split('_')
    if model_name.lower() == 'stegaformer':
        if splits[0] == '1' and splits[1] == '1':
            bpp = 1
        elif splits[0] == '1' and splits[1] == '3':
            bpp = 3
        elif splits[0] == '3' and splits[1] == '3':
            bpp = 6
        elif splits[0] == '15' and splits[1] == '2':
            bpp = 8
        else:
            raise ValueError(f'Unknown experiment name format: {exp_name}')
    elif model_name.lower() == 'stegformer':
        if splits[0] == '1' and splits[1] == '1':
            bpp = 1
        elif splits[0] == '3' and splits[1] == '3':
            bpp = 3
        elif splits[0] == '3' and splits[1] == '6':
            bpp = 6
        elif splits[0] == '2' and splits[1] == '8':
            bpp = 8
        else:
            raise ValueError(f'Unknown experiment name format: {exp_name}')
    else:
        raise ValueError(f'Unknown model name: {model_name}')
    return bpp
 
 
def get_exp_name(model_name: str, bpp: int) -> str:
    """Map bpp back to experiment name format."""
    if model_name.lower() == 'stegaformer':
        mapping = {1: '1_1', 3: '1_3', 6: '3_3', 8: '15_2'}
    elif model_name.lower() == 'stegformer':
        mapping = {1: '1_1', 3: '3_3', 6: '3_6', 8: '2_8'}
    else:
        raise ValueError(f'Unknown model name: {model_name}')
 
    prefix = mapping.get(bpp)
    if prefix is None:
        raise ValueError(f'Unknown bpp: {bpp}')
 
    # Search for the full experiment name in the directory
    base_dir = f'../experiments/output/watermarking/{model_name}'
    if os.path.exists(base_dir):
        for d in os.listdir(base_dir):
            if d.startswith(prefix):
                return d
    return f'{prefix}_255_w16_learn_im'
 
 
def generate_avg_residual_heatmaps(
    datasets: list,
    train_datasets: list,
    model_name: str,
    mode: str,
    bpp: int,
    num_images: int = 50,
    normalize_mode: str = 'dataset',
    seed: int = 42,
    save_path: str = None,
):
    """
    Generate a multi-panel figure showing averaged signed residual
    heatmaps (watermarked - original) across datasets and training domains.
 
    Rows correspond to evaluation datasets, columns to training domains.
    Color encodes the mean signed change summed across RGB channels:
    red = pixel intensity increased, blue = decreased, white = no change.
 
    Parameters
    ----------
    datasets : list of str
        Evaluation datasets (e.g. ['CFD', 'facelab_london', 'ONOT_set1', 'SCface', 'LFW']).
    train_datasets : list of str
        Training domains (e.g. ['coco', 'celeba_hq']).
    model_name : str
        Watermarking model (e.g. 'stegformer').
    mode : str
        Evaluation mode (e.g. 'offline').
    bpp : int
        Bits per pixel (1, 3, 6, or 8).
    num_images : int
        Number of random images to average over per combination.
    seed : int
        Random seed for reproducibility.
    save_path : str
        Output filename without extension.
    """
    exp_name = get_exp_name(model_name, bpp)
    n_rows = len(train_datasets) 
    n_cols = len(datasets)
 
    # Short display names for subplot titles
    short_names = {
        'facelab_london': 'FaceLab London',
        'CFD': 'CFD',
        'ONOT_set1': 'ONOT Set 1',
        'SCface': 'SCface',
        'LFW': 'LFW',
        'ONOT': 'ONOT',
    }
 
    train_labels = {
        'coco': 'COCO',
        'celeba_hq': 'CelebA-HQ',
    }
 
    # First pass: compute all mean residuals and find global min/max
    mean_residuals = {}
    global_vmin = 0
    global_vmax = 0
 
    for ds in datasets:
        for train_ds in train_datasets:
            base_output_dir = f'../experiments/output/watermarking/{model_name}/{exp_name}/inference/{train_ds}/{ds}'
            original_images_dir = f'../datasets/{ds}/processed/test'
            watermarked_images_dir = f'{base_output_dir}/watermarked_images'
 
            # Get image file list
            if ds in ['ONOT', 'ONOT_set1']:
                image_files = sorted([f for f in os.listdir(original_images_dir) if f.endswith(('.png', '.PNG'))])
            else:
                image_files = sorted([f for f in os.listdir(original_images_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.PNG'))])
 
            if not image_files:
                print(f"Warning: No images found in {original_images_dir}")
                mean_residuals[(ds, train_ds)] = None
                continue
 
            # Random sample
            random.seed(seed)
            sampled_files = random.sample(image_files, min(num_images, len(image_files)))
 
            accumulated = None
            count = 0
 
            for filename in sampled_files:
                try:
                    ext = filename.split('.')[-1]
                    original_img_path = os.path.join(original_images_dir, filename)

                    original_img = Image.open(original_img_path).convert('RGB')
                    original_img = ImageOps.fit(original_img, (256, 256))
                    orig_np = np.array(original_img).astype(float)

                    if mode == 'online':
                        npy_filename = os.path.splitext(filename)[0] + '.npy'
                        watermarked_img_path = os.path.join(watermarked_images_dir, npy_filename)
                        wm_np = np.load(watermarked_img_path).astype(float)
                        if wm_np.shape[0] == 3:
                            wm_np = np.transpose(wm_np, (1, 2, 0))
                        if wm_np.max() <= 1.0:
                            wm_np = wm_np * 255.0
                    else:
                        if ds in ['ONOT', 'ONOT_set1']:
                            watermarked_img_path = os.path.join(watermarked_images_dir, filename)
                        else:
                            watermarked_img_path = os.path.join(watermarked_images_dir, filename.replace(ext, 'png'))
                        watermarked_img = Image.open(watermarked_img_path).convert('RGB')
                        wm_np = np.array(watermarked_img).astype(float)
 
                    # Signed difference summed across channels
                    diff = wm_np - orig_np
                    signed_magnitude = np.sum(diff, axis=-1)
 
                    if accumulated is None:
                        accumulated = signed_magnitude
                    else:
                        accumulated += signed_magnitude
 
                    count += 1
 
                except FileNotFoundError as e:
                    print(f"File not found: {e}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
 
            if count > 0:
                mean_res = accumulated / count
                mean_residuals[(ds, train_ds)] = mean_res
                global_vmin = min(global_vmin, mean_res.min())
                global_vmax = max(global_vmax, mean_res.max())
            else:
                mean_residuals[(ds, train_ds)] = None
 
    # Second pass: plot
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4.5 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    if normalize_mode == 'global':
        abs_max = max(abs(global_vmin), abs(global_vmax))
        if abs_max == 0:
            abs_max = 1
        norm_dict = {ds: TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max) for ds in datasets}
    elif normalize_mode == 'dataset':
        norm_dict = {}
        for ds in datasets:
            ds_vals = [mean_residuals.get((ds, td)) for td in train_datasets if mean_residuals.get((ds, td)) is not None]
            if ds_vals:
                ds_min = min(v.min() for v in ds_vals)
                ds_max = max(v.max() for v in ds_vals)
                ds_abs = max(abs(ds_min), abs(ds_max))
                if ds_abs == 0:
                    ds_abs = 1
                norm_dict[ds] = TwoSlopeNorm(vmin=-ds_abs, vcenter=0, vmax=ds_abs)
            else:
                norm_dict[ds] = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    for r_i, train_ds in enumerate(train_datasets):
        for c_i, ds in enumerate(datasets):
            ax = axes[r_i, c_i]
            mean_res = mean_residuals.get((ds, train_ds))

            if mean_res is not None:
                im = ax.imshow(mean_res, cmap='bwr', norm=norm_dict[ds])
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12)

            ds_label = short_names.get(ds, ds)
            train_label = train_labels.get(train_ds, train_ds)
            ax.set_title(f'{ds_label} — {train_label}', fontsize=11, fontweight='bold')
            ax.axis('off')

            if normalize_mode == 'dataset' and mean_res is not None:
                cbar = fig.colorbar(
                    plt.cm.ScalarMappable(norm=norm_dict[ds], cmap='bwr'),
                    ax=ax, orientation='vertical', fraction=0.046, pad=0.04,
                )
                cbar.ax.tick_params(labelsize=7)
                fig.text(0.5, -0.02,
                 'Mean signed pixel change (summed RGB). Red = increase, Blue = decrease, White = no change',
                 ha='center', fontsize=12)

    if normalize_mode == 'global':
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm_dict[datasets[0]], cmap='bwr'),
            ax=axes, orientation='horizontal',
            fraction=0.04, pad=0.06, shrink=0.6,
        )
        cbar.set_label(
            'Mean signed pixel change (summed RGB). Red = increase, Blue = decrease, White = no change',
            fontsize=12,
        )
        cbar.ax.tick_params(labelsize=9)
 
    fig.suptitle(
        f'Spatial Watermark Placement — {model_name.capitalize()} at {bpp} bpp {mode}\n'
        f'Averaged over {num_images} random images per configuration',
        fontsize=14, fontweight='bold',
    )
 
    # Output path
    if save_path is None:
        save_path = f'../evaluation/visualizations/{model_name}/general/training_domains_{mode}_bpp_{bpp}'
    else:
        save_path = f'{save_path}'

    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(f'{save_path}.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{save_path}.png', bbox_inches='tight', dpi=300)
    plt.show()
    print(f"Saved to {save_path}.pdf and {save_path}.png")
 
 
if __name__ == '__main__':
    import argparse
 
    parser = argparse.ArgumentParser(
        description="Generate averaged spatial residual heatmaps for the paper."
    )
    parser.add_argument('--datasets', nargs='+',
                        default=['CFD', 'facelab_london', 'ONOT_set1', 'SCface', 'LFW'],
                        help='List of evaluation datasets')
    parser.add_argument('--train_datasets', nargs='+',
                        default=['coco', 'celeba_hq'],
                        help='List of training domains')
    parser.add_argument('--model_name', type=str, default='stegformer',
                        help='Watermarking model name')
    parser.add_argument('--mode', type=str, default='offline',
                        help='Evaluation mode')
    parser.add_argument('--bpp', type=int, default=6,
                        help='Bits per pixel')
    parser.add_argument('--num_images', type=int, default=5,
                        help='Number of random images to average')
    parser.add_argument('--normalize_mode', type=str, default='global',
                        choices=['global', 'dataset'],
                        help='Normalize colorbar globally or per dataset')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Output filename without extension')
 
    args = parser.parse_args()
    generate_avg_residual_heatmaps(
        datasets=args.datasets,
        train_datasets=args.train_datasets,
        model_name=args.model_name,
        mode=args.mode,
        bpp=args.bpp,
        num_images=args.num_images,
        seed=args.seed,
        save_path=args.save_path,
        normalize_mode=args.normalize_mode,
    )   

#python plot_spatial_heatmaps.py \
#    --datasets CFD facelab_london ONOT_set1 SCface LFW \
#    --train_datasets coco celeba_hq \
#    --model_name stegformer \
#    --mode offline \
#    --bpp 6 \
#    --num_images 50 \
#    --normalize_mode global \
#    --save_path spatial_heatmaps