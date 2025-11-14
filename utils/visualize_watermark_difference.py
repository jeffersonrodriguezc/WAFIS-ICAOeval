import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from matplotlib.colors import TwoSlopeNorm

def compute_bpp(model_name: str, exp_name: str) -> int:
    """Función auxiliar para determinar bpp basado en el nombre del modelo."""
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

def visualize_watermark_divergence(
    exp_name: str,
    dataset: str,
    train_dataset: str,
    model_name: str,
    num_images_to_visualize: int,
    normalize: bool = False,
    recognizer: str = 'facenet',
) -> None:
    """
    Visualiza la dirección y magnitud del cambio introducido por el watermark
    para cada canal RGB, junto con el mapa de magnitud total.

    Args:
        exp_name (str): Nombre del experimento.
        dataset (str): Dataset de prueba.
        train_dataset (str): Dataset usado para entrenamiento.
        model_name (str): Modelo de watermarking.
        num_images_to_visualize (int): Número de imágenes a visualizar.
        normalize (bool): Si True, usa rangos dinámicos basados en los valores reales;
                          si False, mantiene rangos fijos [-255,255] y [0,765].
    """
    bpp = compute_bpp(model_name, exp_name)
    output_viz_dir = f'../evaluation/visualizations/{model_name}/differences/{train_dataset}/{dataset}/{recognizer}/{bpp}'
    base_output_dir = f'../experiments/output/watermarking/{model_name}/{exp_name}/inference/{train_dataset}/{dataset}'
    original_images_dir = f'../datasets/{dataset}/processed/test'
    watermarked_images_dir = f'{base_output_dir}/watermarked_images'
    os.makedirs(output_viz_dir, exist_ok=True)

    # Obtener lista de archivos
    if dataset in ['ONOT', 'ONOT_set1']:
        image_files = sorted([f for f in os.listdir(original_images_dir) if f.endswith(('.png', '.PNG'))])
    else:
        image_files = sorted([f for f in os.listdir(original_images_dir) if f.endswith(('.jpg', '.jpeg'))])

    if not image_files:
        print(f"No se encontraron imágenes en {original_images_dir}")
        return

    for i, filename in enumerate(image_files[:num_images_to_visualize]):
        try:
            ext = filename.split('.')[-1]
            original_img_path = os.path.join(original_images_dir, filename)
            if dataset in ['ONOT', 'ONOT_set1']:
                watermarked_img_path = os.path.join(watermarked_images_dir, filename)
            else:
                watermarked_img_path = os.path.join(watermarked_images_dir, filename.replace(ext, 'png'))

            # Abrir imágenes
            original_img = Image.open(original_img_path).convert('RGB')
            original_img = ImageOps.fit(original_img, (256, 256))
            watermarked_img = Image.open(watermarked_img_path).convert('RGB')
            #watermarked_img = ImageOps.fit(watermarked_img, (256, 256))

            orig_np = np.array(original_img).astype(float)
            wm_np = np.array(watermarked_img).astype(float)

            # Diferencia firmada
            diff = wm_np - orig_np  # signed difference
            abs_diff = np.abs(diff)
            mag = np.sum(abs_diff, axis=-1)

            # Calcular límites para cada canal
            channel_names = ['Red Channel (Change & Direction)', 'Green Channel (Change & Direction)', 'Blue Channel (Change & Direction)']
            cmaps = ['bwr', 'PiYG', 'bwr_r']
            channel_diffs = [diff[..., 0], diff[..., 1], diff[..., 2]]

            # Preparar figura 2x3
            fig, axes = plt.subplots(2, 3, figsize=(18, 15))
            fig.suptitle(f'BPP: {bpp} | Dataset training/testing: {train_dataset}/{dataset} | Model: {model_name}', fontsize=14)

            # -------- Fila superior --------
            # Original
            axes[0, 0].imshow(original_img)
            axes[0, 0].set_title('Original Image', fontsize=15)
            axes[0, 0].axis('off')

            # Watermarked
            axes[0, 1].imshow(watermarked_img)
            axes[0, 1].set_title('Watermarked Image', fontsize=15)
            axes[0, 1].axis('off')

            # Magnitud total
            if normalize:
                max_mag = mag.max() if mag.max() > 0 else 1
                mag_display = mag / max_mag
                label_mag = f'Magnitude (max={max_mag:.2f} / scaled to 1)'
            else:
                mag_display = mag / 765.0
                label_mag = f'Magnitude (max={mag.max():.2f} / range 0–765)'

            im_mag = axes[0, 2].imshow(mag_display, cmap='hot_r', vmin=0, vmax=1)
            axes[0, 2].set_title('Total Change Magnitude', fontsize=15)
            axes[0, 2].axis('off')

            # Barra vertical de magnitud
            cbar_mag = fig.colorbar(im_mag, ax=axes[0, 2], orientation='vertical', fraction=0.046, pad=0.04)
            cbar_mag.set_label(label_mag, fontsize=14)
            cbar_mag.ax.tick_params(labelsize=14)

            # -------- Fila inferior: Canales --------
            for j, (ch_diff, cmap, name) in enumerate(zip(channel_diffs, cmaps, channel_names)):
                if normalize:
                    vmin, vmax = ch_diff.min(), ch_diff.max()
                    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                    label = f'{name}\n(min={vmin:.1f}, max={vmax:.1f})'
                else:
                    norm = TwoSlopeNorm(vmin=-255, vcenter=0, vmax=255)
                    label = f'{name}\n(range −255 to +255)'

                im = axes[1, j].imshow(ch_diff, cmap=cmap, norm=norm)
                axes[1, j].set_title(name, fontsize=15)
                axes[1, j].axis('off')

                # Barra inferior para cada canal
                cbar = fig.colorbar(im, ax=axes[1, j], orientation='horizontal', fraction=0.046, pad=0.08)
                cbar.set_label(label, fontsize=14)
                cbar.ax.tick_params(labelsize=14)


            plt.tight_layout(rect=[0, 0, 1, 0.95])
            save_path = os.path.join(output_viz_dir, f'divergence_{i+1}_{filename}.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
            print(f"✅ Saved divergence visualization for {filename} -> {save_path}")

        except FileNotFoundError as fnfe:
            print(f"❌ File not found: {fnfe}")
        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize per-channel divergence and total change in watermarked images.")
    parser.add_argument('--exp_name', type=str, default='1_3_255_w16_learn_im', help='Experiment name')
    parser.add_argument('--dataset', type=str, default='CFD', choices=['facelab_london', 'CFD', 'ONOT', 'ONOT_set1', 'LFW', 'SCface'])
    parser.add_argument('--train_dataset', type=str, default='celeba_hq', choices=['celeba_hq', 'coco'])
    parser.add_argument('--model_name', type=str, default='stegaformer', help='Model name')
    parser.add_argument('--num_images', type=int, default=5, help='Number of images to visualize')
    parser.add_argument('--normalize', action='store_true', help='Enable normalization per image and channel')
    parser.add_argument('--recognizer', type=str, default='facenet', help='Face recognizer used (not used in this script)')
    
    args = parser.parse_args()
    visualize_watermark_divergence(
        exp_name=args.exp_name,
        dataset=args.dataset,
        train_dataset=args.train_dataset,
        model_name=args.model_name,
        num_images_to_visualize=args.num_images,
        normalize=args.normalize,
        recognizer=args.recognizer
    )

