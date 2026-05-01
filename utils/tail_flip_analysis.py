import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


def compute_bpp(model_name: str, exp_name: str) -> int:
    """Determine bpp based on experiment name and model."""
    splits = exp_name.split('_')
    if model_name.lower() == 'stegaformer':
        mapping = {'1_1': 1, '1_3': 3, '3_3': 6, '15_2': 8}
    elif model_name.lower() == 'stegformer':
        mapping = {'1_1': 1, '3_3': 3, '3_6': 6, '2_8': 8}
    else:
        raise ValueError(f'Unknown model name: {model_name}')

    key = f'{splits[0]}_{splits[1]}'
    if key in mapping:
        return mapping[key]
    raise ValueError(f'Unknown experiment name format: {exp_name}')


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

    base_dir = f'../experiments/output/watermarking/{model_name}'
    if os.path.exists(base_dir):
        for d in os.listdir(base_dir):
            if d.startswith(prefix):
                return d
    return f'{prefix}_255_w16_learn_im'


def compute_tar_fixed_threshold_no_labels(model_name, train_dataset, test_dataset,
                                           recognizer, bpp_values, mode='online',
                                           far_target=0.001, max_thresholds=20000):
    """
    Compute TAR under fixed OO threshold without impostor pair labels.
    Used for LFW where impostor labels are unavailable.
    Returns a DataFrame with TAR_OO, TAR_OW, TAR_WW and deltas.
    """
    results = []
    exp_name = get_exp_name(model_name, bpp_values[0])

    for bpp in bpp_values:
        exp_name = get_exp_name(model_name, bpp)
        base_path = (f'../experiments/output/recognition/{model_name}/{exp_name}/'
                     f'{train_dataset}/{test_dataset}/{recognizer}/distances')
        mode_suffix = f'_online' if mode == 'online' else ''

        try:
            # Load distances
            genuine_oo = pd.read_csv(
                f'{base_path}/cosine_genuine_distances_baseline{mode_suffix}_mtcnn.csv')['distance'].values
            genuine_ow = pd.read_csv(
                f'{base_path}/cosine_genuine_distances_watermarked{mode_suffix}_mtcnn.csv')['distance'].values
            genuine_ww = pd.read_csv(
                f'{base_path}/cosine_genuine_distances_watermarked_both{mode_suffix}_mtcnn.csv')['distance'].values
            impostor_oo = pd.read_csv(
                f'{base_path}/cosine_impostor_distances_baseline{mode_suffix}_mtcnn.csv')['distance'].values

            # Compute threshold from OO
            threshold = compute_threshold_at_far(genuine_oo, impostor_oo,
                                                  far_target, max_thresholds)

            # Count genuines accepted under fixed threshold
            accepted_oo = int(np.sum(genuine_oo <= threshold))
            accepted_ow = int(np.sum(genuine_ow <= threshold))
            accepted_ww = int(np.sum(genuine_ww <= threshold))
            total_genuine = len(genuine_oo)

            tar_oo = round(accepted_oo / total_genuine * 100, 1)
            tar_ow = round(accepted_ow / total_genuine * 100, 1)
            tar_ww = round(accepted_ww / total_genuine * 100, 1)

            results.append({
                'dataset': test_dataset,
                'train': train_dataset,
                'FR': recognizer,
                'bpp': bpp,
                'threshold': round(threshold, 6),
                'TAR_OO': tar_oo,
                'TAR_OW': tar_ow,
                'TAR_WW': tar_ww,
                'delta_OW': round(tar_ow - tar_oo, 1),
                'delta_WW': round(tar_ww - tar_oo, 1),
                'total_genuine': total_genuine,
                'correctly_accepted_OO': accepted_oo,
                'genuine_to_impostor_OW': accepted_oo - accepted_ow,
                'genuine_to_impostor_WW': accepted_oo - accepted_ww,
            })

        except Exception as e:
            print(f'Error: {test_dataset}/{train_dataset}/{recognizer}/{bpp}bpp: {e}')

    return pd.DataFrame(results)


def compute_threshold_at_far(genuine_distances, impostor_distances, 
                              far_target=0.001, max_thresholds=None):
    """
    Compute threshold at a given FAR replicating calculate_metrics logic.
    For large datasets, limits the number of thresholds evaluated.
    """
    distances = np.concatenate([genuine_distances, impostor_distances])
    
    if max_thresholds is not None:
        # Sample evenly across the range instead of using all distances
        min_dist = distances.min()
        max_dist = distances.max()
        thresholds_to_check = np.linspace(min_dist, max_dist, max_thresholds)
    else:
        thresholds_to_check = np.sort(distances)
    
    best_threshold = None
    best_far = -1
    
    for threshold in thresholds_to_check:
        fp = np.sum(impostor_distances <= threshold)
        tn = np.sum(impostor_distances > threshold)
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        if far <= far_target and far > best_far:
            best_far = far
            best_threshold = threshold
    
    return best_threshold


def read_all_distances(model_name, exp_name, train_dataset, test_dataset,
                       recognizer, mode='online'):
    """
    Read genuine and impostor distance files for all conditions (OO, OW, WW)
    and return a single concatenated DataFrame.
    """
    base_path = (f'../experiments/output/recognition/{model_name}/{exp_name}/'
                 f'{train_dataset}/{test_dataset}/{recognizer}/distances')

    mode_suffix = f'_{mode}' if mode == 'online' else ''

    file_map = {
        ('OO', 'genuine'):  f'cosine_genuine_distances_baseline{mode_suffix}_mtcnn.csv',
        ('OW', 'genuine'):  f'cosine_genuine_distances_watermarked{mode_suffix}_mtcnn.csv',
        ('WW', 'genuine'):  f'cosine_genuine_distances_watermarked_both{mode_suffix}_mtcnn.csv',
        ('OO', 'impostor'): f'cosine_impostor_distances_baseline{mode_suffix}_mtcnn.csv',
        ('OW', 'impostor'): f'cosine_impostor_distances_watermarked{mode_suffix}_mtcnn.csv',
        ('WW', 'impostor'): f'cosine_impostor_distances_watermarked_both{mode_suffix}_mtcnn.csv',
    }

    pair_labels = {
        'genuine':  f'{test_dataset}_genuine_pairs.xlsx',
        'impostor': f'{test_dataset}_impostor_pairs.xlsx',
    }

    dfs = []
    for (condition, pair_type), filename in file_map.items():
        filepath = os.path.join(base_path, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping.")
            continue

        df = pd.read_csv(filepath)
        df['condition'] = condition
        df['pair_type'] = pair_type

        # Attach pair labels
        label_path = os.path.join(base_path, pair_labels[pair_type])
        if os.path.exists(label_path):
            labels_df = pd.read_excel(label_path)
            df = pd.concat([labels_df, df], axis=1)

        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No distance files found in {base_path}")

    return pd.concat(dfs, ignore_index=True)


def find_label_flips(df, far_target=0.001):
    """
    Identify pairs that change their verification decision (genuine <-> impostor)
    when going from OO to OW or WW.

    The threshold is computed automatically from the OO impostor distribution
    at the specified FAR target, ensuring consistency with reported TAR@FAR metrics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: id_a, id_b, distance, condition, pair_type.
    far_target : float
        FAR operating point for threshold computation (default 0.001 = 0.1%).

    Returns
    -------
    threshold : float
        The computed threshold at FAR=far_target.
    flips_OW : pd.DataFrame
        Pairs that flip between OO and OW.
    flips_WW : pd.DataFrame
        Pairs that flip between OO and WW.
    summary : pd.DataFrame
        Summary of flips per pair.
    """
    oo_genuine = df[
        (df['condition'] == 'OO') & (df['pair_type'] == 'genuine')
    ]['distance'].values

    oo_impostors = df[
        (df['condition'] == 'OO') & (df['pair_type'] == 'impostor')
    ]['distance'].values

    threshold = compute_threshold_at_far(oo_genuine, oo_impostors, far_target)

    # Assign labels based on threshold
    df = df.copy()
    df['label_thr'] = np.where(df['distance'] <= threshold, 'genuine', 'impostor')

    # Get OO baseline
    base = (
        df[df['condition'] == 'OO']
        [['id_a', 'id_b', 'pair_type', 'distance', 'label_thr']]
        .rename(columns={'distance': 'dist_OO', 'label_thr': 'label_OO'})
    )

    def _compute_flips(target_condition):
        other = (
            df[df['condition'] == target_condition]
            [['id_a', 'id_b', 'distance', 'label_thr']]
            .rename(columns={
                'distance': f'dist_{target_condition}',
                'label_thr': f'label_{target_condition}',
            })
        )
        merged = base.merge(other, on=['id_a', 'id_b'], how='inner')
        merged = merged[merged['label_OO'] == merged['pair_type']]
        flips = merged[merged['label_OO'] != merged[f'label_{target_condition}']].copy()

        if flips.empty:
            return flips

        flips['change_type'] = flips['label_OO'] + '->' + flips[f'label_{target_condition}']
        flips['delta_distance'] = flips[f'dist_{target_condition}'] - flips['dist_OO']
        flips['threshold'] = threshold
        flips = flips.sort_values(
            by=['change_type', 'delta_distance'],
            ascending=[True, False],
        )
        return flips

    flips_OW = _compute_flips('OW')
    flips_WW = _compute_flips('WW')

    # Summary
    all_flips_list = []
    for cond, df_flips in [('OO->OW', flips_OW), ('OO->WW', flips_WW)]:
        if not df_flips.empty:
            tmp = df_flips[['id_a', 'id_b', 'change_type']].copy()
            tmp['transition'] = cond
            all_flips_list.append(tmp)

    if all_flips_list:
        all_flips = pd.concat(all_flips_list, ignore_index=True)
        summary = (
            all_flips.groupby(['id_a', 'id_b'])
            .size()
            .reset_index(name='n_changes')
            .sort_values('n_changes', ascending=False)
        )
    else:
        summary = pd.DataFrame(columns=['id_a', 'id_b', 'n_changes'])

    return threshold, flips_OW, flips_WW, summary


def compute_flip_summary_table(model_name, datasets, train_datasets, recognizers,
                               bpp_values, mode='online', far_target=0.001):
    """
    Compute a summary table of tail-flips across all configurations.
    Returns a DataFrame with columns:
    dataset, train, FR, bpp, threshold, n_genuine_to_impostor_OW/WW,
    n_impostor_to_genuine_OW/WW, total_pairs.

    This provides the numbers needed for the R2 paragraph on tail-flips.
    """
    results = []

    for test_ds in datasets:
        for recognizer in recognizers:
            for train_ds in train_datasets:
                for bpp in bpp_values:
                    try:
                        exp_name = get_exp_name(model_name, bpp)
                        df = read_all_distances(
                            model_name, exp_name, train_ds, test_ds,
                            recognizer, mode
                        )

                        threshold, flips_OW, flips_WW, summary = find_label_flips(
                            df, far_target
                        )
                        # Add label_thr to df using the computed threshold
                        df['label_thr'] = np.where(
                            df['distance'] <= threshold, 'genuine', 'impostor'
                        )

                        # Count flips by type
                        def count_by_type(flips_df):
                            if flips_df.empty:
                                return 0, 0
                            gi = len(flips_df[flips_df['change_type'] == 'genuine->impostor'])
                            ig = len(flips_df[flips_df['change_type'] == 'impostor->genuine'])
                            return gi, ig

                        gi_ow, ig_ow = count_by_type(flips_OW)
                        gi_ww, ig_ww = count_by_type(flips_WW)

                        n_genuine = len(df[(df['condition'] == 'OO') & (df['pair_type'] == 'genuine')])
                        n_impostor = len(df[(df['condition'] == 'OO') & (df['pair_type'] == 'impostor')])
                        
                        correctly_accepted_OO = len(df[
                            (df['condition'] == 'OO') & 
                            (df['pair_type'] == 'genuine') & 
                            (df['label_thr'] == 'genuine')
                        ])
                        correctly_rejected_OO = len(df[
                            (df['condition'] == 'OO') & 
                            (df['pair_type'] == 'impostor') & 
                            (df['label_thr'] == 'impostor')
                        ])

                        results.append({
                            'dataset': test_ds,
                            'train': train_ds,
                            'FR': recognizer,
                            'bpp': bpp,
                            'threshold': round(threshold, 6),
                            'genuine_to_impostor_OW': gi_ow,
                            'genuine_to_impostor_WW': gi_ww,
                            'impostor_to_genuine_OW': ig_ow,
                            'impostor_to_genuine_WW': ig_ww,
                            'total_genuine': n_genuine,
                            'total_impostor': n_impostor,
                            'correctly_accepted_OO': correctly_accepted_OO,
                            'correctly_rejected_OO': correctly_rejected_OO,
                        })

                    except Exception as e:
                        print(f"Error: {test_ds}/{train_ds}/{recognizer}/{bpp}bpp: {e}")

    return pd.DataFrame(results)


def visualize_flips(model_name, exp_name, test_dataset, train_dataset,
                    recognizer, condition, mode='online', far_target=0.001,
                    num_examples=2, save_path=None):
    """
    Visualize examples of pairs that flipped from genuine to impostor
    (or vice versa) after watermarking.

    Parameters
    ----------
    model_name : str
    exp_name : str
    test_dataset : str
    train_dataset : str
    recognizer : str
    condition : str
        'OW' or 'WW'.
    mode : str
        'online' or 'offline'.
    far_target : float
        FAR for threshold computation.
    num_examples : int
        Number of flip examples to visualize.
    save_path : str or None
        Output directory. If None, auto-generated.
    """
    assert condition in ['OW', 'WW'], "condition must be 'OW' or 'WW'"

    bpp = compute_bpp(model_name, exp_name)

    # Read distances
    df = read_all_distances(model_name, exp_name, train_dataset,
                            test_dataset, recognizer, mode)

    # Find flips
    threshold, flips_OW, flips_WW, summary = find_label_flips(df, far_target)
    df_flips = flips_OW if condition == 'OW' else flips_WW

    if df_flips.empty:
        print(f"No flips found for {condition} with these parameters.")
        return

    # Select examples with largest |delta_distance|
    df_flips = df_flips.copy()
    df_flips['abs_delta'] = df_flips['delta_distance'].abs()

    gi = df_flips[df_flips['change_type'] == 'genuine->impostor'].sort_values('abs_delta', ascending=False)
    ig = df_flips[df_flips['change_type'] == 'impostor->genuine'].sort_values('abs_delta', ascending=False)

    selected_rows = []
    if not gi.empty:
        selected_rows.append(gi.iloc[0])
    if not ig.empty:
        selected_rows.append(ig.iloc[0])

    # Fill remaining if needed
    if len(selected_rows) < num_examples:
        remaining = df_flips.sort_values('abs_delta', ascending=False)
        for _, row in remaining.iterrows():
            already = any(
                (row['id_a'] == r['id_a']) and (row['id_b'] == r['id_b'])
                for r in selected_rows
            )
            if not already:
                selected_rows.append(row)
            if len(selected_rows) >= num_examples:
                break

    # Paths
    base_output_dir = (f'../experiments/output/watermarking/{model_name}/'
                       f'{exp_name}/inference/{train_dataset}/{test_dataset}')
    original_images_dir = f'../datasets/{test_dataset}/processed/test'
    original_templates_dir = f'../datasets/{test_dataset}/processed/templates'
    watermarked_images_dir = os.path.join(base_output_dir, 'watermarked_images')
    watermarked_templates_dir = os.path.join(base_output_dir, 'watermarked_templates')

    if save_path is None:
        save_path = (f'../evaluation/visualizations/{model_name}/flips/'
                     f'{train_dataset}/{test_dataset}/{recognizer}/{bpp}')
    os.makedirs(save_path, exist_ok=True)

    is_png_dataset = test_dataset in ['ONOT', 'ONOT_set1']

    for idx, row in enumerate(selected_rows, start=1):
        id_a = str(row['id_a'])
        id_b = str(row['id_b'])

        try:
            # Find filenames
            if is_png_dataset:
                tmpl_path = os.path.join(original_templates_dir, f'{id_a}.png')
                img_files = [f for f in os.listdir(original_images_dir) if f.startswith(id_b)]
                if not img_files:
                    continue
                img_filename = img_files[0]
                probe_orig_path = os.path.join(original_images_dir, img_filename)
                tmpl_wm_path = os.path.join(watermarked_templates_dir, f'{id_a}.png')
                probe_wm_path = os.path.join(watermarked_images_dir, img_filename)
            else:
                tmpl_path = os.path.join(original_templates_dir, f'{id_a}.jpg')
                img_files = [f for f in os.listdir(original_images_dir) if f.startswith(id_b)]
                if not img_files:
                    continue
                img_filename = img_files[0]
                ext = img_filename.split('.')[-1]
                probe_orig_path = os.path.join(original_images_dir, img_filename)
                tmpl_wm_path = os.path.join(watermarked_templates_dir, f'{id_a}.png')
                probe_wm_path = os.path.join(watermarked_images_dir, img_filename.replace(ext, 'png'))

            # Check files exist
            needed = [tmpl_path, probe_orig_path, probe_wm_path]
            if condition == 'WW':
                needed.append(tmpl_wm_path)
            if any(not os.path.exists(p) for p in needed):
                print(f"Skipping pair ({id_a}, {id_b}): missing files.")
                continue

            # Load images
            tmpl_img = ImageOps.fit(Image.open(tmpl_path).convert('RGB'), (256, 256))
            probe_orig = ImageOps.fit(Image.open(probe_orig_path).convert('RGB'), (256, 256))
            probe_wm = Image.open(probe_wm_path).convert('RGB')

            # Compute probe difference
            orig_np = np.array(probe_orig).astype(float)
            wm_np = np.array(probe_wm).astype(float)
            mag_p = np.abs(wm_np - orig_np).mean(axis=-1)
            max_mag_p = mag_p.max() if mag_p.max() > 0 else 1.0
            diff_display_p = 1.0 - (mag_p / max_mag_p)

            if condition == 'OW':
                fig, axes = plt.subplots(1, 4, figsize=(14, 4))
                for ax in axes:
                    ax.axis('off')
                axes[0].imshow(tmpl_img)
                axes[0].set_title('Original Template', fontsize=10)
                axes[1].imshow(probe_orig)
                axes[1].set_title('Original Probe', fontsize=10)
                axes[2].imshow(probe_wm)
                axes[2].set_title('Watermarked Probe', fontsize=10)
                axes[3].imshow(diff_display_p, cmap='gray', vmin=0.0, vmax=1.0)
                axes[3].set_title('Probe Difference', fontsize=10)
            else:
                tmpl_wm = Image.open(tmpl_wm_path).convert('RGB')
                tmpl_np = np.array(tmpl_img).astype(float)
                tmpl_wm_np = np.array(tmpl_wm).astype(float)
                mag_t = np.abs(tmpl_wm_np - tmpl_np).mean(axis=-1)
                max_mag_t = mag_t.max() if mag_t.max() > 0 else 1.0
                diff_display_t = 1.0 - (mag_t / max_mag_t)

                fig, axes = plt.subplots(1, 6, figsize=(20, 4))
                for ax in axes:
                    ax.axis('off')
                axes[0].imshow(tmpl_img)
                axes[0].set_title('Original Template', fontsize=10)
                axes[1].imshow(probe_orig)
                axes[1].set_title('Original Probe', fontsize=10)
                axes[2].imshow(tmpl_wm)
                axes[2].set_title('WM Template', fontsize=10)
                axes[3].imshow(probe_wm)
                axes[3].set_title('WM Probe', fontsize=10)
                axes[4].imshow(diff_display_t, cmap='gray', vmin=0.0, vmax=1.0)
                axes[4].set_title('Template Diff', fontsize=10)
                axes[5].imshow(diff_display_p, cmap='gray', vmin=0.0, vmax=1.0)
                axes[5].set_title('Probe Diff', fontsize=10)

            plt.subplots_adjust(wspace=0.05, hspace=0)

            # Info text
            change_desc = row['change_type'].replace('genuine', 'Genuine').replace('impostor', 'Impostor')
            dist_OO = row['dist_OO']
            dist_cond = row[f'dist_{condition}']

            fig.supxlabel(
                f'{change_desc} | Dist OO: {dist_OO:.4f} | Dist {condition}: {dist_cond:.4f} | '
                f'Threshold: {threshold:.4f} | Pair type: {row["pair_type"]}',
                fontsize=11, y=0.12,
            )

            fig.suptitle(
                f'{model_name.capitalize()} {bpp}bpp | {train_dataset} | '
                f'{test_dataset} | {recognizer} | {condition}',
                fontsize=12, fontweight='bold',
            )

            out_name = f'flip_{condition}_{idx}_{id_a}_{id_b}.png'
            fig.savefig(os.path.join(save_path, out_name), bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f"Saved: {os.path.join(save_path, out_name)}")

        except Exception as e:
            print(f"Error processing pair ({id_a}, {id_b}): {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Tail-flip analysis: compute and visualize verification decision changes.'
    )
    parser.add_argument('--action', type=str, required=True,
                        choices=['summary', 'visualize'],
                        help="'summary' to compute flip counts, 'visualize' to generate examples")
    parser.add_argument('--model_name', type=str, default='stegformer')
    parser.add_argument('--datasets', nargs='+',
                        default=['ONOT_set1', 'LFW', 'SCface', 'CFD', 'facelab_london'])
    parser.add_argument('--train_datasets', nargs='+',
                        default=['coco', 'celeba_hq'])
    parser.add_argument('--recognizers', nargs='+',
                        default=['arcface', 'facenet'])
    parser.add_argument('--bpp_values', nargs='+', type=int,
                        default=[1, 3, 6, 8])
    parser.add_argument('--mode', type=str, default='online',
                        choices=['online', 'offline'])
    parser.add_argument('--far_target', type=float, default=0.001,
                        help='FAR operating point for threshold (default 0.001 = 0.1%%)')
    parser.add_argument('--condition', type=str, default='WW',
                        choices=['OW', 'WW'],
                        help='Condition for visualization')
    parser.add_argument('--num_examples', type=int, default=2,
                        help='Number of flip examples to visualize')
    parser.add_argument('--max_thresholds', type=int, default=None,
                    help='Max thresholds to evaluate. Use 20000 for LFW.')
    parser.add_argument('--save_path', type=str, default=None)

    args = parser.parse_args()

    if args.action == 'summary':
        # Standard datasets with impostor labels
        datasets_standard = [d for d in args.datasets if d != 'LFW']
        datasets_lfw = [d for d in args.datasets if d == 'LFW']

        dfs = []
        if datasets_standard:
            df_standard = compute_flip_summary_table(
                model_name=args.model_name,
                datasets=datasets_standard,
                train_datasets=args.train_datasets,
                recognizers=args.recognizers,
                bpp_values=args.bpp_values,
                mode=args.mode,
                far_target=args.far_target,
            )
            dfs.append(df_standard)

        if datasets_lfw:
            for train_ds in args.train_datasets:
                for rec in args.recognizers:
                    df_lfw = compute_tar_fixed_threshold_no_labels(
                        model_name=args.model_name,
                        train_dataset=train_ds,
                        test_dataset='LFW',
                        recognizer=rec,
                        bpp_values=args.bpp_values,
                        mode=args.mode,
                        far_target=args.far_target,
                        max_thresholds=args.max_thresholds,
                    )
                    dfs.append(df_lfw)

        df_summary = pd.concat(dfs, ignore_index=True)
        print(df_summary.to_string(index=False))
        out_path = args.save_path or f'../evaluation/visualizations/{args.model_name}/flips/flip_summary.csv'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_summary.to_csv(out_path, index=False)
        print(f'\nSummary saved to {out_path}')

    elif args.action == 'visualize':
        for ds in args.datasets:
            for train_ds in args.train_datasets:
                for rec in args.recognizers:
                    for bpp in args.bpp_values:
                        try:
                            exp_name = get_exp_name(args.model_name, bpp)
                            visualize_flips(
                                model_name=args.model_name,
                                exp_name=exp_name,
                                test_dataset=ds,
                                train_dataset=train_ds,
                                recognizer=rec,
                                condition=args.condition,
                                mode=args.mode,
                                far_target=args.far_target,
                                num_examples=args.num_examples,
                                save_path=args.save_path,
                            )
                        except Exception as e:
                            print(f"Error: {ds}/{train_ds}/{rec}/{bpp}bpp: {e}")

# Example usage:
#
# Compute summary table:
# python tail_flip_analysis.py --action summary --model_name stegformer \
#     --datasets ONOT_set1 LFW SCface --mode online
#
# Visualize flips:
# python tail_flip_analysis.py --action visualize --model_name stegformer \
#     --datasets ONOT_set1 --train_datasets coco --recognizers facenet \
#     --bpp_values 6 --condition WW --mode online