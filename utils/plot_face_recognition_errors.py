#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from typing import Optional

def read_thresholds_from_file(model_name, dataset_train, test_ds, bpp, recognizer, far=0.001) -> dict:
    path = r"..\evaluation\results\threholds_by_fars_{}_{}_cosine_online_{}.json".format(model_name, 
                                                                                            recognizer,
                                                                                            dataset_train)
    with open(path, "r") as f:
        data = json.load(f)

    return float(data[test_ds][f"({bpp}, 'OO')"][str(far)])   

def parse_bpp_from_experiment(exp_name: str) -> Optional[int]:
    parts = exp_name.split("_")
    
    if parts[0] == "1" and parts[1] == '1':
        return 1
    elif parts[0] == "1" and parts[1] == '3':
        return 3
    elif parts[0] == "3" and parts[1] == '3' and "stegformer" in exp_name.lower():
        return 3
    elif parts[0] == "3" and parts[1] == '3':
        return 6
    elif parts[0] == "15" and parts[1] == '2':
        return 8
    elif parts[0] == "2" and parts[1] == '8' and "stegformer" in exp_name.lower():
        return 8
    else:
        print(f"[WARN] Could not parse bpp from experiment name: {exp_name}")
        return None

def read_all_distances_files(path_genuine_pairs_labels, path_impostor_pairs_labels,
                             path_OO_genuine, path_OW_genuine, path_WW_genuine,
                             path_OO_impostor, path_OW_impostor, path_WW_impostor) -> pd.DataFrame:
    
    # Leer todos los archivos y concatenar en un solo DataFrame largo
    dfs = []
    pair_genuine_labels_df = pd.read_excel(path_genuine_pairs_labels)
    pair_impostor_labels_df = pd.read_excel(path_impostor_pairs_labels)

    for path, condition, pair_type in [
        (path_OO_genuine, 'OO', 'genuine'),
        (path_OW_genuine, 'OW', 'genuine'),
        (path_WW_genuine, 'WW', 'genuine'),
        (path_OO_impostor, 'OO', 'impostor'),
        (path_OW_impostor, 'OW', 'impostor'),
        (path_WW_impostor, 'WW', 'impostor'),
    ]:
        df = pd.read_csv(path)
        df['condition'] = condition
        df['pair_type'] = pair_type
        if pair_type == 'genuine':
            df = pd.concat([pair_genuine_labels_df, df], axis=1)
        else:
            df = pd.concat([pair_impostor_labels_df, df], axis=1)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def prepare_distance_dataframe(model_name, test_ds, exp_names_list, recognizer) -> pd.DataFrame:
    all_df_list = []
    for exp_name in exp_names_list:
        bpp = parse_bpp_from_experiment(exp_name)
        for train_ds in ['celeba_hq', 'coco']:
            # paths
            path_OO_genuine = r'..\experiments\output\recognition\{}\{}\{}\{}\{}\distances\cosine_genuine_distances_baseline_online.csv'.format(model_name, exp_name, train_ds, test_ds, recognizer)
            path_OW_genuine = r'..\experiments\output\recognition\{}\{}\{}\{}\{}\distances\cosine_genuine_distances_watermarked_online.csv'.format(model_name, exp_name, train_ds, test_ds, recognizer)
            path_WW_genuine = r'..\experiments\output\recognition\{}\{}\{}\{}\{}\distances\cosine_genuine_distances_watermarked_both_online.csv'.format(model_name, exp_name, train_ds, test_ds, recognizer)
            path_OO_impostor = r'..\experiments\output\recognition\{}\{}\{}\{}\{}\distances\cosine_impostor_distances_baseline_online.csv'.format(model_name, exp_name, train_ds, test_ds, recognizer)
            path_OW_impostor = r'..\experiments\output\recognition\{}\{}\{}\{}\{}\distances\cosine_impostor_distances_watermarked_online.csv'.format(model_name, exp_name, train_ds, test_ds, recognizer)
            path_WW_impostor = r'..\experiments\output\recognition\{}\{}\{}\{}\{}\distances\cosine_impostor_distances_watermarked_both_online.csv'.format(model_name, exp_name, train_ds, test_ds, recognizer)
            path_genuine_pairs_labels = r'..\experiments\output\recognition\{}\{}\{}\{}\{}\distances\{}_genuine_pairs.xlsx'.format(model_name, exp_name, train_ds, test_ds, recognizer, test_ds)
            path_impostor_pairs_labels = r'..\experiments\output\recognition\{}\{}\{}\{}\{}\distances\{}_impostor_pairs.xlsx'.format(model_name, exp_name, train_ds, test_ds, recognizer, test_ds)

            all_df = read_all_distances_files(path_genuine_pairs_labels, path_impostor_pairs_labels,
                                path_OO_genuine, path_OW_genuine, path_WW_genuine,
                                path_OO_impostor, path_OW_impostor, path_WW_impostor)
            all_df['bpp'] = bpp
            all_df['train_dataset'] = train_ds
            all_df['test_dataset'] = test_ds
            all_df_list.append(all_df)
    all_distances_df = pd.concat(all_df_list, ignore_index=True)

    return all_distances_df

def find_label_flips(df: pd.DataFrame, bpp: int, threshold: float, train_ds: str):
    """
    Given a fixed bpp and a distance threshold, identify the pairs (id_a, id_b)
    that change their class (genuine <-> impostor) when going from condition
    OO (baseline) to OW or WW.

    The class is recomputed with the rule:
        distance <= threshold -> 'genuine'
        distance >  threshold -> 'impostor'

    Returns
    -------
    summary : pd.DataFrame
        One row per (id_a, id_b) with the number of changes observed
        across OO->OW and OO->WW transitions, sorted by n_changes desc.
    flips_OO_OW : pd.DataFrame
        Detailed rows where the label changes between OO and OW.
    flips_OO_WW : pd.DataFrame
        Detailed rows where the label changes between OO and WW.
    """
    # 1) Fijar bpp y calcular etiqueta según el umbral
    df_bpp = df[(df["bpp"] == bpp) & (df['train_dataset']==train_ds)].copy()
    df_bpp["label_thr"] = np.where(
        df_bpp["distance"] <= threshold, "genuine", "impostor"
    )

    # 2) Tomar la condición base OO
    base = (
        df_bpp[df_bpp["condition"] == "OO"]
        [["id_a", "id_b", "distance", "label_thr"]]
        .rename(columns={
            "distance": "dist_OO",
            "label_thr": "label_OO",
        })
    )

    #print(base.loc[(base.id_a == 'BF016') & (base.id_b == 'BF038')])
    def _compute_flips(target_condition: str) -> pd.DataFrame:
        """Merge OO with otra condición (OW o WW) y detectar cambios."""
        other = (
            df_bpp[df_bpp["condition"] == target_condition]
            [["id_a", "id_b", "distance", "label_thr"]]
            .rename(columns={
                "distance": f"dist_{target_condition}",
                "label_thr": f"label_{target_condition}",
            })
        )

        merged = base.merge(other, on=["id_a", "id_b"], how="inner")
        # Filtrar solo donde hay cambio de etiqueta
        flips = merged[merged["label_OO"] != merged[f"label_{target_condition}"]].copy()
        #print(flips)
        if flips.empty:
            return flips

        # Tipo de cambio y diferencia de distancias (opcional para ordenar)
        flips["change_type"] = (
            flips["label_OO"] + "->" + flips[f"label_{target_condition}"]
        )
        flips["delta_distance"] = (
            flips[f"dist_{target_condition}"] - flips["dist_OO"]
        )

        # Ordenar, por ejemplo, por tipo de cambio y |delta_distance|
        flips = flips.sort_values(
            by=["change_type", "delta_distance"],
            ascending=[True, False],
        )
        return flips

    # 3) Cambios OO->OW y OO->WW
    flips_OO_OW = _compute_flips("OW")
    flips_OO_WW = _compute_flips("WW")

    # 4) Resumen: ¿qué parejas tuvieron más cambios?
    all_flips_list = []
    for cond, df_flips in [("OO->OW", flips_OO_OW), ("OO->WW", flips_OO_WW)]:
        if not df_flips.empty:
            tmp = df_flips[["id_a", "id_b", "change_type"]].copy()
            tmp["transition"] = cond
            all_flips_list.append(tmp)

    if all_flips_list:
        all_flips = pd.concat(all_flips_list, ignore_index=True)
        summary = (
            all_flips.groupby(["id_a", "id_b"])
            .size()
            .reset_index(name="n_changes")
            .sort_values("n_changes", ascending=False)
        )
    else:
        summary = pd.DataFrame(columns=["id_a", "id_b", "n_changes"])

    return summary, flips_OO_OW, flips_OO_WW

def compute_bpp(model_name: str, exp_name: str) -> int:
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

def process_condition_flips(model_name, df_flips, exp_name, train_dataset, dataset, bpp, recognizer, condition):

    if df_flips.empty:
        print(f"No hay cambios de etiqueta para la condición {condition} con esos parámetros.")
        return

    # Seleccionar los 2 casos con mayor |delta_distance|
    df_flips = df_flips.copy()
    df_flips["abs_delta"] = df_flips["delta_distance"].abs()

    gi = df_flips[df_flips["change_type"] == "genuine->impostor"].sort_values("abs_delta", ascending=False)
    ig = df_flips[df_flips["change_type"] == "impostor->genuine"].sort_values("abs_delta", ascending=False)

    selected_rows = []

    if not gi.empty:
        selected_rows.append(gi.iloc[0])
    if not ig.empty:
        selected_rows.append(ig.iloc[0])

    if len(selected_rows) < 2:
        remaining = df_flips.sort_values("abs_delta", ascending=False)
        for _, row in remaining.iterrows():
            already = any(
                (row["id_a"] == r["id_a"]) and
                (row["id_b"] == r["id_b"]) and
                (row["change_type"] == r["change_type"])
                for r in selected_rows
            )
            if not already:
                selected_rows.append(row)
            if len(selected_rows) == 2:
                break

    # Rutas 
    base_output_dir = r'..\experiments\output\watermarking\{}\{}\inference\{}\{}'.format(model_name, 
                                                                                        exp_name, train_dataset, dataset)
    original_images_dir = r'..\datasets\{}\processed\test'.format(dataset)
    original_templates_dir = r'..\datasets\{}\processed\templates'.format(dataset)
    watermarked_images_dir = os.path.join(base_output_dir, 'watermarked_images')
    watermarked_templates_dir = os.path.join(base_output_dir, 'watermarked_templates')

    output_viz_dir = r'..\evaluation\visualizations\{}\differences\{}\{}\{}\{}\flips_{}'.format(model_name, 
                                                                                            train_dataset, dataset, recognizer, bpp, condition)
    os.makedirs(output_viz_dir, exist_ok=True)
    for idx, row in enumerate(selected_rows, start=1):
        id_a = str(row["id_a"])
        id_b = str(row["id_b"])
        if dataset == 'SCface':
            id_a = str(id_a).zfill(3)
            id_b = str(id_b).zfill(3)

        if dataset in ["ONOT", "ONOT_set1"]:
            tmpl_path = os.path.join(original_templates_dir, f"{id_a}.png")
            img_filename = [img_name for img_name in os.listdir(original_images_dir) if img_name.startswith(id_b)][0]
            probe_orig_path = os.path.join(original_images_dir, f"{img_filename}")
            tmpl_wm_path = os.path.join(watermarked_templates_dir, f"{id_a}.png")
            probe_wm_path = os.path.join(watermarked_images_dir, f"{img_filename.replace('.jpg', '.png')}")
        else:
            if dataset == 'SCface':
                tmpl_path = os.path.join(original_templates_dir, f"{id_a}.JPG")
            else:
                tmpl_path = os.path.join(original_templates_dir, f"{id_a}.jpg")
                
            img_filename = [img_name for img_name in os.listdir(original_images_dir) if img_name.startswith(id_b)][0]
            probe_orig_path = os.path.join(original_images_dir, f"{img_filename.replace('.png', '.jpg')}")
            tmpl_wm_path = os.path.join(watermarked_templates_dir, f"{id_a}.png")
            probe_wm_path = os.path.join(watermarked_images_dir, f"{img_filename.replace('.jpg', '.png')}")

        needed = [tmpl_path, probe_orig_path, probe_wm_path]
        if condition == "WW":
            needed.append(tmpl_wm_path)

        if any(not os.path.exists(p) for p in needed):
            print(f"Saltando par ({id_a}, {id_b}) por falta de archivos de imagen.")
            continue

        # Cargar imágenes
        tmpl_img = Image.open(tmpl_path).convert("RGB")
        probe_orig = Image.open(probe_orig_path).convert("RGB")
        probe_wm = Image.open(probe_wm_path).convert("RGB")
        tmpl_img = ImageOps.fit(tmpl_img, (256, 256))
        probe_orig = ImageOps.fit(probe_orig, (256, 256))
        #probe_wm = ImageOps.fit(probe_wm, (256, 256))

        if condition == "WW":
            tmpl_wm = Image.open(tmpl_wm_path).convert("RGB")
            #tmpl_wm = ImageOps.fit(tmpl_wm, (256, 256))

        # Diferencia probe
        orig_np = np.array(probe_orig).astype(float)
        wm_np = np.array(probe_wm).astype(float)
        abs_diff_p = np.abs(wm_np - orig_np)
        mag_p = abs_diff_p.mean(axis=-1) 
        max_mag_p = mag_p.max() if mag_p.max() > 0 else 1.0
        diff_display_p = 1.0 - (mag_p / max_mag_p)

        if condition == "WW":
            # Diferencia template
            tmpl_np = np.array(tmpl_img).astype(float)
            tmpl_wm_np = np.array(tmpl_wm).astype(float)
            abs_diff_t = np.abs(tmpl_wm_np - tmpl_np)
            mag_t = abs_diff_t.mean(axis=-1) 
            max_mag_t = mag_t.max() if mag_t.max() > 0 else 1.0
            diff_display_t = 1.0 - (mag_t / max_mag_t)

        # Figuras
        if condition == "OW":
            fig, axes = plt.subplots(1, 4, figsize=(12, 4))
            for ax in axes:
                ax.axis("off")

            axes[0].imshow(tmpl_img)
            axes[0].set_title("Original template")

            axes[1].imshow(probe_orig)
            axes[1].set_title("Original probe")

            axes[2].imshow(probe_wm)
            axes[2].set_title("Watermarked probe")

            axes[3].imshow(diff_display_p, cmap="gray", vmin=0.0, vmax=1.0)
            axes[3].set_title("Difference")

        else:  # WW → 6 plots
            fig, axes = plt.subplots(1, 6, figsize=(18, 4))
            for ax in axes:
                ax.axis("off")

            axes[0].imshow(tmpl_img)
            axes[0].set_title("Original template")

            axes[1].imshow(probe_orig)
            axes[1].set_title("Original probe")

            axes[2].imshow(tmpl_wm)
            axes[2].set_title("Watermarked template")

            axes[3].imshow(probe_wm)
            axes[3].set_title("Watermarked probe")

            axes[4].imshow(diff_display_t, cmap="gray", vmin=0.0, vmax=1.0)
            axes[4].set_title("Template difference")

            axes[5].imshow(diff_display_p, cmap="gray", vmin=0.0, vmax=1.0)
            axes[5].set_title("Probe difference")

        plt.subplots_adjust(wspace=0, hspace=0)

        # Texto inferior
        change_desc = row["change_type"].replace("genuine", "Genuine").replace("impostor", "Impostor")
        dist_OO = row["dist_OO"]
        if condition == "OW":
            dist_other = row["dist_OW"]
            label_other = "Original vs Watermarked"
        else:
            dist_other = row["dist_WW"]
            label_other = "Watermarked vs Watermarked"

        fig.supxlabel(
            f"{change_desc} - Original vs Original: {dist_OO:.4f} - {label_other}: {dist_other:.4f}",
            fontsize=14,
            y=0.15
        )
        
        out_name = f"flip_{condition}_{idx}_{id_a}_{id_b}.png"
        save_path = os.path.join(output_viz_dir, out_name)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"✅ Guardado: {save_path}")

def visualize_flips_condition(
    df: pd.DataFrame,
    exp_name: str,
    dataset: str,
    train_dataset: str,
    model_name: str,
    threshold: float,
    recognizer: str,
) -> None:
    bpp = compute_bpp(model_name, exp_name)

    summary, flips_OW, flips_WW = find_label_flips(
        df, bpp=bpp, threshold=threshold, train_ds=train_dataset
    )

    process_condition_flips(model_name, flips_OW, exp_name, train_dataset, dataset, bpp, recognizer, condition="OW")
    process_condition_flips(model_name, flips_WW, exp_name, train_dataset, dataset, bpp, recognizer, condition="WW")

# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualizar ejemplos de cambios de etiqueta (OO->OW o OO->WW) con template/probe y mapas de diferencia."
    )
    parser.add_argument("--exp_names_list", type=str, nargs='+', default=['1_1_255_w16_learn_im','1_3_255_w16_learn_im','3_3_255_w16_learn_im','15_2_255_w16_learn_im'],
                        help="Lista de nombres de experimentos para extraer bpp")
    parser.add_argument("--dataset", type=str, default="facelab_london",
                        choices=["facelab_london", "CFD", "ONOT", "ONOT_set1", "LFW", "SCface"],
                        help="Dataset de prueba (test_dataset)")
    parser.add_argument("--train_dataset", type=str, default="celeba_hq",
                        choices=["celeba_hq", "coco"],
                        help="Dataset de entrenamiento")
    parser.add_argument("--model_name", type=str, default="stegaformer",
                        help="Nombre del modelo de watermarking")
    parser.add_argument("--recognizer", type=str, default="facenet",
                        help="Reconocedor facial utilizado")
    parser.add_argument("--far", type=float, default=0.001,
                        help="Valor de FAR deseado")

    args = parser.parse_args()

    all_distances_df = prepare_distance_dataframe(model_name=args.model_name, test_ds=args.dataset, 
                                              exp_names_list=args.exp_names_list, 
                                              recognizer=args.recognizer)

    for exp_name in args.exp_names_list:
        bpp = compute_bpp(args.model_name, exp_name)
        threshold_value = read_thresholds_from_file(args.model_name, args.train_dataset, args.dataset, 
                                                    bpp, args.recognizer, args.far)
        visualize_flips_condition(
            df=all_distances_df,
            exp_name=exp_name,
            dataset=args.dataset,
            train_dataset=args.train_dataset,
            model_name=args.model_name,
            threshold=threshold_value,
            recognizer=args.recognizer
        )
