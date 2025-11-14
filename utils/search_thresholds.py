import os
import re
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from joblib import Parallel, delayed
import numpy as np
import argparse


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
    else:
        print(f"[WARN] Could not parse bpp from experiment name: {exp_name}")
        return None

def find_distance_files(dist_dir: Path, metric: str, mode: str) -> Dict[Tuple[str, str], Path]:
    """
    Return mapping {(condition, pair_type) -> file_path} if exists.
    Supports offline & online suffixes.
    Condition: OO, OW, WW
    Pair type: genuine, impostor
    """
    if mode == 'online':
        patterns = {
            ("OO", "genuine"):   [f"{metric}_genuine_distances_baseline_online.csv"],
            ("OO", "impostor"):  [f"{metric}_impostor_distances_baseline_online.csv"],
            ("OW", "genuine"):   [f"{metric}_genuine_distances_watermarked_online.csv"],
            ("OW", "impostor"):  [f"{metric}_impostor_distances_watermarked_online.csv"],
            ("WW", "genuine"):   [f"{metric}_genuine_distances_watermarked_both_online.csv"],
            ("WW", "impostor"):  [f"{metric}_impostor_distances_watermarked_both_online.csv"],
        }
    else:
        patterns = {
            ("OO", "genuine"):   [f"{metric}_genuine_distances_baseline.csv"],
            ("OO", "impostor"):  [f"{metric}_impostor_distances_baseline.csv"],
            ("OW", "genuine"):   [f"{metric}_genuine_distances_watermarked.csv"],
            ("OW", "impostor"):  [f"{metric}_impostor_distances_watermarked.csv"],
            ("WW", "genuine"):   [f"{metric}_genuine_distances_watermarked_both.csv"],
            ("WW", "impostor"):  [f"{metric}_impostor_distances_watermarked_both.csv"],
        }
    out = {}
    for key, candidates in patterns.items():
        for name in candidates:
            p = dist_dir / name
            if p.is_file():
                out[key] = p
                break
    return out

def read_distance_column(csv_path: Path) -> np.ndarray:
    """
    Reads a CSV that may have header or multiple columns and returns a 1D float array.
    Prefers columns named ['distance','dist','score']; otherwise picks the column with most numeric values.
    """
    df = pd.read_csv(csv_path, sep=None, engine="python")
    if df.shape[1] == 1:
        s = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        return s.dropna().to_numpy(dtype=float)
    preferred = [c for c in df.columns if str(c).strip().lower() in {"distance", "dist", "score"}]
    if preferred:
        s = pd.to_numeric(df[preferred[0]], errors="coerce")
        return s.dropna().to_numpy(dtype=float)
    # fallback: pick the numerically richest column
    best = None; best_count = -1
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        count = s.notna().sum()
        if count > best_count:
            best_count = count; best = s
    return best.dropna().to_numpy(dtype=float)

def collect_records(root: Path, watermark: str, recognizer: str, metric: str,
                    train_filter: Optional[str] = None,
                    test_filter: Optional[List[str]] = None,
                    mode = str) -> List[Dict]:
    """
    Walks the directory tree and collects long-form records.
    Expected layout:
      root / <watermark> / <experiment> / <train_dataset> / <test_dataset> / <recognizer> / distances / *.csv
    """
    records: List[Dict] = []

    algo_dir = root / watermark
    if not algo_dir.is_dir():
        raise FileNotFoundError(f"Algorithm dir not found: {algo_dir}")

    for exp_dir in sorted([p for p in algo_dir.iterdir() if p.is_dir()]):
        bpp = parse_bpp_from_experiment(exp_dir.name)
        # If bpp can't be inferred, skip (avoid polluting stats)
        if bpp is None:
            continue

        # <train_dataset> dirs
        for train_dir in sorted([p for p in exp_dir.iterdir() if p.is_dir()]):
            train_dataset = train_dir.name
            if train_filter and train_dataset.lower() != train_filter.lower():
                continue

            # <test_dataset> dirs
            for test_dir in sorted([p for p in train_dir.iterdir() if p.is_dir()]):
                test_dataset = test_dir.name
                if test_filter and test_dataset not in test_filter:
                    continue

                recog_dir = test_dir / recognizer / "distances"
                if not recog_dir.is_dir():
                    # some structures store distances directly under recognizer
                    recog_dir = test_dir / recognizer
                if not recog_dir.is_dir():
                    # Nothing to do
                    continue

                files = find_distance_files(recog_dir, metric=metric, mode=mode)
                if not files:
                    continue

                for (condition, pair_type), csv_path in files.items():
                    try:
                        arr = read_distance_column(csv_path)
                        if arr.size == 0:
                            continue
                        for val in arr:
                            records.append({
                                "watermark": watermark,
                                "recognizer": recognizer,
                                "metric": metric,
                                "train_dataset": train_dataset,
                                "test_dataset": test_dataset,
                                "bpp": int(bpp),
                                "condition": condition,    # OO, OW, WW
                                "pair_type": pair_type,    # genuine, impostor
                                "distance": float(val),
                                "experiment": exp_dir.name,
                                "path": str(csv_path)
                            })
                    except Exception as e:
                        print(f"[WARN] Failed reading {csv_path}: {e}")
                        continue
    return records

def compute_threshold_by_bpp_and_condition(
    df_dataset: pd.DataFrame,
    far_list: list[float],
    n_jobs: int = -1
) -> dict:
    """
    Calcula el umbral (Threshold) para una lista de Tasas de Falsa Aceptación (FAR)
    específicas para cada combinación de 'bpp' y 'condition' en el DataFrame de un
    único dataset, utilizando procesamiento paralelo.

    Args:
        df_dataset (pd.DataFrame): El DataFrame de entrada (ya filtrado para un solo dataset).
        far_list (list[float]): Lista de valores FAR deseados (e.g., [0.001, 0.0001]).
        n_jobs (int): Número de trabajos paralelos a usar. -1 usa todos los núcleos.

    Returns:
        dict: Un diccionario anidado. La clave exterior es la tupla ('bpp', 'condition'),
              la clave interior es el FAR, y el valor es el Threshold o el mensaje de error.
    """

    # 1. Preparación de los datos: Filtrar solo pares 'impostor'
    columnas_necesarias = ['bpp', 'condition', 'pair_type', 'distance']
    df_impostor = df_dataset[df_dataset['pair_type'] == 'impostor'][columnas_necesarias].copy()

    if df_impostor.empty:
        return {"Error": "El DataFrame no contiene pares de tipo 'impostor'."}

    # 2. Agrupar por 'bpp' y 'condition'
    # Las claves del diccionario serán tuplas (bpp, condition)
    grupos_combinados = {
        (bpp, condition): group['distance'].sort_values().values
        for (bpp, condition), group in df_impostor.groupby(['bpp', 'condition'])
    }

    # 3. Definir la función de trabajo (la lógica de cálculo del umbral no cambia)
    def _compute_th(group_key: tuple, distancias: np.ndarray, far_list: list[float]) -> dict:
        """Calcula los umbrales para una única combinación de BPP y Condition."""
        bpp_value, condition_value = group_key
        resultados_combinacion = {}
        total_impostores = len(distancias)

        for far in far_list:
            far_prop = far * 100

            # --- Criterio de Suficiencia de Datos ---
            min_impostores_necesarios = int(1 / far_prop)

            if total_impostores < min_impostores_necesarios:
                 resultados_combinacion[far] = (
                    f"No puedo calcular un FAR de {far:.4g}%."
                    f" Necesito al menos {min_impostores_necesarios:,} impostores."
                    f" Solo tengo {total_impostores:,} (para BPP={bpp_value}, Cond={condition_value})."
                )
                 continue

            # --- Cálculo del Umbral ---
            threshold = np.percentile(distancias, far_prop)

            resultados_combinacion[far] = threshold

        return {group_key: resultados_combinacion}

    # 4. Ejecutar la función de trabajo en paralelo
    resultados_paralelos = Parallel(n_jobs=n_jobs)(
        delayed(_compute_th)(group_key, dists, far_list)
        for group_key, dists in grupos_combinados.items()
    )

    # 5. Combinar los resultados
    resultados_finales = {}
    for res in resultados_paralelos:
        if res:
            resultados_finales.update(res)

    return resultados_finales

def compute_threshold_by_bpp(
    df_dataset: pd.DataFrame,
    far_list: list[float],
    n_jobs: int = -1
) -> dict:
    """
    Calcula el umbral (Threshold) para una lista de Tasas de Falsa Aceptación (FAR)
    específicas para cada nivel de 'bpp' en el DataFrame de un único dataset,
    utilizando procesamiento paralelo.

    Args:
        df_dataset (pd.DataFrame): El DataFrame de entrada (ya filtrado para un solo dataset).
        far_list (list[float]): Lista de valores FAR deseados (e.g., [0.001, 0.0001]).
        n_jobs (int): Número de trabajos paralelos a usar. -1 usa todos los núcleos.

    Returns:
        dict: Un diccionario anidado. La clave exterior es el valor 'bpp',
              la clave interior es el FAR, y el valor es el Threshold o
              el mensaje de error ("No puedo calcular...").
    """

    # 1. Preparación de los datos: Filtrar solo pares 'impostor'
    # Solo necesitamos 'bpp', 'pair_type' y 'distance'
    df_impostor = df_dataset[df_dataset['pair_type'] == 'impostor'][['bpp', 'distance']].copy()

    if df_impostor.empty:
        return {"Error": "El DataFrame no contiene pares de tipo 'impostor'."}

    # 2. Agrupar por 'bpp'
    grupos_bpp = {
        bpp: group['distance'].sort_values().values
        for bpp, group in df_impostor.groupby('bpp')
    }


    # 3. Definir la función de trabajo (igual que antes, pero el nombre del grupo es 'bpp')
    def _compute_th(bpp_value: int, distancias: np.ndarray, far_list: list[float]) -> dict:
        """Calcula los umbrales para un único valor de BPP."""
        resultados_bpp = {}
        total_impostores = len(distancias)

        for far in far_list:
            far_prop = far * 100

            # --- Criterio de Suficiencia de Datos (Min. 1 comparación por cada 1/FAR) ---
            min_impostores_necesarios = int(1 / far_prop)

            if total_impostores < min_impostores_necesarios:
                 resultados_bpp[far] = (
                    f"No puedo calcular un FAR de {far:.4g}%."
                    f" Necesito al menos {min_impostores_necesarios:,} comparaciones de impostores,"
                    f" pero solo tengo {total_impostores:,} (para BPP={bpp_value})."
                )
                 continue

            # --- Cálculo del Umbral ---
            threshold = np.percentile(distancias, far_prop)


            resultados_bpp[far] = threshold

        return {bpp_value: resultados_bpp}

    # 4. Ejecutar la función de trabajo en paralelo
    resultados_paralelos = Parallel(n_jobs=n_jobs)(
        delayed(_compute_th)(bpp, dists, far_list)
        for bpp, dists in grupos_bpp.items()
    )

    # 5. Combinar los resultados de los diccionarios
    resultados_finales = {}
    for res in resultados_paralelos:
        if res: # Evitar agregar resultados vacíos si hay un error general
            resultados_finales.update(res)

    return resultados_finales

def search_tars_by_far(df_long: pd.DataFrame,
                       fars_by_dataset: dict):
    results = {}
    for dataset_name in fars_by_dataset.keys():
        print(dataset_name)
        fars = fars_by_dataset[dataset_name]
        if dataset_name == 'LFW':
            r_thresholds = {}
            for condition in ['OO', 'OW', 'WW']:
                r_thresholds_con = compute_threshold_by_bpp(
                                            df_dataset=df_long[(df_long.test_dataset==dataset_name) &
                                                            (df_long.condition==condition)].sort_values(by='distance'),
                                            far_list=fars,
                                            n_jobs=-1
                                        )
                for bpp in [1,3,6,8]:
                    r_thresholds[(bpp, condition)] = r_thresholds_con[bpp]
        else:
            r_thresholds = compute_threshold_by_bpp_and_condition(
                                df_dataset=df_long[(df_long.test_dataset==dataset_name)].sort_values(by='distance'),
                                far_list=fars,
                                n_jobs=-1
                                )
        results[dataset_name] = r_thresholds

    return results

def to_serializable(obj):
    """Convierte recursivamente np.int64, np.float64, y claves no válidas para JSON."""
    
    # Convertir numpy scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)

    # Convertir listas
    if isinstance(obj, list):
        return [to_serializable(x) for x in obj]

    # Convertir tuplas (las vuelve listas para JSON)
    if isinstance(obj, tuple):
        return [to_serializable(x) for x in obj]

    # Convertir diccionarios
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            # Convertir claves no serializables (ej: tuplas np.int64 → strings)
            if isinstance(k, tuple):
                k = str(tuple(to_serializable(x) for x in k))
            else:
                k = to_serializable(k)
            new_dict[k] = to_serializable(v)
        return new_dict

    # Dejar pasar valores normales (str, int, float, bool...)
    return obj

def main():
    parser = argparse.ArgumentParser(description="Searching thresholds for FARs")
    parser.add_argument("--root", 
                        default=r'..\experiments\output\recognition')
    parser.add_argument("--mode", choices=["online","offline"], required=False, help="Select which summaries to consolidate",
                        default="online")
    parser.add_argument("--train_dataset_filter", nargs='?', default='coco', help="Filter by training dataset")
    parser.add_argument("--test_datasets", nargs='*', default=['CFD', 'facelab_london', 'LFW', 'ONOT', 'ONOT_set1', 'SCface'], 
                        help="List of test datasets to include")
    parser.add_argument("--metric", default="cosine", help="Distance metric")
    parser.add_argument("--recognizer", default="facenet", help="facial recognizer")
    parser.add_argument("--watermark", default="stegaformer", help="Watermarking algorithm to process")
    args = parser.parse_args()

    root = Path(args.root)

    # 1) Collect long-form records
    records = collect_records(root, args.watermark, args.recognizer, args.metric,
                                train_filter=args.train_dataset_filter,
                                test_filter=set(args.test_datasets) if args.test_datasets else None,
                                mode=args.mode)

    df_long = pd.DataFrame.from_records(records)

    fars_by_dataset = {
        'CFD': [0.001, 0.0001],
        'LFW': [0.001, 0.0001],
        'ONOT_set1': [0.001, 0.0001],
        'ONOT': [0.001],
        'SCface': [0.001],
        'facelab_london': [0.001]
    }

    th_results_by_dataset = search_tars_by_far(df_long, fars_by_dataset)

    data_clean = to_serializable(th_results_by_dataset) 
    file_name = f"threholds_by_fars_{args.watermark}_{args.recognizer}_{args.metric}_{args.mode}_{args.train_dataset_filter}.json"
    with open(r"C:\Users\JotaR\Documents\Github\WAFIS-ICAOeval\evaluation\results\{}".format(file_name), "w") as f:
        json.dump(data_clean, f, indent=4)

if __name__ == "__main__":
    main()