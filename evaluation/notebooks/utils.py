
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import os
import itertools
from scipy.stats import entropy, gaussian_kde
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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

def parse_bpp_from_experiment(exp_name: str) -> Optional[int]:
    
    parts = exp_name.split("_")
    
    if parts[0] == "1" and parts[1] == '1':
        return 1
    elif parts[0] == "1" and parts[1] == '3':
        return 3
    elif parts[0] == "3" and parts[1] == '3' and "stegformer" in exp_name.lower():
        return 3
    elif parts[0] == "3" and parts[1] == '6' and "stegformer" in exp_name.lower():
        return 6
    elif parts[0] == "3" and parts[1] == '3':
        return 6
    elif parts[0] == "15" and parts[1] == '2':
        return 8
    elif parts[0] == "2" and parts[1] == '8' and "stegformer" in exp_name.lower():
        return 8
    else:
        print(parts)
        print(f"[WARN] Could not parse bpp from experiment name: {exp_name}")
        return None

def find_distance_files(dist_dir: Path, metric: str, mode: str, mtcnn: str) -> Dict[Tuple[str, str], Path]:
    """
    Return mapping {(condition, pair_type) -> file_path} if exists.
    Supports offline & online suffixes.
    Condition: OO, OW, WW
    Pair type: genuine, impostor
    preprocess: mtcnn, False no-mtcnn
    """
    if mode == 'online':
        patterns = {
            ("OO", "genuine"):   [f"{metric}_genuine_distances_baseline_online_{mtcnn}.csv"],
            ("OO", "impostor"):  [f"{metric}_impostor_distances_baseline_online_{mtcnn}.csv"],
            ("OW", "genuine"):   [f"{metric}_genuine_distances_watermarked_online_{mtcnn}.csv"],
            ("OW", "impostor"):  [f"{metric}_impostor_distances_watermarked_online_{mtcnn}.csv"],
            ("WW", "genuine"):   [f"{metric}_genuine_distances_watermarked_both_online_{mtcnn}.csv"],
            ("WW", "impostor"):  [f"{metric}_impostor_distances_watermarked_both_online_{mtcnn}.csv"],
        }
    elif mode == 'offline':
        patterns = {
            ("OO", "genuine"):   [f"{metric}_genuine_distances_baseline_{mtcnn}.csv"],
            ("OO", "impostor"):  [f"{metric}_impostor_distances_baseline_{mtcnn}.csv"],
            ("OW", "genuine"):   [f"{metric}_genuine_distances_watermarked_{mtcnn}.csv"],
            ("OW", "impostor"):  [f"{metric}_impostor_distances_watermarked_{mtcnn}.csv"],
            ("WW", "genuine"):   [f"{metric}_genuine_distances_watermarked_both_{mtcnn}.csv"],
            ("WW", "impostor"):  [f"{metric}_impostor_distances_watermarked_both_{mtcnn}.csv"],
        }
        
    else:
        raise ValueError(f"Unknown mode: {mode}")

    out = {}
    for key, candidates in patterns.items():
        for name in candidates:
            p = dist_dir / name
            if p.is_file():
                out[key] = p
                break
    return out

def collect_records(root: Path, watermark: str, recognizer: str, metric: str,
                    train_filter: Optional[str] = None,
                    test_filter: Optional[List[str]] = None,
                    mode = str,
                    mtcnn = str) -> List[Dict]:
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
            raise ValueError(f"Could not parse bpp from experiment name: {exp_dir.name}")

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

                files = find_distance_files(recog_dir, metric=metric, mode=mode, mtcnn=mtcnn)
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
                                "mtcnn":mtcnn,
                                "mode":mode,
                                "path": str(csv_path)
                            })
                    except Exception as e:
                        print(f"[WARN] Failed reading {csv_path}: {e}")
                        continue
    return records

def parse_json_filename(filename: str) -> Dict[str, str]:
    """
    Parsea el nombre del archivo JSON para extraer modalidad y detección facial.
    
    Ejemplos:
        - results_summary_mtcnn.json -> {'modality': 'offline', 'face_detection': 'mtcnn'}
        - results_summary_no-mtcnn.json -> {'modality': 'offline', 'face_detection': 'no-mtcnn'}
        - results_summary_online_mtcnn.json -> {'modality': 'online', 'face_detection': 'mtcnn'}
        - results_summary_online_no-mtcnn.json -> {'modality': 'online', 'face_detection': 'no-mtcnn'}
    """
    name = filename.replace('.json', '')
    
    # Determinar modalidad
    if 'online' in name:
        modality = 'online'
    else:
        modality = 'offline'
    
    # Determinar detección facial
    if 'no-mtcnn' in name:
        face_detection = 'no-mtcnn'
    elif 'mtcnn' in name:
        face_detection = 'mtcnn'
    else:
        face_detection = 'unknown'
    
    return {
        'modality': modality,
        'face_detection': face_detection
    }
 
def flatten_json_results(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aplana el diccionario JSON anidado a un diccionario plano.
    
    El JSON tiene la estructura:
    {
        "average_distances": {...},
        "std_distances": {...},
        "recognition_metrics": {...}
    }
    """
    flat_dict = {}
    
    for category, metrics in json_data.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                if 'facenet' in metric_name.lower() or 'arcface' in metric_name.lower():
                    metric_name = metric_name.replace('arcface_', '').replace('facenet_', '')
                #print(f"Processing metric: {metric_name}, value: {value}")
                # Simplificar el nombre de la métrica
                flat_dict[metric_name] = value
        else:
            flat_dict[category] = metrics
    
    return flat_dict
 
def consolidate_recognition_results(base_path: Union[str, Path]) -> pd.DataFrame:
    """
    Consolida todos los archivos JSON de resultados de reconocimiento facial
    en un DataFrame de pandas con etiquetas de procedencia.
    
    Parameters
    ----------
    base_path : str or Path
        Ruta a la carpeta raíz que contiene los resultados de reconocimiento.
        Debe apuntar a la carpeta 'recognition_output' o equivalente que contenga
        las subcarpetas de los algoritmos de watermarking (stegaformer, stegformer).
    
    Returns
    -------
    pd.DataFrame
        DataFrame con todas las métricas consolidadas y las siguientes columnas de etiquetas:
        - watermarking_algorithm: algoritmo de watermarking (stegaformer, stegformer)
        - experiment_name: nombre completo del experimento
        - bpp: bits per pixel extraído del nombre del experimento
        - train_dataset: dataset de entrenamiento (celeba_hq, coco)
        - test_dataset: dataset de prueba (CFD, etc.)
        - recognizer: reconocedor facial (arcface, facenet)
        - modality: modalidad de evaluación (offline, online)
        - face_detection: método de detección facial (mtcnn, no-mtcnn)
        - json_file: nombre del archivo JSON de origen
        - full_path: ruta completa al archivo JSON
        - [métricas]: todas las métricas del JSON aplanadas
    
    Example
    -------
    >>> df = consolidate_recognition_results('/path/to/recognition_output')
    >>> print(df.columns.tolist())
    ['watermarking_algorithm', 'experiment_name', 'bpp', 'train_dataset', ...]
    
    >>> # Filtrar por experimentos específicos
    >>> df_6bpp = df[df['bpp'] == 6]
    >>> df_mtcnn = df[df['face_detection'] == 'mtcnn']
    """
    base_path = Path(base_path)
    
    if not base_path.exists():
        raise FileNotFoundError(f"La ruta especificada no existe: {base_path}")
    
    results = []
    
    # Buscar todos los archivos JSON de resultados recursivamente
    # Patrón: results_summary*.json
    json_files = list(base_path.rglob('results_summary*.json'))
    
    if not json_files:
        print(f"No se encontraron archivos JSON en: {base_path}")
        return pd.DataFrame()
    
    print(f"Encontrados {len(json_files)} archivos JSON de resultados")
    
    for json_path in json_files:
        try:
            # Extraer etiquetas de la ruta
            # Estructura esperada: base_path/wm_algorithm/experiment/train_dataset/test_dataset/recognizer/
            relative_path = json_path.relative_to(base_path)
            path_parts = relative_path.parts
            
            # Necesitamos al menos 5 partes: algorithm/experiment/train/test/recognizer/file.json
            if len(path_parts) < 5:
                print(f"Advertencia: Ruta inesperada (muy corta): {json_path}")
                continue
            
            # Extraer componentes de la ruta
            watermarking_algorithm = path_parts[0]
            experiment_name = path_parts[1]
            train_dataset = path_parts[2]
            test_dataset = path_parts[3]
            recognizer = path_parts[4]
            json_filename = path_parts[-1]
            
            # Parsear información del nombre del archivo
            file_info = parse_json_filename(json_filename)
            
            # Extraer bpp del nombre del experimento
            bpp = parse_bpp_from_experiment(experiment_name)
            
            # Leer el contenido del JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Aplanar los resultados del JSON
            flat_metrics = flatten_json_results(json_data)
            
            # Crear registro con etiquetas y métricas
            record = {
                'watermarking_algorithm': watermarking_algorithm,
                'experiment_name': experiment_name,
                'bpp': bpp,
                'train_dataset': train_dataset,
                'test_dataset': test_dataset,
                'recognizer': recognizer,
                'modality': file_info['modality'],
                'face_detection': file_info['face_detection'],
                'json_file': json_filename,
                'full_path': str(json_path),
                **flat_metrics
            }
            
            results.append(record)
            
        except Exception as e:
            print(f"Error procesando {json_path}: {e}")
            continue
    
    # Crear DataFrame
    df = pd.DataFrame(results)
    
    # Ordenar columnas: primero las etiquetas, luego las métricas
    label_cols = [
        'watermarking_algorithm', 'experiment_name', 'bpp', 
        'train_dataset', 'test_dataset', 'recognizer', 
        'modality', 'face_detection', 'json_file', 'full_path'
    ]
    metric_cols = [col for col in df.columns if col not in label_cols]
    df = df[label_cols + sorted(metric_cols)]
    
    # Ordenar filas
    df = df.sort_values([
        'watermarking_algorithm', 'bpp', 'train_dataset', 
        'recognizer', 'modality', 'face_detection'
    ]).reset_index(drop=True)
    
    print(f"DataFrame creado con {len(df)} registros y {len(df.columns)} columnas")
    
    return df
 
METRIC_LABELS: Dict[str, str] = {
    # TAR@FAR metrics
    'TAR_at_FAR_watermarked_both_cosine': 'TAR% @FAR=0.1% (WW)',
    'TAR_at_FAR_watermarked_cosine': 'TAR% @FAR=0.1% (OW)',
    'TAR_at_FAR_baseline_cosine': 'TAR% @FAR=0.1% (OO)',
    # EER metrics
    'EER_watermarked_both_cosine': 'EER% (WW)',
    'EER_watermarked_cosine': 'EER% (OW)',
    'EER_baseline_cosine': 'EER% (OO)',
    # FAR@EER metrics
    'FAR_at_EER_watermarked_both_cosine': 'FAR% @EER (WW)',
    'FAR_at_EER_watermarked_cosine': 'FAR% @EER (OW)',
    'FAR_at_EER_baseline_cosine': 'FAR% @EER (OO)',
    # FRR@EER metrics
    'FRR_at_EER_watermarked_both_cosine': 'FRR% @EER (WW)',
    'FRR_at_EER_watermarked_cosine': 'FRR% @EER (OW)',
    'FRR_at_EER_baseline_cosine': 'FRR% @EER (OO)',
    # AUC metrics
    'AUC_watermarked_both_cosine': 'AUC (WW)',
    'AUC_watermarked_cosine': 'AUC (OW)',
    'AUC_baseline_cosine': 'AUC (OO)',
}

METRIC_TO_BASELINE: Dict[str, str] = {
    'TAR_at_FAR_watermarked_both_cosine': 'TAR_at_FAR_baseline_cosine',
    'TAR_at_FAR_watermarked_cosine': 'TAR_at_FAR_baseline_cosine',
    'EER_watermarked_both_cosine': 'EER_baseline_cosine',
    'EER_watermarked_cosine': 'EER_baseline_cosine',
    'FAR_at_EER_watermarked_both_cosine': 'FAR_at_EER_baseline_cosine',
    'FAR_at_EER_watermarked_cosine': 'FAR_at_EER_baseline_cosine',
    'FRR_at_EER_watermarked_both_cosine': 'FRR_at_EER_baseline_cosine',
    'FRR_at_EER_watermarked_cosine': 'FRR_at_EER_baseline_cosine',
    'AUC_watermarked_both_cosine': 'AUC_baseline_cosine',
    'AUC_watermarked_cosine': 'AUC_baseline_cosine',
}
 
 
def get_metric_label(metric: str) -> str:
    """
    Obtiene la etiqueta legible para una métrica.
    Si no está en el mapeo, devuelve la métrica original.
    """
    return METRIC_LABELS.get(metric, metric)
 
def plot_fr_metrics_by_train_dataset(
    df: pd.DataFrame,
    watermarking_algorithm: str,
    test_dataset: str,
    metrics: Union[str, List[str]] = 'TAR_at_FAR_watermarked_both_cosine',
    baseline_metrics: Optional[dict] = None,
    face_detection: str = 'no-mtcnn',
    bpp_order: Sequence[int] = (1, 3, 6, 8),
    dpi: int = 600,
    col_width: float = 4.0,
    row_height: float = 3.2,
    online_linewidth: float = 1.8,
    offline_linewidth: float = 2.1,
    online_band_alpha: float = 0.18,
    offline_band_alpha: float = 0.28,
    offline_linestyle: str = "--",
    online_marker: str = "o",
    offline_marker: str = "s",
    marker_size: float = 5.0,
    save_path: Optional[str] = None,
    color_map_train: Optional[dict] = None,
    show_baseline: bool = True,
) -> plt.Figure:
    """
    Grid de métricas FR vs BPP por training dataset.

    Filas  = métricas (e.g. TAR@FAR WW, TAR@FAR OW, EER …)
    Columnas = FR models (arcface, facenet)

    Líneas:
      - azul  = celeba_hq   |  naranja = coco
      - sólida + ○ = online  |  discontinua + □ = offline

    Leyenda global en la parte inferior.
    """

    # ---- Display names ----
    display_names = {
        "ONOT": "ONOT (set 2)",
        "ONOT_set1": "ONOT (set 1)",
    }

    def _display(ds: str) -> str:
        return display_names.get(ds, ds)

    # ---- Normalizar métricas ----
    metrics_list = [metrics] if isinstance(metrics, str) else list(metrics)
    n_metrics = len(metrics_list)

    # ---- Filtrar ----
    mask = (
        (df["watermarking_algorithm"] == watermarking_algorithm)
        & (df["test_dataset"] == test_dataset)
        & (df["face_detection"] == face_detection)
        & (df["bpp"].isin(bpp_order))
    )
    dff = df[mask].copy()

    if dff.empty:
        raise ValueError(
            f"No data for algorithm='{watermarking_algorithm}', "
            f"test='{test_dataset}', face_detection='{face_detection}'."
        )

    fr_models = sorted(dff["recognizer"].unique())
    train_datasets = sorted(dff["train_dataset"].dropna().unique())
    n_cols = len(fr_models)
    n_rows = n_metrics

    print(f"[INFO] FR models: {fr_models}")
    print(f"[INFO] Training datasets: {train_datasets}")

    # ---- Colores ----
    if color_map_train is None:
        base_palette = {
            "celeba_hq": "#1f77b4",
            "celeba": "#1f77b4",
            "coco": "#ff7f0e",
            "coco2017": "#ff7f0e",
        }
        color_map_train = {
            ds: base_palette.get(ds.lower(), sns.color_palette("tab10")[i % 10])
            for i, ds in enumerate(train_datasets)
        }

    # ---- Baseline mapping ----
    if baseline_metrics is None:
        baseline_metrics = {}
        # auto-detect: si existe METRIC_TO_BASELINE úsalo
        if "METRIC_TO_BASELINE" in dir():
            for m in metrics_list:
                bl = METRIC_TO_BASELINE.get(m)
                if bl:
                    baseline_metrics[m] = bl

    # ---- Estilo visual ----
    sns.set_theme(context="paper", style="whitegrid")
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
    })

    figsize = (col_width * n_cols, row_height * n_rows)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharex=True,
        sharey=False,
        squeeze=False,
        constrained_layout=False,
    )

    # ---- Helper ----
    def _plot_trace(ax, x, y, label, color, linestyle, marker, lw):
        valid = ~(np.isnan(y) | np.isnan(x))
        if valid.sum() == 0:
            return None
        (line,) = ax.plot(
            x[valid],
            y[valid],
            marker=marker,
            markersize=marker_size,
            linestyle=linestyle,
            linewidth=lw,
            label=label,
            color=color,
        )
        return line

    # ---- Dibujar ----
    baseline_styles = {
        "offline": {"linestyle": ":", "color": "#666666", "label": "Baseline Offline (OO)"},
        "online": {"linestyle": "-.", "color": "#999999", "label": "Baseline Online (OO)"},
    }

    for row_idx, metric in enumerate(metrics_list):
        metric_label = get_metric_label(metric)
        bl_metric = baseline_metrics.get(metric)

        for col_idx, fr_model in enumerate(fr_models):
            ax = axes[row_idx, col_idx]
            df_cell = dff[dff["recognizer"] == fr_model]

            if metric not in df_cell.columns:
                ax.text(0.5, 0.5, f"No column:\n{metric}", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8, color="red")
                continue

            baseline_values = {}

            for ds in train_datasets:
                color = color_map_train[ds]
                for modality, ls, mk, lw in [
                    ("online", "-", online_marker, online_linewidth),
                    ("offline", offline_linestyle, offline_marker, offline_linewidth),
                ]:
                    sub = df_cell[
                        (df_cell["train_dataset"] == ds)
                        & (df_cell["modality"] == modality)
                    ].sort_values("bpp")

                    if sub.empty:
                        continue

                    x = sub["bpp"].to_numpy(dtype=float)
                    y = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)

                    _plot_trace(
                        ax, x, y,
                        f"{modality} ({ds})",
                        color, ls, mk, lw,
                    )

                    # Baseline (solo de celeba_hq para no duplicar)
                    if show_baseline and bl_metric and bl_metric in sub.columns and ds == "celeba_hq":
                        bl_val = pd.to_numeric(sub[bl_metric], errors="coerce").mean()
                        if not np.isnan(bl_val):
                            baseline_values[modality] = bl_val

            # Dibujar baselines
            if show_baseline:
                same_bl = (
                    "offline" in baseline_values
                    and "online" in baseline_values
                    and np.isclose(baseline_values["offline"], baseline_values["online"], atol=1e-6)
                )
                for modality, bl_val in baseline_values.items():
                    bstyle = baseline_styles[modality]
                    ax.axhline(y=bl_val, linestyle=bstyle["linestyle"],
                               color=bstyle["color"], linewidth=1.2, alpha=0.7)
                    if same_bl:
                        x_text = 0.05 if modality == "offline" else 0.95
                        ha = "left" if modality == "offline" else "right"
                        va = "bottom" if modality == "offline" else "top"
                    else:
                        x_text, ha, va = 0.60, "right", "bottom"

                    ax.text(x_text, bl_val, bstyle["label"],
                            color=bstyle["color"], fontsize=7,
                            va=va, ha=ha, transform=ax.get_yaxis_transform())

            # Ejes
            ax.set_xticks(list(bpp_order))
            ax.grid(True, linestyle="--", alpha=0.35)

            # Título columna (solo primera fila)
            if row_idx == 0:
                ax.set_title(fr_model.upper(), fontweight="bold")

            # Y-label (solo primera columna)
            if col_idx == 0:
                ax.set_ylabel(metric_label)

            # X-label (solo última fila)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Bits per pixel (BPP)")
            else:
                ax.set_xlabel("")

    # ---- Suptitle ----
    fig.suptitle(
        f"{watermarking_algorithm.upper()} - Test: {_display(test_dataset)}",
        y=0.99, fontsize=12, fontweight="bold",
    )

    # ---- Leyenda global ----
    # Construir leyenda manual para evitar duplicados
    legend_elements = []
    for ds in train_datasets:
        color = color_map_train[ds]
        legend_elements.append(
            plt.Line2D([0], [0], color=color, marker=online_marker, linestyle="-",
                       linewidth=online_linewidth, markersize=marker_size,
                       label=f"online ({ds})")
        )
        legend_elements.append(
            plt.Line2D([0], [0], color=color, marker=offline_marker, linestyle=offline_linestyle,
                       linewidth=offline_linewidth, markersize=marker_size,
                       label=f"offline ({ds})")
        )
    if show_baseline:
        for mod, bstyle in baseline_styles.items():
            legend_elements.append(
                plt.Line2D([0], [0], color=bstyle["color"], linestyle=bstyle["linestyle"],
                           linewidth=1.2, label=bstyle["label"])
            )

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=min(len(legend_elements), 6),
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    plt.tight_layout(rect=(0, 0.02, 1, 0.96))

    # ---- Guardar ----
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ext = os.path.splitext(save_path)[1].lower()
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi) #transparent=(ext == ".png")
        print(f"[INFO] Saved to {save_path}")

    return fig
 
def plot_wm_metrics_by_train_for_test(
    excel_path: str,
    save_path: str,
    algorithms: Union[str, List[str]],
    inference_datasets: Union[str, List[str]],
    bpp_order: Sequence[int] = (1, 3, 6, 8),
    dpi: int = 600,
    col_width: float = 4.0,
    row_height: float = 3.2,
    hide_offline_accuracy_std: bool = False,
    acc_ylim_top: float = 1.02,
    ssim_min_margin: float = 0.002,
    ssim_max_margin: float = 0.005,
    online_linewidth: float = 1.8,
    offline_linewidth: float = 2.1,
    online_band_alpha: float = 0.18,
    offline_band_alpha: float = 0.28,
    offline_linestyle: str = "--",
    online_marker: str = "o",
    offline_marker: str = "s",
    marker_size: float = 5.0,
    color_map_train: Optional[dict] = None,
) -> None:
    """
    Para uno o varios datasets de test, compara el rendimiento de diferentes
    datasets de entrenamiento en métricas ACC, PSNR, SSIM (online/offline).

    Cada inference_dataset ocupa una fila; las columnas son ACC / PSNR / SSIM.
    Las columnas comparten eje‑Y y todas comparten eje‑X.
    """
    def _display(ds: str) -> str:
        return display_names.get(ds, ds)

    # ---- Cargar Excel ----
    df = pd.read_excel(excel_path)

    # ---- Normalizar inputs ----
    algos = [algorithms] if isinstance(algorithms, str) else list(algorithms)
    datasets = [inference_datasets] if isinstance(inference_datasets, str) else list(inference_datasets)
    n_rows = len(datasets)
    n_cols = 3  # ACC, PSNR, SSIM

    # ---- Display names for inference datasets ----
    display_names = {
        "ONOT": "ONOT (set 2)",
        "ONOT_set1": "ONOT (set 1)",
    }
    
    # ---- Filtrar ----
    dff = df[
        (df["inference_dataset"].isin(datasets)) &
        (df["algorithm"].isin(algos)) &
        (df["bpp"].isin(bpp_order))
    ].copy()

    if dff.empty:
        raise ValueError(
            f"No rows found for inference_datasets={datasets} and algorithms={algos}."
        )

    # ---- Training datasets disponibles (global) ----
    train_datasets = sorted(dff["training_dataset"].dropna().unique().tolist())
    if not train_datasets:
        raise ValueError("No training_dataset found.")
    print(f"[INFO] Training datasets found: {train_datasets}")

    # ---- Color por training_dataset ----
    if color_map_train is None:
        base_palette = {
            "celeba_hq": "#1f77b4",
            "celeba": "#1f77b4",
            "coco": "#ff7f0e",
            "coco2017": "#ff7f0e",
        }
        color_map_train = {
            ds: base_palette.get(ds.lower(), sns.color_palette("tab10")[i % 10])
            for i, ds in enumerate(train_datasets)
        }

    # ---- Agregación ----
    def _combine_std(std_series: pd.Series) -> float:
        vals = pd.to_numeric(std_series, errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            return np.nan
        return float(np.sqrt(np.nanmean(vals ** 2)))

    agg = dff.groupby(["inference_dataset", "training_dataset", "algorithm", "bpp"]).agg({
        "accuracy": "mean",
        "accuracy_std": _combine_std,
        "accuracy_offline": "mean",
        "accuracy_offline_std": _combine_std,
        "psnr": "mean",
        "psnr_std": _combine_std,
        "psnr_offline": "mean",
        "psnr_offline_std": _combine_std,
        "ssim": "mean",
        "ssim_std": _combine_std,
        "ssim_offline": "mean",
        "ssim_offline_std": _combine_std,
    }).reset_index()

    agg["bpp"] = pd.Categorical(agg["bpp"], categories=list(bpp_order), ordered=True)
    agg = agg.sort_values(
        ["inference_dataset", "training_dataset", "bpp", "algorithm"]
    ).reset_index(drop=True)

    # ---- Calcular límites globales para ejes compartidos ----
    def _global_limits(col_on, col_off, margin_lo=0.0, margin_hi=0.0, floor=None, ceil=None):
        vals = pd.to_numeric(
            pd.concat([agg[col_on], agg[col_off]]), errors="coerce"
        ).dropna().to_numpy(dtype=float)
        if vals.size == 0:
            return (0, 1)
        lo = float(np.nanmin(vals)) - margin_lo
        hi = float(np.nanmax(vals)) + margin_hi
        if floor is not None:
            lo = max(floor, lo)
        if ceil is not None:
            hi = min(ceil, hi)
        return (lo, hi)

    acc_ylim = (0.3, acc_ylim_top)
    psnr_ylim = _global_limits("psnr", "psnr_offline", margin_lo=1.0, margin_hi=1.0)
    ssim_ylim = _global_limits(
        "ssim", "ssim_offline",
        margin_lo=ssim_min_margin, margin_hi=ssim_max_margin,
        floor=0.2, ceil=1.02,
    )

    # ---- Configuración visual ----
    sns.set_theme(context="paper", style="whitegrid")
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
    })

    figsize = (col_width * n_cols, row_height * n_rows)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=figsize,
        sharex=True,
        sharey="col",            # cada columna comparte eje‑Y
        squeeze=False,           # siempre 2‑D
        constrained_layout=False,
    )

    # ---- Helper para dibujar ----
    def _plot_trace(ax, x, y, ystd, label, color, linestyle, marker, lw, alpha_band):
        valid = ~(np.isnan(y) | np.isnan(x))
        if valid.sum() == 0:
            return None
        (line,) = ax.plot(
            x[valid], y[valid],
            marker=marker, markersize=marker_size,
            linestyle=linestyle, linewidth=lw,
            label=label, color=color,
        )
        if ystd is not None:
            ystd = np.where(np.isnan(ystd), 0.0, ystd)
            ax.fill_between(
                x[valid],
                y[valid] - ystd[valid],
                y[valid] + ystd[valid],
                alpha=alpha_band, linewidth=0, color=color,
            )
        return line

    def _draw_panel(ax, inf_ds, metric_key, metric_std_key, ylim):
        sub_ds = agg[agg["inference_dataset"] == inf_ds]
        for ds in train_datasets:
            for algo in algos:
                sub = sub_ds[(sub_ds["training_dataset"] == ds) & (sub_ds["algorithm"] == algo)]
                if sub.empty:
                    continue

                x = sub["bpp"].astype(int).to_numpy()
                color = color_map_train[ds]

                # ONLINE
                y_on = pd.to_numeric(sub[metric_key], errors="coerce").to_numpy(dtype=float)
                s_on = pd.to_numeric(sub[metric_std_key], errors="coerce").to_numpy(dtype=float)
                _plot_trace(
                    ax, x, y_on, s_on,
                    #f"{algo} ({ds} · online)", color,
                    f"{ds} · online", color,
                    "-", online_marker, online_linewidth, online_band_alpha,
                )

                # OFFLINE
                off_key = metric_key.split("_")[0]  # accuracy / psnr / ssim
                y_off = pd.to_numeric(sub[f"{off_key}_offline"], errors="coerce").to_numpy(dtype=float)
                s_off = pd.to_numeric(sub[f"{off_key}_offline_std"], errors="coerce").to_numpy(dtype=float)
                #if off_key == "accuracy" and hide_offline_accuracy_std:
                #    s_off = np.zeros_like(s_off)

                _plot_trace(
                    ax, x, y_off, s_off,
                    #f"{algo} ({ds} · offline)", color,
                    f"{ds} · offline", color,
                    offline_linestyle, offline_marker, offline_linewidth, offline_band_alpha,
                )

        ax.set_ylim(ylim)
        ax.set_xticks(list(bpp_order))
        ax.grid(True, linestyle="--", alpha=0.35)

    # ---- Dibujar todas las celdas ----
    metric_spec = [
        ("accuracy", "accuracy_std", "ACC", "Accuracy", acc_ylim),
        ("psnr", "psnr_std", "PSNR", "PSNR (dB)", psnr_ylim),
        ("ssim", "ssim_std", "SSIM", "SSIM", ssim_ylim),
    ]

    for row_idx, inf_ds in enumerate(datasets):
        for col_idx, (mkey, mstd, title, ylabel, ylim) in enumerate(metric_spec):
            ax = axes[row_idx, col_idx]
            _draw_panel(ax, inf_ds, mkey, mstd, ylim)

            # Títulos de columna solo en la primera fila
            if row_idx == 0:
                ax.set_title(title)

            # Y‑label solo en la primera columna
            if col_idx == 0:
                ax.set_ylabel(f"{_display(inf_ds)}\n{ylabel}")
            else:
                ax.set_ylabel("")

            # X‑label solo en la última fila
            if row_idx == n_rows - 1:
                ax.set_xlabel("Bits per pixel (BPP)")
            else:
                ax.set_xlabel("")

    # ---- Suptitle ----
    algo_str = ", ".join(algos)
    fig.suptitle(f"Online-to-Offline Degradation in Watermark Quality and Recovery Across Training and Evaluation Domains", y=0.99, fontsize=12) 

    # ---- Leyenda global ----
    handles_labels = [ax.get_legend_handles_labels() for ax in axes.flat]
    handles = sum((hl[0] for hl in handles_labels), [])
    labels = sum((hl[1] for hl in handles_labels), [])
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]

    fig.legend(
        [h for h, _ in uniq],
        [l for _, l in uniq],
        loc="lower center",
        ncol=min(len(uniq), max(4, 2 * len(algos))),
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    plt.tight_layout(rect=(0, 0.02, 1, 0.98))

    # ---- Guardar ----
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ext = os.path.splitext(save_path)[1].lower()
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi )#transparent=(ext == ".png")
    #else:
    plt.show()
    plt.close()

def plot_wm_bar_psnr_by_algorithm_for_test(
    excel_path: str,
    save_path: str,
    algorithms: Sequence[str] = ("stegaformer", "stegformer"),
    inference_datasets: Sequence[str] = ("CFD", "ONOT_set1", "LFW"),
    bpp_order: Sequence[int] = (1, 3, 6, 8),
    dpi: int = 600,
    col_width: float = 3.8,
    row_height: float = 3.2,
    bar_ylim_top: float = 1.02,
    online_linewidth: float = 1.8,
    offline_linewidth: float = 2.1,
    online_band_alpha: float = 0.18,
    offline_band_alpha: float = 0.28,
    offline_linestyle: str = "--",
    online_marker: str = "o",
    offline_marker: str = "s",
    marker_size: float = 5.0,
    color_map_train: Optional[dict] = None,
) -> None:
    """
    Compara BAR y PSNR para varios datasets de inferencia y dos algoritmos.

    Organización de columnas:
        1. BAR — StegaFormer
        2. BAR — StegFormer
        3. PSNR — StegaFormer
        4. PSNR — StegFormer

    Cada fila corresponde a un inference_dataset.
    Las curvas comparan training_dataset y modalidad online/offline.
    """

    def _display(ds: str) -> str:
        return display_names.get(ds, ds)

    def _display_algo(algo: str) -> str:
        return algorithm_display_names.get(algo.lower(), algo)

    # ---- Cargar Excel ----
    df = pd.read_excel(excel_path)

    algos = list(algorithms)
    datasets = list(inference_datasets)

    n_rows = len(datasets)
    n_cols = 4

    # ---- Nombres visibles ----
    display_names = {
        "ONOT": "ONOT (set 2)",
        "ONOT_set1": "ONOT (set 1)",
        "CFD": "CFD",
        "LFW": "LFW",
    }

    algorithm_display_names = {
        "stegaformer": "StegaFormer",
        "stegformer": "StegFormer",
    }

    # ---- Filtrar ----
    dff = df[
        (df["inference_dataset"].isin(datasets)) &
        (df["algorithm"].isin(algos)) &
        (df["bpp"].isin(bpp_order))
    ].copy()

    if dff.empty:
        raise ValueError(
            f"No rows found for inference_datasets={datasets} and algorithms={algos}."
        )

    # ---- Training datasets disponibles ----
    train_datasets = sorted(dff["training_dataset"].dropna().unique().tolist())

    if not train_datasets:
        raise ValueError("No training_dataset found.")

    print(f"[INFO] Training datasets found: {train_datasets}")

    # ---- Color por training_dataset ----
    if color_map_train is None:
        base_palette = {
            "celeba_hq": "#1f77b4",
            "celeba": "#1f77b4",
            "coco": "#ff7f0e",
            "coco2017": "#ff7f0e",
        }

        color_map_train = {
            ds: base_palette.get(ds.lower(), sns.color_palette("tab10")[i % 10])
            for i, ds in enumerate(train_datasets)
        }

    # ---- Combinar std cuando haya varias filas ----
    def _combine_std(std_series: pd.Series) -> float:
        vals = pd.to_numeric(std_series, errors="coerce").dropna().to_numpy(dtype=float)

        if vals.size == 0:
            return np.nan

        return float(np.sqrt(np.nanmean(vals ** 2)))

    # ---- Agregación ----
    agg = dff.groupby(
        ["inference_dataset", "training_dataset", "algorithm", "bpp"]
    ).agg({
        "accuracy": "mean",
        "accuracy_std": _combine_std,
        "accuracy_offline": "mean",
        "accuracy_offline_std": _combine_std,
        "psnr": "mean",
        "psnr_std": _combine_std,
        "psnr_offline": "mean",
        "psnr_offline_std": _combine_std,
    }).reset_index()

    agg["bpp"] = pd.Categorical(
        agg["bpp"],
        categories=list(bpp_order),
        ordered=True
    )

    agg = agg.sort_values(
        ["inference_dataset", "algorithm", "training_dataset", "bpp"]
    ).reset_index(drop=True)

    # ---- Límites globales ----
    def _global_limits(
        col_on: str,
        col_off: str,
        margin_lo: float = 0.0,
        margin_hi: float = 0.0,
        floor: Optional[float] = None,
        ceil: Optional[float] = None,
    ):
        vals = pd.to_numeric(
            pd.concat([agg[col_on], agg[col_off]]),
            errors="coerce"
        ).dropna().to_numpy(dtype=float)

        if vals.size == 0:
            return (0, 1)

        lo = float(np.nanmin(vals)) - margin_lo
        hi = float(np.nanmax(vals)) + margin_hi

        if floor is not None:
            lo = max(floor, lo)

        if ceil is not None:
            hi = min(ceil, hi)

        return (lo, hi)

    bar_ylim = (0.3, bar_ylim_top)

    psnr_ylim = _global_limits(
        "psnr",
        "psnr_offline",
        margin_lo=1.0,
        margin_hi=1.0
    )

    # ---- Estilo visual ----
    sns.set_theme(context="paper", style="whitegrid")

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
    })

    figsize = (col_width * n_cols, row_height * n_rows)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharex=True,
        squeeze=False,
        constrained_layout=False,
    )

    # ---- Helper para dibujar curvas ----
    def _plot_trace(
        ax,
        x,
        y,
        ystd,
        label,
        color,
        linestyle,
        marker,
        lw,
        alpha_band,
    ):
        valid = ~(np.isnan(y) | np.isnan(x))

        if valid.sum() == 0:
            return None

        line, = ax.plot(
            x[valid],
            y[valid],
            marker=marker,
            markersize=marker_size,
            linestyle=linestyle,
            linewidth=lw,
            label=label,
            color=color,
        )

        if ystd is not None:
            ystd = np.where(np.isnan(ystd), 0.0, ystd)

            ax.fill_between(
                x[valid],
                y[valid] - ystd[valid],
                y[valid] + ystd[valid],
                alpha=alpha_band,
                linewidth=0,
                color=color,
            )

        return line

    # ---- Helper para cada panel ----
    def _draw_panel(
        ax,
        inf_ds: str,
        algo: str,
        metric_key: str,
        metric_std_key: str,
        ylim,
    ):
        sub_ds = agg[
            (agg["inference_dataset"] == inf_ds) &
            (agg["algorithm"] == algo)
        ]

        for ds in train_datasets:
            sub = sub_ds[sub_ds["training_dataset"] == ds].sort_values("bpp")

            if sub.empty:
                continue

            x = sub["bpp"].astype(int).to_numpy()
            color = color_map_train[ds]

            # ONLINE
            y_on = pd.to_numeric(
                sub[metric_key],
                errors="coerce"
            ).to_numpy(dtype=float)

            s_on = pd.to_numeric(
                sub[metric_std_key],
                errors="coerce"
            ).to_numpy(dtype=float)

            _plot_trace(
                ax=ax,
                x=x,
                y=y_on,
                ystd=s_on,
                label=f"{ds} · online",
                color=color,
                linestyle="-",
                marker=online_marker,
                lw=online_linewidth,
                alpha_band=online_band_alpha,
            )

            # OFFLINE
            off_key = metric_key.split("_")[0]

            y_off = pd.to_numeric(
                sub[f"{off_key}_offline"],
                errors="coerce"
            ).to_numpy(dtype=float)

            s_off = pd.to_numeric(
                sub[f"{off_key}_offline_std"],
                errors="coerce"
            ).to_numpy(dtype=float)

            _plot_trace(
                ax=ax,
                x=x,
                y=y_off,
                ystd=s_off,
                label=f"{ds} · offline",
                color=color,
                linestyle=offline_linestyle,
                marker=offline_marker,
                lw=offline_linewidth,
                alpha_band=offline_band_alpha,
            )

        ax.set_ylim(ylim)
        ax.set_xticks(list(bpp_order))
        ax.grid(True, linestyle="--", alpha=0.35)

    # ---- Especificación de columnas ----
    # Orden pedido:
    # BAR StegaFormer | BAR StegFormer | PSNR StegaFormer | PSNR StegFormer
    column_spec = [
        ("accuracy", "accuracy_std", algos[0], "BAR", "BAR", bar_ylim),
        ("accuracy", "accuracy_std", algos[1], "BAR", "BAR", bar_ylim),
        ("psnr", "psnr_std", algos[0], "PSNR", "PSNR (dB)", psnr_ylim),
        ("psnr", "psnr_std", algos[1], "PSNR", "PSNR (dB)", psnr_ylim),
    ]

    # ---- Dibujar ----
    for row_idx, inf_ds in enumerate(datasets):
        for col_idx, (mkey, mstd, algo, metric_title, ylabel, ylim) in enumerate(column_spec):
            ax = axes[row_idx, col_idx]

            _draw_panel(
                ax=ax,
                inf_ds=inf_ds,
                algo=algo,
                metric_key=mkey,
                metric_std_key=mstd,
                ylim=ylim,
            )

            # Título de columna solo en la primera fila
            if row_idx == 0:
                ax.set_title(f"{metric_title} — {_display_algo(algo)}")

            # Y-label
            if col_idx == 0:
                ax.set_ylabel(f"{_display(inf_ds)}\n{ylabel}")
            elif col_idx == 2:
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel("")

            # X-label solo en la última fila
            if row_idx == n_rows - 1:
                ax.set_xlabel("Bits per pixel (BPP)")
            else:
                ax.set_xlabel("")

    # ---- Ajustar ejes Y compartidos manualmente por tipo de métrica ----
    # Columnas 0 y 1: BAR
    for row_idx in range(n_rows):
        axes[row_idx, 0].set_ylim(bar_ylim)
        axes[row_idx, 1].set_ylim(bar_ylim)

    # Columnas 2 y 3: PSNR
    for row_idx in range(n_rows):
        axes[row_idx, 2].set_ylim(psnr_ylim)
        axes[row_idx, 3].set_ylim(psnr_ylim)

    # ---- Título general ----
    fig.suptitle(
        "Online-to-Offline Degradation in Watermark Quality and Recovery Across Training and Evaluation Domains",
        y=0.965,
        fontsize=16,
    )

    # ---- Leyenda global ----
    handles_labels = [ax.get_legend_handles_labels() for ax in axes.flat]

    handles = sum((hl[0] for hl in handles_labels), [])
    labels = sum((hl[1] for hl in handles_labels), [])

    seen = set()
    uniq = [
        (h, l)
        for h, l in zip(handles, labels)
        if not (l in seen or seen.add(l))
    ]

    fig.legend(
        [h for h, _ in uniq],
        [l for _, l in uniq],
        loc="lower center",
        ncol=min(len(uniq), 4),
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    plt.tight_layout(rect=(0, 0.03, 1, 0.97))

    # ---- Guardar ----
    if save_path is not None:
        save_dir = os.path.dirname(save_path)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        plt.savefig(
            save_path,
            bbox_inches="tight",
            dpi=dpi,
        )

    plt.show()
    plt.close()

def plot_wm_metrics_from_excel(
    excel_path: str,
    save_path: str,
    algorithms: Union[str, List[str]],
    training_dataset: str,
    inference_dataset: str,
    bpp_order: Sequence[int] = (1, 3, 6, 8),
    dpi: int = 600,
    figsize: tuple = (12, 4),
    palette: Optional[Sequence[str]] = None,
    # ---- tweaks / opciones ----
    hide_offline_accuracy_std: bool = True,   # parche temporal
    acc_ylim_top: float = 1.02,               # margen superior p/ACC
    ssim_min_margin: float = 0.002,           # margen inferior dinámico p/SSIM
    ssim_max_margin: float = 0.005,           # margen superior dinámico p/SSIM
    online_linewidth: float = 1.8,
    offline_linewidth: float = 2.0,
    online_band_alpha: float = 0.20,
    offline_band_alpha: float = 0.30,
    offline_linestyle: str = "--",            # más diferenciación visual
    online_marker: str = "o",
    offline_marker: str = "s",
    marker_size: float = 5.0,) -> None:
    """
    Plot ACC, PSNR, SSIM (online & offline) vs BPP para uno o varios algoritmos.
    - Leyenda global abajo, título con Train/Test.
    - SSIM: auto rango vertical estrecho para que se vean cambios sutiles.
    - Parche: oculta la banda std offline de ACC si hide_offline_accuracy_std=True.
    """

    # ---------- Load ----------
    df = pd.read_excel(excel_path)

    # ---------- Normalize inputs ----------
    algos = [algorithms] if isinstance(algorithms, str) else list(algorithms)

    # Filter datasets
    mask = (
        (df['training_dataset'] == training_dataset) &
        (df['inference_dataset'] == inference_dataset) &
        (df['algorithm'].isin(algos))
    )
    dff = df.loc[mask].copy()
    if dff.empty:
        raise ValueError(
            f"No rows after filtering with training_dataset='{training_dataset}', "
            f"inference_dataset='{inference_dataset}', algorithms={algos}."
        )

    # BPP filter/order
    dff = dff[dff['bpp'].isin(bpp_order)].copy()
    if dff.empty:
        raise ValueError("No rows matching the provided bpp_order.")

    # ---------- Aggregation ----------
    def _combine_std(std_series: pd.Series) -> float:
        vals = pd.to_numeric(std_series, errors='coerce').dropna().to_numpy(dtype=float)
        if vals.size == 0:
            return np.nan
        return float(np.sqrt(np.nanmean(vals**2)))

    agg = dff.groupby(['algorithm', 'bpp']).agg({
        'accuracy': 'mean',
        'accuracy_std': _combine_std,
        'accuracy_offline': 'mean',
        'accuracy_offline_std': _combine_std,
        'psnr': 'mean',
        'psnr_std': _combine_std,
        'psnr_offline': 'mean',
        'psnr_offline_std': _combine_std,
        'ssim': 'mean',
        'ssim_std': _combine_std,
        'ssim_offline': 'mean',
        'ssim_offline_std': _combine_std,
    }).reset_index()

    # Order BPP
    cat = pd.Categorical(agg['bpp'], categories=list(bpp_order), ordered=True)
    agg = agg.assign(bpp=cat).sort_values(['bpp', 'algorithm']).reset_index(drop=True)

    # ---------- Plot config ----------
    sns.set_theme(context="paper", style="whitegrid")
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
        "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "figure.dpi": dpi, "savefig.dpi": dpi,
    })

    if palette is None:
        palette = sns.color_palette("tab20", n_colors=max(6, 2*len(algos)))

    # Color por (algoritmo, modo)
    modes = ["online", "offline"]
    color_map = {}
    c_idx = 0
    for algo in algos:
        for mode in modes:
            color_map[(algo, mode)] = palette[c_idx % len(palette)]
            c_idx += 1

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=False)

    def _plot_trace(ax, x, y, ystd, label, color, marker, lw, alpha_band, linestyle="-"):
        valid = ~(np.isnan(y) | np.isnan(x))
        if valid.sum() == 0:
            return None
        line, = ax.plot(
            x[valid], y[valid],
            marker=marker, markersize=marker_size,
            linestyle=linestyle, linewidth=lw,
            label=label, color=color
        )
        if ystd is not None:
            ystd = np.where(np.isnan(ystd), 0.0, ystd)
            ax.fill_between(
                x[valid],
                (y[valid] - ystd[valid]),
                (y[valid] + ystd[valid]),
                alpha=alpha_band, linewidth=0, color=color
            )
        return line

    def _draw_panel(ax, metric_key: str, metric_std_key: str, title: str, auto_tight: bool = False):
        for algo in algos:
            sub = agg[agg['algorithm'] == algo]
            if sub.empty:
                continue

            x = sub['bpp'].astype(int).to_numpy()

            # ONLINE
            y_on = pd.to_numeric(sub[metric_key], errors='coerce').to_numpy(dtype=float)
            s_on = pd.to_numeric(sub[metric_std_key], errors='coerce').to_numpy(dtype=float)
            _plot_trace(ax, x, y_on, s_on, f"{algo} (online)",
                        color_map[(algo,"online")], online_marker,
                        online_linewidth, online_band_alpha, "-")

            # OFFLINE
            if metric_key.startswith("accuracy"):
                y_off = pd.to_numeric(sub['accuracy_offline'], errors='coerce').to_numpy(dtype=float)
                s_off = pd.to_numeric(sub['accuracy_offline_std'], errors='coerce').to_numpy(dtype=float)
                if hide_offline_accuracy_std:
                    s_off = np.zeros_like(s_off)
            elif metric_key.startswith("psnr"):
                y_off = pd.to_numeric(sub['psnr_offline'], errors='coerce').to_numpy(dtype=float)
                s_off = pd.to_numeric(sub['psnr_offline_std'], errors='coerce').to_numpy(dtype=float)
            elif metric_key.startswith("ssim"):
                y_off = pd.to_numeric(sub['ssim_offline'], errors='coerce').to_numpy(dtype=float)
                s_off = pd.to_numeric(sub['ssim_offline_std'], errors='coerce').to_numpy(dtype=float)
            else:
                raise ValueError("Unknown metric key pattern.")

            _plot_trace(ax, x, y_off, s_off, f"{algo} (offline)",
                        color_map[(algo,"offline")], offline_marker,
                        offline_linewidth, offline_band_alpha, offline_linestyle)

        ax.set_title(title)
        ax.set_xlabel("Bits per pixel (BPP)")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

        if metric_key.startswith("accuracy"):
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0.0, acc_ylim_top)
        elif metric_key.startswith("psnr"):
            ax.set_ylabel("PSNR (dB)")
        elif metric_key.startswith("ssim"):
            ax.set_ylabel("SSIM")
            # --- auto-rango SSIM: estrecho en torno a los datos visibles ---
            y_all = []
            for algo in algos:
                sub = agg[agg['algorithm'] == algo]
                if sub.empty:
                    continue
                y_all.append(pd.to_numeric(sub['ssim'], errors='coerce').to_numpy(dtype=float))
                y_all.append(pd.to_numeric(sub['ssim_offline'], errors='coerce').to_numpy(dtype=float))
            y_all = np.concatenate([a[~np.isnan(a)] for a in y_all if a is not None]) if y_all else np.array([])
            if y_all.size > 0:
                y_min, y_max = float(np.nanmin(y_all)), float(np.nanmax(y_all))
                lo = max(0.0, y_min - ssim_min_margin)
                hi = min(1.02, y_max + ssim_max_margin)
                if hi <= lo:  # fallback por seguridad
                    lo, hi = 0.0, 1.02
                ax.set_ylim(lo, hi)
            else:
                ax.set_ylim(0.0, 1.02)

        ax.set_xticks(list(bpp_order))

    # Panels
    _draw_panel(axes[0], "accuracy", "accuracy_std", "ACC")
    _draw_panel(axes[1], "psnr", "psnr_std", "PSNR")
    _draw_panel(axes[2], "ssim", "ssim_std", "SSIM")

    # Título general
    fig.suptitle(f"Train: {training_dataset}  |  Test: {inference_dataset}", y=1.03)

    # Leyenda fuera abajo
    handles_labels = [ax.get_legend_handles_labels() for ax in axes]
    handles = sum((hl[0] for hl in handles_labels), [])
    labels = sum((hl[1] for hl in handles_labels), [])
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]

    fig.legend(
        [h for h, _ in uniq],
        [l for _, l in uniq],
        loc="lower center",
        ncol=min(len(uniq), max(4, 2*len(algos))),
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )

    # Espacio para suptitle y leyenda
    plt.tight_layout(rect=(0, 0.08, 1, 0.95))

    # Save / Show
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ext = os.path.splitext(save_path)[1].lower()
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi, transparent=(ext == ".png"))
    else:
        plt.show()
    plt.close()

def get_summary_by_scenario(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un resumen de las métricas principales agrupadas por escenario.
    
    Escenarios de watermarking:
    - OO (baseline): sin marca de agua en template ni probe
    - OW (watermarked): solo el probe tiene marca de agua  
    - WW (watermarked_both): ambos tienen marca de agua
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame generado por consolidate_recognition_results()
    
    Returns
    -------
    pd.DataFrame
        DataFrame resumido con las métricas EER y AUC por escenario
    """
    # Columnas de métricas de interés por escenario
    scenario_metrics = {
        'OO': {
            'EER': 'arcface_EER_baseline_cosine',
            'AUC': 'arcface_AUC_baseline_cosine',
            'TAR_at_FAR': 'arcface_TAR_at_FAR_baseline_cosine'
        },
        'OW': {
            'EER': 'arcface_EER_watermarked_cosine',
            'AUC': 'arcface_AUC_watermarked_cosine',
            'TAR_at_FAR': 'arcface_TAR_at_FAR_watermarked_cosine'
        },
        'WW': {
            'EER': 'arcface_EER_watermarked_both_cosine',
            'AUC': 'arcface_AUC_watermarked_both_cosine',
            'TAR_at_FAR': 'arcface_TAR_at_FAR_watermarked_both_cosine'
        }
    }
    
    summary_records = []
    
    for _, row in df.iterrows():
        base_info = {
            'watermarking_algorithm': row['watermarking_algorithm'],
            'experiment_name': row['experiment_name'],
            'bpp': row['bpp'],
            'train_dataset': row['train_dataset'],
            'test_dataset': row['test_dataset'],
            'recognizer': row['recognizer'],
            'modality': row['modality'],
            'face_detection': row['face_detection'],
        }
        
        for scenario, metrics in scenario_metrics.items():
            record = base_info.copy()
            record['scenario'] = scenario
            
            for metric_name, col_name in metrics.items():
                if col_name in row:
                    record[metric_name] = row[col_name]
                else:
                    record[metric_name] = None
            
            summary_records.append(record)
    
    summary_df = pd.DataFrame(summary_records)
    
    return summary_df

def calcular_threshold_por_bpp_y_condition(
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
    def _calcular_umbral(group_key: tuple, distancias: np.ndarray, far_list: list[float]) -> dict:
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
        delayed(_calcular_umbral)(group_key, dists, far_list)
        for group_key, dists in grupos_combinados.items()
    )
    
    # 5. Combinar los resultados
    resultados_finales = {}
    for res in resultados_paralelos:
        if res:
            resultados_finales.update(res)
        
    return resultados_finales

def calcular_threshold_por_bpp(
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
    def _calcular_umbral(bpp_value: int, distancias: np.ndarray, far_list: list[float]) -> dict:
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
        delayed(_calcular_umbral)(bpp, dists, far_list)
        for bpp, dists in grupos_bpp.items()
    )
    
    # 5. Combinar los resultados de los diccionarios
    resultados_finales = {}
    for res in resultados_paralelos:
        if res: # Evitar agregar resultados vacíos si hay un error general
            resultados_finales.update(res)
        
    return resultados_finales

def calcular_tar(df_genuine: pd.DataFrame, threshold: float) -> float:
    """
    Calcula la Tasa de Aceptación Verdadera (TAR) para un DataFrame de pares 'genuine'
    dado un umbral de distancia.
    
    Args:
        df_genuine (pd.DataFrame): DataFrame que contiene solo pares 'genuine'.
        threshold (float): El umbral de distancia (distancia <= threshold se considera 'aceptado').
        
    Returns:
        float: El TAR en porcentaje.
    """
    if df_genuine.empty:
        return 0.0
    
    # Contamos cuántas distancias genuinas son menores o iguales al umbral (Aceptación Verdadera)
    true_accepts = (df_genuine['distance'] <= threshold).sum()
    
    # Calculamos el TAR
    tar_rate = (true_accepts / len(df_genuine)) * 100.0
    return tar_rate

def plot_tar_vs_bpp(
    df_long: pd.DataFrame, 
    threshold_dict: Dict[str, Any], 
    dataset_name: str, 
    far_list: List[float]
):
    """
    Calcula el TAR para los umbrales dados y genera un gráfico por cada FAR, 
    mostrando TAR vs BPP para cada condición (OO, OW, WW).
    
    Args:
        df_long (pd.DataFrame): El DataFrame original con todos los datos.
        threshold_dict (Dict): El diccionario de umbrales calculado (e.g., {'CFD': {...}}).
        dataset_name (str): El nombre del dataset a graficar (e.g., 'CFD').
        far_list (List[float]): Lista de valores FAR para generar un plot por cada uno.
    """
    
    sns.set_style("whitegrid")
    
    # 1. Preparar el subconjunto de datos 'genuine' para el dataset objetivo
    # Solo necesitamos pares 'genuine' para calcular el TAR.
    df_genuine_full = df_long[
        (df_long['test_dataset'] == dataset_name) & 
        (df_long['pair_type'] == 'genuine')
    ].copy()

    if df_genuine_full.empty:
        print(f"Error: No se encontraron pares 'genuine' para el dataset {dataset_name}.")
        return

    # Extraer los resultados del dataset específico
    results_dataset = threshold_dict.get(dataset_name, {})
    
    if not results_dataset:
        print(f"Error: No se encontraron resultados de umbrales para el dataset {dataset_name}.")
        return

    # 2. Iterar por cada FAR y generar un gráfico
    for far_value in far_list:
        data_for_plot = []
        
        # 3. Iterar por cada combinación (BPP, Condition) en los resultados
        for (bpp, condition), thresholds in results_dataset.items():
            threshold = thresholds.get(far_value)
            
            # Verificar si el umbral fue calculado (o si fue 'No puedo calcular...')
            if isinstance(threshold, float):
                
                # Filtrar los pares 'genuine' para el BPP y Condition actual
                df_subset = df_genuine_full[
                    (df_genuine_full['bpp'] == bpp) & 
                    (df_genuine_full['condition'] == condition)
                ]
                
                # Calcular el TAR para este umbral
                tar = calcular_tar(df_subset, threshold)
                
                # Almacenar el resultado para el plot
                data_for_plot.append({
                    'BPP': bpp,
                    'Condition': condition,
                    'TAR': tar,
                    'Threshold': threshold
                })
        
        if not data_for_plot:
            print(f"Advertencia: No hay datos de TAR para FAR={far_value} en {dataset_name}. Omitiendo plot.")
            continue
            
        df_plot = pd.DataFrame(data_for_plot)
        
        # 4. Generar el gráfico
        far_percent = far_value * 100 if far_value <= 1.0 else far_value
        
        plt.figure(figsize=(8, 6))
        
        # Usamos BPP en el eje X (aseguramos orden con sort_values)
        # La librería seaborn mapea automáticamente 'Condition' a diferentes colores/marcadores
        ax = sns.lineplot(
            data=df_plot.sort_values(by='BPP'), 
            x='BPP', 
            y='TAR', 
            hue='Condition', 
            style='Condition', 
            markers=True, 
            dashes=False,
            palette=['tab:blue', 'tab:orange', 'tab:green'], # Colores que se asemejan a la imagen
            markersize=10,
            linewidth=2.5
        )
        
        # Replicar el formato del título: "TAR at FAR ~0.01% — ONOT_set1"
        ax.set_title(f"TAR at FAR ~{far_percent:.2g}% — {dataset_name}", fontsize=16)
        ax.set_xlabel("BPP (Bits per Pixel)", fontsize=12)
        ax.set_ylabel("TAR (%)", fontsize=12)
        
        # Formato de la leyenda para que se parezca a las etiquetas (OO -> Original-Original, etc.)
        legend_labels = {
            'OO': 'Original - Original', 
            'OW': 'Original - Watermark', 
            'WW': 'Watermark - Watermark'
        }
        
        handles, labels = ax.get_legend_handles_labels()
        # Asegurar que los labels de la leyenda se mapeen correctamente
        new_labels = [legend_labels.get(l, l) for l in labels]

        # Colocar la leyenda fuera del plot para no obstruir los datos
        plt.legend(handles=handles, labels=new_labels, title=None, 
                   loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize='large',
                   frameon=True, fancybox=True, shadow=True, borderpad=1) 
        
        # Asegurar que el eje X tenga ticks solo en los valores de BPP presentes
        bpp_ticks = sorted(df_plot['BPP'].unique())
        ax.set_xticks(bpp_ticks)
        
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Ajustar para la leyenda
        plt.show()

def plot_tar_vs_bpp_subplots(
    df_long: pd.DataFrame, 
    threshold_dict: Dict[str, Any], 
    dataset_name: str, 
    far_list: List[float]
):
    """
    Calcula el TAR y genera subplots en una fila única (por FAR), 
    compartiendo el eje Y y con una única leyenda externa.
    """
    
    sns.set_style("whitegrid")
    
    # 1. Preparar el subconjunto de datos 'genuine' para el dataset objetivo
    df_genuine_full = df_long[
        (df_long['test_dataset'] == dataset_name) & 
        (df_long['pair_type'] == 'genuine')
    ].copy()

    if df_genuine_full.empty:
        print(f"Error: No se encontraron pares 'genuine' para el dataset {dataset_name}.")
        return

    results_dataset = threshold_dict.get(dataset_name, {})
    if not results_dataset:
        print(f"Error: No se encontraron resultados de umbrales para el dataset {dataset_name}.")
        return

    # 2. ACUMULAR TODOS LOS DATOS en un solo DataFrame
    data_for_all_plots = []
    
    for far_value in far_list:
        far_label = f"TAR at FAR ~{far_value * 100:.2g}%"
        
        for (bpp, condition), thresholds in results_dataset.items():
            threshold = thresholds.get(far_value)
            
            if isinstance(threshold, float):
                df_subset = df_genuine_full[
                    (df_genuine_full['bpp'] == bpp) & 
                    (df_genuine_full['condition'] == condition)
                ]
                
                tar = calcular_tar(df_subset, threshold)
                
                data_for_all_plots.append({
                    'BPP': bpp,
                    'Condition': condition,
                    'TAR': tar,
                    'FAR_Label': far_label # Nueva columna para distinguir los subplots
                })

    if not data_for_all_plots:
        print(f"Advertencia: No hay datos de TAR para los FARs especificados en {dataset_name}. Omitiendo plot.")
        return
        
    df_plot_all = pd.DataFrame(data_for_all_plots)
    
    # 3. CREAR SUBPLOTS EN UNA FILA
    n_fars = len(far_list)
    fig, axes = plt.subplots(
        nrows=1, 
        ncols=n_fars, 
        sharey=False, # ¡Compartir el eje Y!
        figsize=(8 * n_fars, 6) # Ajustar el ancho
    )
    
    # Asegurarse de que 'axes' sea un array si solo hay 1 FAR
    if n_fars == 1:
        axes = [axes]
    
    # Mapeo de etiquetas
    legend_labels = {
        'OO': 'Original - Original', 
        'OW': 'Original - Watermark', 
        'WW': 'Watermark - Watermark'
    }
    
    # 4. Generar los plots dentro de los subplots
    for i, far_label in enumerate(df_plot_all['FAR_Label'].unique()):
        ax = axes[i]
        df_subset = df_plot_all[df_plot_all['FAR_Label'] == far_label]

        # Usamos BPP en el eje X, Condition para color/estilo
        sns.lineplot(
            data=df_subset.sort_values(by='BPP'), 
            x='BPP', 
            y='TAR', 
            hue='Condition', 
            style='Condition', 
            markers=True, 
            dashes=False,
            palette=['tab:blue', 'tab:orange', 'tab:green'], 
            markersize=10,
            linewidth=2.5,
            ax=ax,
            #legend='full' if i == 0 else False # Solo la leyenda en el primer plot
        )
        
        # Título y etiquetas
        ax.set_title(f"{far_label} — {dataset_name}", fontsize=14)
        ax.set_xlabel("BPP (Bits per Pixel)", fontsize=12)
        
        # Eliminar la etiqueta Y de los plots que no son el primero (por sharey=True)
        if i > 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("TAR (%)", fontsize=12)
            
        # Asegurar que el eje X tenga ticks solo en los valores de BPP presentes
        bpp_ticks = sorted(df_subset['BPP'].unique())
        ax.set_xticks(bpp_ticks)


    # 5. MANEJAR LA LEYENDA ÚNICA FUERA DEL PLOT
    # Tomamos la leyenda del primer subplot
    h, l = axes[0].get_legend_handles_labels()
    new_labels = [legend_labels.get(label, label) for label in l]
    
    # Eliminar las leyendas duplicadas si es que existen (Hue y Style son la misma cosa)
    if 'Condition' in new_labels:
        # Esto ocurre porque hue y style añaden 'Condition' como label al inicio
        h = h[1:]
        new_labels = new_labels[1:]


    #fig.legend(
    #    h, new_labels, 
    #    title=None, 
    #    loc='center right', 
    #    bbox_to_anchor=(1.03, 0.5), # Posición de la leyenda a la derecha
    #    fontsize='large',
    #    frameon=True, 
    #    fancybox=True, 
    #    shadow=True, 
    #    borderpad=1
    #)
    
    # Ajustar el layout para dar espacio a la leyenda
    plt.tight_layout(rect=[0, 0, 0.9, 1]) 
    plt.show()

def plot_distance_distributions(df, test_dataset):
    """
    Plot genuine and impostor distance distributions for each BPP and recognizer.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: watermark, recognizer, metric, train_dataset,
        test_dataset, bpp, condition (OO/OW/WW), pair_type (genuine/impostor),
        distance, experiment, mtcnn, mode, path.
    test_dataset : str
        Name of the test dataset to filter (e.g. 'CFD', 'SCface').

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Filter by test dataset
    df = df[df["test_dataset"] == test_dataset].copy()

    # Extract metadata from the (already single-valued) columns
    watermark_name = df["watermark"].iloc[0]
    train_ds = df["train_dataset"].iloc[0]
    mode = df["mode"].iloc[0]

    # Sort BPP values
    bpp_values = sorted(df["bpp"].unique())
    recognizers = sorted(df["recognizer"].unique())
    conditions = ["OO", "OW", "WW"]

    n_rows = len(bpp_values)
    n_cols = len(recognizers)

    # Color palette: one color per condition
    palette = {"OO": "#1f77b4", "OW": "#ff7f0e", "WW": "#2ca02c"}

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7 * n_cols, 3.2 * n_rows),
        sharex=True, sharey=False,
        constrained_layout=True,
    )

    # Ensure axes is always 2D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for col_idx, rec in enumerate(recognizers):
        for row_idx, bpp in enumerate(bpp_values):
            ax = axes[row_idx, col_idx]
            subset = df[(df["recognizer"] == rec) & (df["bpp"] == bpp)]

            for cond in conditions:
                for pair_type in ["genuine", "impostor"]:
                    data = subset[
                        (subset["condition"] == cond) & (subset["pair_type"] == pair_type)
                    ]["distance"]

                    if data.empty:
                        continue

                    ls = "-" if pair_type == "genuine" else "--"
                    label = f"{pair_type}, {cond}"

                    sns.kdeplot(
                        data,
                        ax=ax,
                        color=palette[cond],
                        linestyle=ls,
                        linewidth=1.4,
                        label=label,
                        warn_singular=False,
                    )

            ax.set_ylabel("Density", fontsize=9)
            ax.set_xlabel("Distance", fontsize=9)
            ax.set_title(f"{rec.capitalize()} — BPP={bpp}", fontsize=10, fontweight="bold")
            ax.tick_params(labelsize=8)
            ax.grid(True, linestyle="--", alpha=0.4)

    # Single shared legend at the top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Remove per-subplot legends
    for ax_row in axes:
        for ax in ax_row:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=len(conditions) * 2,
        fontsize=9,
        frameon=True,
        title="Pair, Condition",
        title_fontsize=9,
        bbox_to_anchor=(0.5, 1.04),
    )

    fig.suptitle(
        f"Genuine & Impostor Distance Distributions — {watermark_name} "
        f"(train: {train_ds}, test: {test_dataset}, {mode})",
        fontsize=13,
        fontweight="bold",
        y=1.07,
    )

    return fig

def plot_distance_distributions_multi_dataset(df, test_datasets, bpp=6, mode="online"):
    """
    Plot genuine and impostor distance distributions for Stegformer COCO
    at a fixed BPP across multiple test datasets.
 
    Each row corresponds to one test dataset. Each column corresponds to
    one FR system (ArcFace, FaceNet). All plots show OO, OW, and WW
    conditions with genuine (solid) and impostor (dashed) distributions.
 
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: watermark, recognizer, metric, train_dataset,
        test_dataset, bpp, condition (OO/OW/WW), pair_type (genuine/impostor),
        distance, experiment, mtcnn, mode, path.
    test_datasets : list of str
        List of test dataset names to include, one per row
        (e.g. ['CFD', 'ONOT_set1', 'LFW', 'SCface']).
    bpp : int, optional
        Bits per pixel to display. Default is 6.
    mode : str, optional
        Evaluation mode to filter ('online' or 'offline'). Default is 'online'.
 
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Filter: Stegformer, COCO, selected BPP and mode
    mask = (
        (df["watermark"].str.lower().str.contains("stegformer"))
        & (df["train_dataset"].str.lower().str.contains("coco"))
        & (df["bpp"] == bpp)
        & (df["mode"] == mode)
        & (df["test_dataset"].isin(test_datasets))
    )
    df_filtered = df[mask].copy()
 
    if df_filtered.empty:
        raise ValueError(
            f"No data found for Stegformer COCO at {bpp} bpp ({mode} mode) "
            f"with test datasets {test_datasets}. Check filter criteria."
        )
 
    recognizers = sorted(df_filtered["recognizer"].unique())
    conditions = ["OO", "OW", "WW"]
 
    n_rows = len(test_datasets)
    n_cols = len(recognizers)
 
    # Color palette per condition
    palette = {"OO": "#1f77b4", "OW": "#ff7f0e", "WW": "#2ca02c"}
 
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7 * n_cols, 3.2 * n_rows),
        sharex=False, sharey=False,
        constrained_layout=True,
    )
 
    # Ensure axes is always 2D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]
 
    for row_idx, test_ds in enumerate(test_datasets):
        for col_idx, rec in enumerate(recognizers):
            ax = axes[row_idx, col_idx]
            subset = df_filtered[
                (df_filtered["test_dataset"] == test_ds)
                & (df_filtered["recognizer"] == rec)
            ]
 
            for cond in conditions:
                for pair_type in ["genuine", "impostor"]:
                    data = subset[
                        (subset["condition"] == cond)
                        & (subset["pair_type"] == pair_type)
                    ]["distance"]
 
                    if data.empty:
                        continue
 
                    ls = "-" if pair_type == "genuine" else "--"
                    label = f"{pair_type.capitalize()}, {cond}"
 
                    sns.kdeplot(
                        data,
                        ax=ax,
                        color=palette[cond],
                        linestyle=ls,
                        linewidth=1.4,
                        label=label,
                        warn_singular=False,
                    )
 
            ax.set_ylabel("Density", fontsize=9)
            ax.set_xlabel("Cosine Distance", fontsize=9)
            ax.set_title(
                f"{test_ds} — {rec.capitalize()}",
                fontsize=10, fontweight="bold",
            )
            ax.tick_params(labelsize=8)
            ax.grid(True, linestyle="--", alpha=0.4)
 
    # Single shared legend at the top
    handles, labels = axes[0, 0].get_legend_handles_labels()
 
    # Remove per-subplot legends
    for ax_row in axes:
        for ax in ax_row:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
 
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=len(conditions) * 2,
        fontsize=9,
        frameon=True,
        title="Pair type, Condition",
        title_fontsize=9,
        bbox_to_anchor=(0.5, -0.05),
    )
 
    fig.suptitle(
        f"Genuine & Impostor Distance Distributions — Stegformer "
        f"(train: COCO, {bpp} bpp, {mode})",
        fontsize=13,
        fontweight="bold",
        y=1.05,
    )
 
    return fig

def compute_fr_distributions_stats(distances_df):
    
    def kl_from_kde(p_samples, q_samples, n_grid=500):
        """Estima KL(P||Q) via KDE sobre grilla compartida."""
        min_val = min(p_samples.min(), q_samples.min())
        max_val = max(p_samples.max(), q_samples.max())
        grid = np.linspace(min_val, max_val, n_grid)
        
        p_kde = gaussian_kde(p_samples)(grid)
        q_kde = gaussian_kde(q_samples)(grid)
        
        # Suavizado mínimo para evitar log(0)
        p_kde += 1e-10
        q_kde += 1e-10
        
        p_kde /= p_kde.sum()
        q_kde /= q_kde.sum()
        
        return entropy(p_kde, q_kde)

    def js_from_kde(p_samples, q_samples, n_grid=500):
        """Estimates JS(P||Q) via KDE on shared grid. Returns value in [0,1]."""
        min_val = min(p_samples.min(), q_samples.min())
        max_val = max(p_samples.max(), q_samples.max())
        grid = np.linspace(min_val, max_val, n_grid)
        dx = grid[1] - grid[0]
        
        p_kde = gaussian_kde(p_samples)(grid)
        q_kde = gaussian_kde(q_samples)(grid)
        
        p_kde += 1e-10
        q_kde += 1e-10

        p_prob = p_kde * dx
        q_prob = q_kde * dx
        
        #p_kde /= p_kde.sum()
        #q_kde /= q_kde.sum()

        p_prob /= p_prob.sum()
        q_prob /= q_prob.sum()
        
        # jensenshannon returns sqrt(JS) by default, square it to get JS
        return jensenshannon(p_kde, q_kde, base=2) ** 2
    
    group_keys = ['watermark', 'recognizer', 'train_dataset', 'test_dataset', 'bpp', 'mode']
    records = []
    
    for keys, grp in distances_df.groupby(group_keys):
        row = dict(zip(group_keys, keys))
        
        for pair in ['genuine', 'impostor']:
            oo = grp[(grp['condition'] == 'OO') & (grp['pair_type'] == pair)]['distance']
            ow = grp[(grp['condition'] == 'OW') & (grp['pair_type'] == pair)]['distance']
            ww = grp[(grp['condition'] == 'WW') & (grp['pair_type'] == pair)]['distance']
            
            if len(oo) < 10:
                continue
            
            for cond_name, cond_dist in [('OW', ow), ('WW', ww)]:
                if len(cond_dist) < 10:
                    continue

                pooled_std = np.sqrt((oo.std()**2 + cond_dist.std()**2) / 2)
                r = row.copy()
                r['pair_type'] = pair
                r['condition'] = cond_name
                r['kl_divergence'] = kl_from_kde(oo.values, cond_dist.values)
                r['delta_mean'] = cond_dist.mean() - oo.mean()
                r['wasserstein'] =  wasserstein_distance(oo.values, cond_dist.values)
                r['js_divergence']  = js_from_kde(oo.values, cond_dist.values)
                r['cohen_d']     = (cond_dist.mean() - oo.mean()) / (pooled_std + 1e-10)
                r['n_oo'] = len(oo)
                r['n_cond'] = len(cond_dist)
                records.append(r)
    
    return pd.DataFrame(records)

def join_fr_dist_stats_wm_metrics_by_condition_pair(df_dis_stats, df_wm_long, condition='WW', pair_type='genuine'):
    if condition == "both":
        df_dis_stats_join = df_dis_stats[
        (df_dis_stats['pair_type'] == pair_type)
        ][['watermark','train_dataset','test_dataset','bpp','mode','recognizer','condition',
           'kl_divergence','delta_mean','wasserstein','cohen_d','js_divergence']]
    
    else:
        df_dis_stats_join = df_dis_stats[
            (df_dis_stats['pair_type'] == pair_type) & 
            (df_dis_stats['condition'] == condition)
        ][['watermark','train_dataset','test_dataset','bpp','mode','recognizer', 'condition',
           'kl_divergence','delta_mean','wasserstein','cohen_d','js_divergence']]

    # Join
    df_merged = df_wm_long.merge(
        df_dis_stats_join,
        on=['watermark','train_dataset','test_dataset','bpp','mode'],
        how='inner'
    )

    #print(df_merged.shape)
    #df_merged[['watermark','recognizer','mode','train_dataset','test_dataset','bpp']].drop_duplicates().shape

    return df_merged

def normalize_col(series, invert=False, min_val=None, max_val=None):
    if min_val is not None and max_val is not None:
        norm = (series - min_val) / (max_val - min_val + 1e-10)
    else:
        mn, mx = series.min(), series.max()
        norm = (series - mn) / (mx - mn + 1e-10)
    return 1 - norm if invert else norm

def add_normalized_wasserstein(
    df_agg,
    distances_range,
    eps=1e-10,
    inverse_scope="global"
):
    """
    Adds two columns to df_agg:

    1. ws_norm:
       Wasserstein distance normalized by the OO genuine distance range
       for each watermark, bpp, recognizer and mode.

    2. ws_inv_norm:
       Inverted normalized Wasserstein score.
       ws_norm = 0 -> ws_inv_norm = 1
       largest ws_norm -> ws_inv_norm = 0

    Parameters
    ----------
    df_agg : pd.DataFrame
        Aggregated dataframe containing the column 'wasserstein'.

    distances_range : pd.DataFrame
        Dataframe containing min_distance and max_distance for OO genuine scores.

    eps : float
        Small constant to avoid division by zero.

    inverse_scope : str
        "global"     -> inverse normalization using max ws_norm over the full dataframe.
        "recognizer" -> inverse normalization separately per recognizer.
        "mode"       -> inverse normalization separately per recognizer and mode.

    Returns
    -------
    pd.DataFrame
        Copy of df_agg with added columns:
        - genuine_ref_range
        - ws_norm
        - ws_inv_norm
    """

    df = df_agg.copy()

    # 1. Keep only OO genuine ranges as reference
    ref = distances_range[
        (distances_range["condition"] == "OO") &
        (distances_range["pair_type"] == "genuine")
    ].copy()

    # 2. Compute reference range
    ref["genuine_ref_range"] = ref["max_distance"] - ref["min_distance"]

    # 3. Keep only the columns needed for the merge
    merge_keys = ["watermark", "bpp", "recognizer", "mode"]

    ref = ref[
        merge_keys + ["min_distance", "max_distance", "genuine_ref_range"]
    ].drop_duplicates(subset=merge_keys)

    # 4. Merge OO genuine range into df_agg
    df = df.merge(
        ref,
        on=merge_keys,
        how="left",
        validate="many_to_one"
    )

    # 5. Check missing references
    missing_ref = df["genuine_ref_range"].isna().sum()
    if missing_ref > 0:
        print(f"Warning: {missing_ref} rows did not find OO genuine reference range.")

    # 6. Normalize Wasserstein by the OO genuine range
    df["ws_norm"] = df["wasserstein"] / (df["genuine_ref_range"] + eps)

    # 7. Inverted normalization
    if inverse_scope == "global":
        max_ws = df["ws_norm"].max()
        df["ws_inv_norm"] = 1 - (df["ws_norm"] / (max_ws + eps))
    
    elif inverse_scope == "traditional":
        max_ws = 1.0
        df["ws_inv_norm"] = 1 - (df["ws_norm"] / (max_ws + eps))

    elif inverse_scope == "recognizer":
        df["ws_inv_norm"] = df.groupby("recognizer")["ws_norm"].transform(
            lambda s: 1 - (s / (s.max() + eps))
        )

    elif inverse_scope == "mode":
        df["ws_inv_norm"] = df.groupby(["recognizer", "mode"])["ws_norm"].transform(
            lambda s: 1 - (s / (s.max() + eps))
        )

    else:
        raise ValueError("inverse_scope must be 'global', 'recognizer', or 'mode'.")

    # 8. Clip numerical noise
    df["ws_inv_norm"] = df["ws_inv_norm"].clip(0, 1)

    return df

def aggregate_for_plots(df_merged):
    """
    Aggregates by watermark x bpp x recognizer x mode.
    Online and offline are kept separate for spider plot comparison.
    """
    df_agg = df_merged.groupby(['watermark', 'bpp', 'recognizer', 'mode', 'condition']).agg(
        accuracy=('accuracy', 'mean'),
        psnr=('psnr', 'mean'),
        ssim=('ssim', 'mean'),
        kl_divergence=('kl_divergence', 'mean'),
        wasserstein=('wasserstein', 'mean'),
        js_divergence=('js_divergence', 'mean'),
    ).reset_index()

    # Normalize globally across all rows so scales are comparable
    df_agg['acc_norm']  = normalize_col(df_agg['accuracy'])
    df_agg['psnr_norm'] = normalize_col(df_agg['psnr'])
    df_agg['kl_inv_norm'] = normalize_col(df_agg['kl_divergence'], invert=True, min_val=0, max_val=df_agg['kl_divergence'].max())
    #df_agg['ws_inv_norm'] = normalize_col(df_agg['wasserstein'],    invert=True, min_val=0, max_val=df_agg['wasserstein'].max())
    df_agg['js_inv_norm'] = 1 - df_agg['js_divergence']

    return df_agg

def plot_spider(df_agg, dist_metric='js_inv_norm', condition='OW', save_path='spider_trilema'):
    """
    4 spider plots: one per (watermark x recognizer) combination.
    Each spider shows 4 bpp x 2 modes (online/offline).
    Line style: solid=online, dashed=offline.
    Marker: circle=online, square=offline.
    Color: one per bpp.
    Axes:
        - Recovery BAR (%): bit accuracy rate displayed as percentage
        - Imperceptibility PSNR (dB): real dB values
        - FR Preservation: normalized [0,1], higher=better
    Grid levels 0.25, 0.50, 0.75:
        From axis-level intersection, draw dashed line perpendicular to axis
        in clockwise direction, length proportional to level.
        Value label placed at midpoint of that segment, rotated along it.
    Level 1.00: value placed directly at axis vertex.
    """
    df_agg = df_agg.copy()

    if condition != 'both':
        df_agg = df_agg[df_agg['condition'] == condition].copy()

    if df_agg.empty:
        raise ValueError(f"No hay datos para condition='{condition}'")
    
    metric_names = {
        'kl_inv_norm': 'FR Preservation\n(inv. KL Divergence)',
        'js_inv_norm': 'FR Preservation\n(inv. Jensen-Shannon)',
        'ws_inv_norm': 'FR Preservation\n(1 - normalized Wasserstein)',
    }
    fr_label = metric_names.get(dist_metric, f'FR Preservation\n({dist_metric})')
    categories = ['Recovery BAR\n(%)', 'Imperceptibility PSNR\n(dB)', fr_label]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    bpp_colors   = {1: '#2196F3', 3: '#4CAF50', 6: '#FF9800', 8: '#F44336'}
    mode_styles  = {'online': '-',  'offline': '--'}
    mode_markers = {'online': 'o',  'offline': '*'}

    watermarks  = sorted(df_agg['watermark'].unique())
    recognizers = sorted(df_agg['recognizer'].unique())

    psnr_min = df_agg['psnr'].min()
    psnr_max = df_agg['psnr'].max()
    acc_min  = df_agg['accuracy'].min()
    acc_max  = df_agg['accuracy'].max()
    ws_norm_max = df_agg['ws_norm'].max() if dist_metric == 'ws_inv_norm' else None
    ws_norm_min = df_agg['ws_norm'].min() if dist_metric == 'ws_inv_norm' else None

    grid_levels  = [0.25, 0.50, 0.75]
    chord_factor = 0.35  # length of perpendicular segment = level * chord_factor

    def get_label(axis_idx, level):
        """Returns real-value label for a given axis and normalized level."""
        if axis_idx == 0:
            val = (acc_min + level * (acc_max - acc_min)) * 100
            return f"{val:.0f}%"

        elif axis_idx == 1:
            val = psnr_min + level * (psnr_max - psnr_min)
            return f"{val:.1f}dB"

        else:
            if dist_metric == 'ws_inv_norm':
                # level = 1 means minimum Wasserstein change
                # level = 0 means maximum Wasserstein change
                ws_at_level = ws_norm_max - level * (ws_norm_max - ws_norm_min)
                preservation = (1 - ws_at_level) * 100
                return f"{preservation:.0f}%"

            else:
                return f"{level * 100:.0f}%"

    fig, axes = plt.subplots(
        len(recognizers), len(watermarks),
        figsize=(7 * len(watermarks), 6 * len(recognizers)),
        subplot_kw=dict(polar=True)
    )

    if len(recognizers) == 1 and len(watermarks) == 1:
        axes = np.array([[axes]])
    elif len(recognizers) == 1:
        axes = axes.reshape(1, -1)
    elif len(watermarks) == 1:
        axes = axes.reshape(-1, 1)

    for r_i, rec in enumerate(recognizers):
        for w_i, wm in enumerate(watermarks):
            ax  = axes[r_i, w_i]
            sub = df_agg[(df_agg['recognizer'] == rec) & (df_agg['watermark'] == wm)]

            # Concentric circles — more visible
            ax.set_yticklabels([])
            ax.set_yticks(grid_levels + [1.0])
            ax.grid(True, color='gray', linestyle=':', linewidth=1.0, alpha=0.6)

            # Perpendicular segments per axis per level
            for a_i, angle in enumerate(angles):
                for level in grid_levels:
                    # Start point P
                    px = level * np.cos(angle)
                    py = level * np.sin(angle)

                    # Clockwise perpendicular direction
                    dx = np.sin(angle)
                    dy = -np.cos(angle)

                    # Length to reach outer circle (r=1.0)
                    t = np.sqrt(1 - level**2)

                    # End point on outer circle
                    ex = px + t * dx
                    ey = py + t * dy

                    # Convert to polar
                    r_start = level
                    t_start = angle
                    r_end   = np.sqrt(ex**2 + ey**2)
                    t_end   = np.arctan2(ey, ex)

                    ax.plot([t_start, t_end], [r_start, r_end],
                            color='dimgray', linestyle='--',
                            linewidth=1.0, alpha=0.75)

                    # Midpoint label
                    mx = (px + ex) / 2
                    my = (py + ey) / 2
                    mr = np.sqrt(mx**2 + my**2)
                    mt = np.arctan2(my, mx)

                    chord_deg = np.degrees(np.arctan2(dy, dx))
                    if 90 < chord_deg % 360 < 270:
                        chord_deg += 180

                    ax.text(mt, mr, get_label(a_i, level),
                            ha='center', va='bottom',
                            fontsize=5.5, color='#222222',
                            rotation=chord_deg,
                            rotation_mode='anchor',
                            bbox=dict(boxstyle='round,pad=0.1',
                                    facecolor='white',
                                    edgecolor='none',
                                    alpha=0.8))

                # Level 1.0 at vertex
                ax.text(angle, 1.08, get_label(a_i, 1.0),
                        ha='center', va='bottom', fontsize=5.5, color='#222222',
                        bbox=dict(boxstyle='round,pad=0.1',
                                facecolor='white', edgecolor='none', alpha=0.8))

            # Data lines
            plot_angles = angles + [angles[0]]
            for _, row in sub.iterrows():
                values = [row['acc_norm'], row['psnr_norm'], row[dist_metric]]
                values += values[:1]

                ax.plot(
                    plot_angles, values,
                    linestyle=mode_styles[row['mode']],
                    color=bpp_colors[row['bpp']],
                    linewidth=1.8,
                    marker=mode_markers[row['mode']],
                    markersize=6,
                    alpha=0.85
                )
                ax.fill(plot_angles, values,
                        color=bpp_colors[row['bpp']],
                        alpha=0.04)

            ax.set_xticks(angles)
            ax.set_xticklabels(categories, fontsize=9)
            ax.tick_params(axis='x', pad=21)
            ax.set_ylim(0, 1.15)
            ax.set_title(f'{wm.capitalize()} — {rec.capitalize()}',
                         fontsize=11, fontweight='bold', pad=20)

    # Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_elements = []
    for bpp, color in bpp_colors.items():
        legend_elements.append(
            Patch(facecolor=color, edgecolor='k', linewidth=0.5, label=f'{bpp} bpp')
        )
    legend_elements.append(
        Line2D([0], [0], color='gray', linestyle='-',
               marker='o', markersize=6, label='Online')
    )
    legend_elements.append(
        Line2D([0], [0], color='gray', linestyle='--',
               marker='*', markersize=6, label='Offline')
    )

    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=len(bpp_colors) + 2,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.03),
        framealpha=0.9,
        title='BPP and Mode',
        title_fontsize=9
    )

    fig.suptitle(
    f'Watermarking Trade-off Trilema — Condition: {condition}\n'
    '(normalized [0,1] internally — axis labels show real values)',
    fontsize=13, fontweight='bold', y=1.01
    )

    plt.tight_layout()
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{save_path}.png', bbox_inches='tight', dpi=300)
    plt.show()

# ============================================================
# PLOT 2: PARETO 2D
# ============================================================

def plot_pareto_2d(df_merged,
                   fr_metric='kl_divergence',
                   condition='together',
                   psnr_threshold=None,
                   save_path='pareto_2d'):
    """
    Pareto frontier plot: PSNR (imperceptibility) vs ACC (recovery).
    Color encodes FR impact metric (log scale).
    Size encodes |delta_mean| of genuine pairs.

    Parameters
    ----------
    df_merged       : DataFrame with columns watermark, recognizer, psnr,
                      accuracy, delta_mean, condition, and fr_metric column.
    fr_metric       : FR impact column to use for color encoding.
                      e.g. 'kl_divergence', 'js_divergence', 'wasserstein'.
    condition       : 'together' → one row, all conditions plotted together.
                      'split'    → two rows: OW on top, WW on bottom.
    psnr_threshold  : float or None. If given, shades PSNR zones.
    save_path       : output filename (no extension).
    """

    metric_labels = {
        'kl_divergence': 'log₁₀(KL Divergence)',
        'js_divergence': 'log₁₀(JS Divergence)',
        'wasserstein':   'log₁₀(Wasserstein)',
    }
    fr_label    = metric_labels.get(fr_metric, f'log₁₀({fr_metric})')
    wm_markers  = {'stegformer': 'o', 'stegaformer': '^'}
    recognizers = sorted(df_merged['recognizer'].unique())

    # Row structure
    row_conditions = ['OW', 'WW'] if condition == 'split' else ['together']
    n_rows = len(row_conditions)
    n_cols = len(recognizers)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(7 * n_cols, 5 * n_rows),
                              squeeze=False)

    # Global size scaling
    abs_dm      = df_merged['delta_mean'].abs()
    size_scaled = 40 + 200 * (abs_dm - abs_dm.min()) / (abs_dm.max() - abs_dm.min() + 1e-10)

    # Global color scale
    log_metric  = np.log10(df_merged[fr_metric] + 1e-6)
    global_vmin = log_metric.min()
    global_vmax = log_metric.max()

    for r_i, cond in enumerate(row_conditions):
        for c_i, rec in enumerate(recognizers):
            ax = axes[r_i, c_i]

            # Filter
            mask = df_merged['recognizer'] == rec
            if cond != 'together':
                mask &= df_merged['condition'] == cond
            sub = df_merged[mask].copy()

            sub['fr_log'] = np.log10(sub[fr_metric] + 1e-6)
            sub['size']   = size_scaled[sub.index]

            # PSNR threshold zones
            if psnr_threshold is not None:
                psnr_min = sub['psnr'].min() - 2
                psnr_max = sub['psnr'].max() + 2
                ax.axvspan(psnr_min, psnr_threshold,
                           color='#ffcccc', alpha=0.35, zorder=0)
                ax.axvspan(psnr_threshold, psnr_max,
                           color='#ccffcc', alpha=0.35, zorder=0)
                ax.axvline(psnr_threshold, color='gray',
                           linestyle='--', linewidth=1.0, alpha=0.7)

            # Scatter per watermark
            sc = None
            for wm in sorted(sub['watermark'].unique()):
                wsub = sub[sub['watermark'] == wm]
                sc = ax.scatter(
                    wsub['psnr'], wsub['accuracy'],
                    c=wsub['fr_log'],
                    cmap='RdYlGn_r',
                    vmin=global_vmin, vmax=global_vmax,
                    s=wsub['size'],
                    marker=wm_markers[wm],
                    alpha=0.85,
                    edgecolors='k', linewidths=0.5,
                    zorder=2
                )
                for _, row in wsub.iterrows():
                    ax.annotate(
                        f"{int(row['bpp'])}",
                        (row['psnr'], row['accuracy']),
                        fontsize=6, alpha=0.75,
                        xytext=(3, 3), textcoords='offset points'
                    )

            row_label = 'All Conditions' if cond == 'together' else cond
            ax.set_title(f'{rec.capitalize()} — {row_label}',
                         fontsize=11, fontweight='bold')
            ax.set_xlabel('PSNR — Imperceptibility (dB)', fontsize=9)
            ax.set_ylabel('BAR — Message Recovery Rate', fontsize=9)
            ax.grid(True, alpha=0.3, zorder=1)
            ax.set_xlim(sub['psnr'].min() - 1, sub['psnr'].max() + 1)
            ax.text(0.02, 0.02, 'Size = |Δmean FR (genuine)|',
                    transform=ax.transAxes, fontsize=7, alpha=0.6)

            if sc is not None:
                cbar = plt.colorbar(sc, ax=ax, pad=0.02)
                cbar.set_label(fr_label, fontsize=8)

    # Shared legend — algorithms only, black markers, same size
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_elements = [
        Line2D([0], [0], marker=wm_markers[wm], color='k',
               markerfacecolor='k', markersize=7,
               linestyle='None', label=wm.capitalize())
        for wm in sorted(df_merged['watermark'].unique())
    ]
    if psnr_threshold is not None:
        legend_elements += [
            Patch(facecolor='#ffcccc', alpha=0.6,
                  label=f'PSNR < {psnr_threshold} dB — FR affected'),
            Patch(facecolor='#ccffcc', alpha=0.6,
                  label=f'PSNR ≥ {psnr_threshold} dB — FR safe'),
        ]

    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=len(legend_elements),
        fontsize=9,
        bbox_to_anchor=(0.5, -0.08),
        framealpha=0.9,
        title='Watermarking Algorithm',
        title_fontsize=9
    )

    cond_title = 'OW vs WW — Split View' if condition == 'split' else 'All Conditions Together'
    fig.suptitle(
        f'Pareto Frontier: PSNR vs Recovery | Color = {fr_label} | Size = |Δmean FR|\n'
        f'{cond_title} — Genuine Pairs',
        fontsize=12, fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{save_path}.png', bbox_inches='tight', dpi=300)
    plt.show()

def plot_pareto_2d_stacked(df_merged,
                           fr_metric='wasserstein',
                           average_by=None,
                           psnr_threshold=None,
                           save_path='pareto_2d_stacked'):
    """
    Pareto frontier plot: PSNR (x) vs BAR (y), stacked vertically by FR model.
    Color encodes FR impact metric (log scale).
    Size encodes |delta_mean| of genuine pairs.
 
    Parameters
    ----------
    df_merged       : DataFrame with columns watermark, recognizer, psnr,
                      accuracy, delta_mean, condition, train_dataset,
                      test_dataset, bpp, and fr_metric column.
    fr_metric       : FR impact column for color encoding.
                      e.g. 'kl_divergence', 'js_divergence', 'wasserstein'.
    average_by      : list of columns to average over, or None for no averaging.
                      e.g. ['train_dataset', 'condition'] to average over both.
                      e.g. ['train_dataset'] to average only over train domain.
                      e.g. None to show all individual points.
    psnr_threshold  : float or None. If given, shades PSNR zones.
    save_path       : output filename (no extension).
    """
 
    metric_labels = {
        'kl_divergence': 'log\u2081\u2080(KL Divergence)',
        'js_divergence': 'log\u2081\u2080(JS Divergence)',
        'wasserstein':   'log\u2081\u2080(Wasserstein)',
    }
    fr_label   = metric_labels.get(fr_metric, f'log\u2081\u2080({fr_metric})')
    wm_markers = {'stegformer': 'o', 'stegaformer': '^'}
 
    # Define grouping columns (always group by these)
    base_group = ['watermark', 'recognizer', 'test_dataset', 'bpp']
 
    if average_by is not None:
        # Remove the columns we want to average over from the grouping
        group_cols = [c for c in base_group if c not in average_by]
        # Add any remaining categorical columns that should stay
        agg_dict = {
            'psnr': 'mean',
            'accuracy': 'mean',
            'delta_mean': 'mean',
            fr_metric: 'mean',
        }
        # Also average ssim if present
        if 'ssim' in df_merged.columns:
            agg_dict['ssim'] = 'mean'
 
        df_plot = df_merged.groupby(group_cols, as_index=False).agg(agg_dict)
    else:
        df_plot = df_merged.copy()
 
    recognizers = sorted(df_plot['recognizer'].unique())
    n_rows = len(recognizers)
 
    fig, axes = plt.subplots(n_rows, 1,
                              figsize=(9, 5.5 * n_rows),
                              squeeze=False)
 
    # Global size scaling
    abs_dm      = df_plot['delta_mean'].abs()
    size_scaled = 40 + 200 * (abs_dm - abs_dm.min()) / (abs_dm.max() - abs_dm.min() + 1e-10)
    df_plot['size'] = size_scaled
 
    # Global color scale
    log_metric  = np.log10(df_plot[fr_metric] + 1e-6)
    global_vmin = log_metric.min()
    global_vmax = log_metric.max()
 
    for r_i, rec in enumerate(recognizers):
        ax = axes[r_i, 0]
        sub = df_plot[df_plot['recognizer'] == rec].copy()
        sub['fr_log'] = np.log10(sub[fr_metric] + 1e-6)
 
        # PSNR threshold zones
        if psnr_threshold is not None:
            psnr_min = sub['psnr'].min() - 2
            psnr_max = sub['psnr'].max() + 2
            ax.axvspan(psnr_min, psnr_threshold,
                       color='#ffcccc', alpha=0.35, zorder=0)
            ax.axvspan(psnr_threshold, psnr_max,
                       color='#ccffcc', alpha=0.35, zorder=0)
            ax.axvline(psnr_threshold, color='gray',
                       linestyle='--', linewidth=1.0, alpha=0.7)
 
        # Scatter per watermark
        sc = None
        for wm in sorted(sub['watermark'].unique()):
            wsub = sub[sub['watermark'] == wm]
            sc = ax.scatter(
                wsub['psnr'], wsub['accuracy'],
                c=wsub['fr_log'],
                cmap='RdYlGn_r',
                vmin=global_vmin, vmax=global_vmax,
                s=wsub['size'],
                marker=wm_markers[wm],
                alpha=0.85,
                edgecolors='k', linewidths=0.5,
                zorder=2
            )
 
            # Labels: bpp + dataset
            # Track already placed labels to avoid duplicates when averaged
            # Label offset per dataset to avoid overlapping
            label_offsets = {
                'facelab_london': (4, 6),    # above
                'CFD':            (4, -10),  # below
                'ONOT_set1':      (4, 6),    # above
                'SCface':         (4, -10),  # below
            }

            # Short display names for labels
            short_names = {
                'facelab_london': 'FaceLab',
                'CFD':            'CFD',
                'ONOT_set1':      'ONOT',
                'SCface':         'SCface',
            }

            placed_labels = set()
            for _, row in wsub.iterrows():
                display_name = short_names.get(row['test_dataset'], row['test_dataset'])
                label_text = f"{int(row['bpp'])} {display_name}"
                label_key = (round(row['psnr'], 1), round(row['accuracy'], 3), label_text)
                if label_key not in placed_labels:
                    offset = label_offsets.get(row['test_dataset'], (4, 4))
                    ax.annotate(
                        label_text,
                        (row['psnr'], row['accuracy']),
                        fontsize=5.5, alpha=0.7,
                        xytext=offset, textcoords='offset points'
                    )
                    placed_labels.add(label_key)
 
        ax.set_title(f'{rec.capitalize()}',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('PSNR — Imperceptibility (dB)', fontsize=11)
        ax.set_ylabel('BAR — Message Recovery Rate', fontsize=11)
        ax.grid(True, alpha=0.3, zorder=1)
        ax.set_xlim(sub['psnr'].min() - 1, sub['psnr'].max() + 1)
        ax.text(0.02, 0.02, 'Size = |Δmean FR (genuine)|',
                transform=ax.transAxes, fontsize=7, alpha=0.6)
 
        if sc is not None:
            cbar = plt.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label(fr_label, fontsize=11)
            cbar.ax.tick_params(labelsize=9)
 
    # Shared legend at bottom
    legend_elements = [
        Line2D([0], [0], marker=wm_markers[wm], color='k',
               markerfacecolor='k', markersize=7,
               linestyle='None', label=wm.capitalize())
        for wm in sorted(df_plot['watermark'].unique())
    ]
    if psnr_threshold is not None:
        legend_elements += [
            Patch(facecolor='#ffcccc', alpha=0.6,
                  label=f'PSNR < {psnr_threshold} dB — FR affected'),
            Patch(facecolor='#ccffcc', alpha=0.6,
                  label=f'PSNR ≥ {psnr_threshold} dB — FR safe'),
        ]
 
    avg_label = f"Averaged over: {', '.join(average_by)}" if average_by else "OW/WW"
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=len(legend_elements),
        fontsize=9,
        bbox_to_anchor=(0.5, -0.07),
        framealpha=0.9,
        title='Watermarking Algorithm',
        title_fontsize=9
    )
 
    fig.suptitle(
        f'Pareto Frontier: PSNR vs Recovery | Color = {fr_label} | Size = |Δmean FR|\n'
        f'Offline Mode — Genuine Pairs — {avg_label}',
        fontsize=12, fontweight='bold'
    )
 
    plt.tight_layout()
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{save_path}.png', bbox_inches='tight', dpi=300)
    plt.show()

def plot_fr_dist_by_train_domain(df_merged, fr_dist_metric='kl_divergence', condition='both', save_path='kl_by_train_domain'):
    train_colors = {'coco': '#E91E63', 'celeba_hq': '#9C27B0'}
    mode_styles  = {'online': '', 'offline': '//'}  # hatch
    recognizers  = df_merged['recognizer'].unique()

    fig, axes = plt.subplots(len(recognizers), 2, figsize=(16, 5 * len(recognizers)),
                              sharey=False)
    if len(recognizers) == 1:
        axes = axes.reshape(1, -1)

    for row_i, rec in enumerate(recognizers):
        for col_i, mode in enumerate(['online', 'offline']):
            ax = axes[row_i, col_i]

            if condition != 'both':
                sub = df_merged[
                    (df_merged['recognizer'] == rec) &
                    (df_merged['mode'] == mode) &
                    (df_merged['condition'] == condition)
                ].groupby(['watermark', 'bpp', 'train_dataset'])[fr_dist_metric].mean().reset_index()                
            else:
                sub = df_merged[
                    (df_merged['recognizer'] == rec) &
                    (df_merged['mode'] == mode)
                ].groupby(['watermark', 'bpp', 'train_dataset'])[fr_dist_metric].mean().reset_index()

            watermarks = sorted(sub['watermark'].unique())
            bpps       = sorted(sub['bpp'].unique())
            trains     = sorted(sub['train_dataset'].unique())

            # Posiciones: un grupo por watermark×bpp
            group_labels = [f"{wm}\n{b}bpp" for wm in watermarks for b in bpps]
            n_groups  = len(group_labels)
            n_trains  = len(trains)
            width     = 0.35
            x         = np.arange(n_groups)

            for t_i, train in enumerate(trains):
                values = []
                for wm in watermarks:
                    for b in bpps:
                        mask = (sub['watermark'] == wm) & \
                               (sub['bpp'] == b) & \
                               (sub['train_dataset'] == train)
                        val = sub[mask][fr_dist_metric].values
                        values.append(val[0] if len(val) > 0 else 0)

                offset = (t_i - n_trains / 2 + 0.5) * width
                bars = ax.bar(
                    x + offset, values,
                    width=width,
                    color=train_colors[train],
                    edgecolor='k', linewidth=0.5,
                    alpha=0.85, label=train
                )

            ax.set_xticks(x)
            ax.set_xticklabels(group_labels, fontsize=7, rotation=45, ha='right')
            ax.set_ylabel(f'{fr_dist_metric} - in genuine {condition} vs OO', fontsize=9)
            ax.set_title(f'{rec.capitalize()} — {mode}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, title='Train domain')
            ax.grid(True, axis='y', alpha=0.3)
            ax.set_yscale('log')  # log porque stegformer domina la escala

    fig.suptitle('Training domain impact on FR Preservation\n'
                 f'({fr_dist_metric} genuine {condition} vs OO — escala log)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{save_path}.png', bbox_inches='tight', dpi=300)
    plt.show()


def compute_V(df_merged, w_acc=0.35, w_psnr=0.30, w_kl=0.35):
    """Calcula V por fila y devuelve df_merged con columna V + df agregado."""
    df = df_merged.copy()

    # Normalizar globalmente
    def norm(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-10)

    df['acc_norm']  = norm(df['accuracy'])
    df['psnr_norm'] = norm(df['psnr'])
    df['kl_norm']   = norm(df['kl_divergence'])

    df['V'] = w_acc * df['acc_norm'] + w_psnr * df['psnr_norm'] - w_kl * df['kl_norm']

    # Agregar por watermark × bpp × recognizer
    df_V = df.groupby(['watermark', 'bpp', 'recognizer', 'mode']).agg(
        V=('V', 'mean'),
        accuracy=('accuracy', 'mean'),
        psnr=('psnr', 'mean'),
        kl_divergence=('kl_divergence', 'mean'),
        delta_mean=('delta_mean', 'mean')
    ).reset_index()

    return df, df_V


def plot_optimal_V(df_V, save_path='optimal_V'):
    recognizers = df_V['recognizer'].unique()
    fig, axes = plt.subplots(1, len(recognizers), figsize=(16, 5), sharey=True)
    if len(recognizers) == 1:
        axes = [axes]

    colors = {'online': '#2196F3', 'offline': '#FF9800'}

    for ax, rec in zip(axes, recognizers):
        sub = df_V[df_V['recognizer'] == rec].copy()
        sub['label'] = sub['watermark'] + '\n' + sub['bpp'].astype(str) + 'bpp'
        sub = sub.sort_values('V', ascending=False)

        bars = ax.bar(
            range(len(sub)),
            sub['V'],
            color=[colors[m] for m in sub['mode']],
            edgecolor='k', linewidth=0.5, alpha=0.85
        )

        # Anotar V encima de cada barra
        for bar, v in zip(bars, sub['V']):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7)

        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub['label'], fontsize=7, rotation=45, ha='right')
        ax.set_ylabel('Índice V (mayor = mejor)', fontsize=10)
        ax.set_title(rec.capitalize(), fontsize=12, fontweight='bold')
        ax.axhline(sub['V'].max(), color='red', linestyle='--',
                   linewidth=1, alpha=0.5, label='óptimo')
        ax.grid(True, axis='y', alpha=0.3)

    # Leyenda de mode
    from matplotlib.patches import Patch
    legend_els = [Patch(color=c, label=m) for m, c in colors.items()]
    fig.legend(handles=legend_els, loc='lower center', ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Índice de Viabilidad V = 0.35·ACC + 0.30·PSNR − 0.35·KL',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{save_path}.png', bbox_inches='tight', dpi=300)
    plt.show()