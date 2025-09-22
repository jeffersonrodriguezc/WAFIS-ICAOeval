# visualize_auth_plots.py
# Multiplot vertical con KDE: una figura por dataset, N filas (una por bpp) y 1 columna.
# Cada subplot: baseline (genuino+impostor) + watermarked del bpp correspondiente (genuino+impostor).
# Estilos:
#   --style lines (default): baseline sólida, wm discontinua
#   --style fill: wm con relleno translúcido + borde punteado
#
# Opciones KDE:
#   --kde_points N  (puntos de la malla, default 512)  [--bins actúa como alias]
#   --bw_scale S    (escala del ancho de banda de Silverman, default 1.0)

import argparse
import re
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- Rutas ----------------

def repo_root() -> Path:
    """Devuelve la raíz del repo asumiendo que este archivo está en <repo>/utils/."""
    return Path(__file__).resolve().parent.parent


# ---------------- Lectura de datos ----------------

def parse_bpp(experiment_name: str) -> str:
    """Extrae el BPP del nombre de la carpeta del experimento (string
    después del primer '_'). Ej: '1_2_255_w16_learn_im' -> '2'."""
    parts = experiment_name.split("_")
    if len(parts) >= 2:
        return parts[1]
    m = re.search(r"(\d+)", experiment_name)
    return m.group(1) if m else experiment_name


def _read_distance_column(fpath: Path) -> np.ndarray:
    """
    Lee un CSV que puede tener encabezado, diferentes separadores o varias columnas,
    y devuelve un vector 1D de floats (NaNs removidos).
    """
    df = pd.read_csv(fpath, sep=None, engine='python')  # infiere separador
    if df.shape[1] == 1:
        s = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        return s.dropna().to_numpy(dtype=float)

    preferred = [c for c in df.columns if str(c).strip().lower() in {"distance", "dist", "score"}]
    if preferred:
        s = pd.to_numeric(df[preferred[0]], errors='coerce')
        return s.dropna().to_numpy(dtype=float)

    best = None
    best_count = -1
    for c in df.columns:
        s = pd.to_numeric(df[c], errors='coerce')
        count = s.notna().sum()
        if count > best_count:
            best_count = count
            best = s
    return best.dropna().to_numpy(dtype=float)


def load_distances(exp_path: Path, dataset: str, metric: str, FRModel: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Carga los cuatro CSVs de distancias para un experimento+dataset+métrica."""
    files = {
        ('genuine', 'baseline'):     f"{metric}_genuine_distances_baseline.csv",
        ('genuine', 'watermarked'):  f"{metric}_genuine_distances_watermarked.csv",
        #('genuine', 'watermarked_both'):  f"{metric}_genuine_distances_watermarked_both.csv",
        ('impostor', 'baseline'):    f"{metric}_impostor_distances_baseline.csv",
        ('impostor', 'watermarked'): f"{metric}_impostor_distances_watermarked.csv",
        #('impostor', 'watermarked_both'): f"{metric}_impostor_distances_watermarked_both.csv",
    }
    dataset_dir = Path(exp_path) / dataset / FRModel / 'distances'
    print(f"Loading distances from {dataset_dir}")
    out = {'genuine': {}, 'impostor': {}}

    for (kind, state), fname in files.items():
        fpath = dataset_dir / fname
        if fpath.is_file():
            try:
                out[kind][state] = _read_distance_column(fpath)
            except Exception as e:
                print(f"WARNING: Couldn't read {fpath}: {e}")
    return out


def discover_experiments(root: Path, algorithm: str) -> List[Path]:
    """Lista carpetas de experimento bajo <root>/<algorithm> (excluye 'plots')."""
    algo_dir = root / algorithm
    if not algo_dir.is_dir():
        raise FileNotFoundError(f"Algorithm dir not found: {algo_dir}")
    return [p for p in algo_dir.iterdir() if p.is_dir()]


# ---------------- KDE ----------------

def _robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return 0.0
    std = np.std(x, ddof=1)
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    iqr_sigma = iqr / 1.34 if iqr > 0 else 0.0
    if std > 0 and iqr_sigma > 0:
        return min(std, iqr_sigma)
    return std if std > 0 else iqr_sigma


def _silverman_bandwidth(x: np.ndarray) -> float:
    n = x.size
    if n < 2:
        return 0.0
    sigma = _robust_sigma(x)
    if sigma <= 0:
        # fallback: usa rango/1.349
        data_range = float(np.max(x) - np.min(x)) if n > 1 else 1.0
        sigma = max(data_range / 1.349, 1e-6)
    h = 1.06 * sigma * n ** (-1/5)
    return max(h, 1e-6)


def kde_eval(x: np.ndarray, grid: np.ndarray, bw: Optional[float] = None, bw_scale: float = 1.0) -> np.ndarray:
    """
    KDE Gaussiana 1D. Devuelve densidad evaluada en 'grid'.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return np.zeros_like(grid, dtype=float)

    h = _silverman_bandwidth(x) if bw is None else bw
    h = max(h * float(bw_scale), 1e-9)

    # (G x N) broadcasting: cuidado si N es enorme
    z = (grid[:, None] - x[None, :]) / h
    # kernel gaussiano
    dens = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    dens = dens.sum(axis=1) / (n * h)
    return dens


# ---------------- Plotting (vertical multipanel) ----------------

def _pick_baseline(per_experiment_data: List[Tuple[str, Dict[str, Dict[str, np.ndarray]]]]):
    """Devuelve (base_genuine, base_impostor) tomando la primera disponible."""
    base_genuine = None
    base_impostor = None
    for _, data in per_experiment_data:
        if base_genuine is None and 'baseline' in data.get('genuine', {}):
            base_genuine = data['genuine']['baseline']
        if base_impostor is None and 'baseline' in data.get('impostor', {}):
            base_impostor = data['impostor']['baseline']
        if base_genuine is not None and base_impostor is not None:
            break
    return base_genuine, base_impostor


def _global_xlim(base_genuine, base_impostor, per_experiment_data):
    all_vals = []
    if base_genuine is not None and len(base_genuine) > 0:
        all_vals.append(base_genuine)
    if base_impostor is not None and len(base_impostor) > 0:
        all_vals.append(base_impostor)
    for _, data in per_experiment_data:
        for arr in (data.get('genuine', {}).get('watermarked', np.array([])),
                    data.get('impostor', {}).get('watermarked', np.array([]))):
            if arr is not None and len(arr) > 0:
                all_vals.append(arr)
    if all_vals:
        xmin = min(float(np.min(a)) for a in all_vals)
        xmax = max(float(np.max(a)) for a in all_vals)
        if xmax <= xmin:
            xmax = xmin + 1e-6
        pad = 0.02 * (xmax - xmin)
        return xmin - pad, xmax + pad
    else:
        return 0.0, 1.0


def plot_dataset_vertical_kde(
    dataset: str,
    per_experiment_data: List[Tuple[str, Dict[str, Dict[str, np.ndarray]]]],
    kde_points: int = 512,
    bw_scale: float = 1.0,
    metric: str = "cosine",
    style: str = "lines"
):
    """
    Crea una figura vertical (1 columna, N filas=bpp) para el dataset dado.
    Cada subplot incluye baseline (KDE genuino+impostor) y SOLO las KDE WM del bpp.
    Comparte ejes X e Y entre todas las filas y usa un bandwidth común para todas las curvas.
    """
    # Ordenar por bpp
    def _try_float(b):
        try:
            return float(b)
        except Exception:
            return float("inf")
    per_experiment_data = sorted(per_experiment_data, key=lambda t: _try_float(t[0]))

    N = len(per_experiment_data)
    if N == 0:
        return None

    # Colores por clase
    color_genuine = "#1f77b4"
    color_impostor = "#d62728"
    lw_all = 1.5

    # Baseline y grid X común
    base_genuine, base_impostor = _pick_baseline(per_experiment_data)
    xmin, xmax = _global_xlim(base_genuine, base_impostor, per_experiment_data)
    grid = np.linspace(xmin, xmax, int(kde_points))

    # ---------- Bandwidth común (mismo para TODAS las curvas) ----------
    bw_data = []
    if base_genuine is not None and len(base_genuine) > 1:
        bw_data.append(np.asarray(base_genuine, dtype=float))
    if base_impostor is not None and len(base_impostor) > 1:
        bw_data.append(np.asarray(base_impostor, dtype=float))
    # si faltan baseline, incorpora WM
    for _, data in per_experiment_data:
        for arr in (data.get('genuine', {}).get('watermarked', None),
                    data.get('impostor', {}).get('watermarked', None)):
            if arr is not None and len(arr) > 1:
                bw_data.append(np.asarray(arr, dtype=float))
    if not bw_data:
        return None  # no hay datos suficientes para KDE

    bw_common = _silverman_bandwidth(np.concatenate(bw_data)) * float(bw_scale)
    bw_common = max(bw_common, 1e-9)

    # ---------- Pre-cálculo de KDEs y y-max global ----------
    base_g_kde = kde_eval(base_genuine, grid, bw=bw_common, bw_scale=1.0) if base_genuine is not None and len(base_genuine) > 1 else None
    base_i_kde = kde_eval(base_impostor, grid, bw=bw_common, bw_scale=1.0) if base_impostor is not None and len(base_impostor) > 1 else None

    wm_kdes = []  # [(bpp_label, g_kde, i_kde)]
    y_max = 0.0
    for arr in (base_g_kde, base_i_kde):
        if arr is not None and arr.size > 0:
            y_max = max(y_max, float(np.nanmax(arr)))

    for bpp_label, data in per_experiment_data:
        g = data.get('genuine', {}).get('watermarked', None)
        i = data.get('impostor', {}).get('watermarked', None)

        g_kde = kde_eval(g, grid, bw=bw_common, bw_scale=1.0) if g is not None and len(g) > 1 else None
        i_kde = kde_eval(i, grid, bw=bw_common, bw_scale=1.0) if i is not None and len(i) > 1 else None

        if g_kde is not None and g_kde.size > 0:
            y_max = max(y_max, float(np.nanmax(g_kde)))
        if i_kde is not None and i_kde.size > 0:
            y_max = max(y_max, float(np.nanmax(i_kde)))

        wm_kdes.append((bpp_label, g_kde, i_kde))

    if y_max <= 0:
        y_max = 1.0
    y_lim = (0.0, y_max * 1.05)

    # ---------- Figura vertical: comparte X e Y ----------
    height = 3.0 * N + 1.0
    fig, axes = plt.subplots(
        nrows=N, ncols=1, figsize=(10, height),
        sharex=True, sharey=True   # << comparte X e Y
    )
    if N == 1:
        axes = [axes]

    # ---------- Dibujo ----------
    for idx, ((bpp_label, _data), ax) in enumerate(zip(per_experiment_data, axes)):
        # Baseline (misma en todos)
        if base_g_kde is not None:
            ax.plot(grid, base_g_kde, color=color_genuine, linewidth=lw_all, alpha=1.0,
                    label="Genuine (base)", linestyle='solid')
        if base_i_kde is not None:
            ax.plot(grid, base_i_kde, color=color_impostor, linewidth=lw_all, alpha=1.0,
                    label="Impostor (base)", linestyle='solid')

        # WM de este bpp (precomputado con bw_common)
        _, g_kde, i_kde = wm_kdes[idx]
        if g_kde is not None:
            if style == "lines":
                ax.plot(grid, g_kde, color=color_genuine, linewidth=lw_all, alpha=0.95,
                        linestyle='dashed', label="Genuine (wm)")
            elif style == "fill":
                ax.fill_between(grid, 0, g_kde, color=color_genuine, alpha=0.25, label="Genuine (wm)")
                ax.plot(grid, g_kde, color=color_genuine, linewidth=lw_all, alpha=0.95, linestyle='dotted')
        if i_kde is not None:
            if style == "lines":
                ax.plot(grid, i_kde, color=color_impostor, linewidth=lw_all, alpha=0.95,
                        linestyle='dashed', label="Impostor (wm)")
            elif style == "fill":
                ax.fill_between(grid, 0, i_kde, color=color_impostor, alpha=0.25, label="Impostor (wm)")
                ax.plot(grid, i_kde, color=color_impostor, linewidth=lw_all, alpha=0.95, linestyle='dotted')

        # Estética
        ax.set_title(f"{dataset} — {metric} — bpp={bpp_label}")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(*y_lim)  # << misma escala Y para todas las filas
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
        if idx == N - 1:
            ax.set_xlabel("Distance")
        ax.set_ylabel("Density")

        # Leyenda sin duplicados
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        h_clean, l_clean = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                h_clean.append(h); l_clean.append(l); seen.add(l)
        ax.legend(h_clean, l_clean, ncol=2, fontsize=8, loc="best")

    fig.suptitle(f"{dataset} — {metric} KDE (baseline + wm by bpp)", y=0.995)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.97])
    return fig



# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser(
        description="Multiplot vertical con KDE: baseline única + WM por bpp (1 columna, N filas)."
    )
    parser.add_argument("--algorithm", type=str, required=True,
                        help="Nombre de la carpeta del algoritmo (ej., stegaformer).")
    parser.add_argument("--metric", type=str, default="cosine",
                        help="Prefijo de métrica usado en los CSVs (ej., cosine).")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["CFD", "facelab_london", "LFW", "ONOT"],
                        help="Nombres de carpetas de datasets a incluir.")
    parser.add_argument("--FRModel", type=str, default="facenet",
                        help="Nombre de la carpeta del modelo de reconocimiento (ej., ArcFace).")
    parser.add_argument("--kde_points", type=int, default=512,
                        help="Número de puntos de la malla para KDE (default 512).")
    parser.add_argument("--bw_scale", type=float, default=1.0,
                        help="Escala multiplicativa del ancho de banda de Silverman (default 1.0).")
    parser.add_argument("--bins", type=int, default=None,
                        help="Alias de --kde_points (compatibilidad).")
    parser.add_argument("--style", type=str, default="lines", choices=["lines", "fill"],
                        help="Estilo de visualización: líneas finas o relleno transparente.")
    parser.add_argument("--save", action="store_true",
                        help="Guardar figuras en <repo>/evaluation/visualizations/<algorithm>/plots")
    parser.add_argument("--show", action="store_true",
                        help="Mostrar figuras interactivamente.")
    args = parser.parse_args()

    # Compatibilidad: --bins como alias de --kde_points
    if args.bins is not None and args.bins > 0:
        args.kde_points = args.bins

    # Rutas
    recognition_root = repo_root() / "experiments" / "output" / "recognition"
    experiments = discover_experiments(recognition_root, args.algorithm)
    print(f"Found {len(experiments)} experiments under {recognition_root / args.algorithm}")

    # Por dataset, una lista de (bpp, data)
    per_dataset: Dict[str, List[Tuple[str, Dict]]] = {ds: [] for ds in args.datasets}

    # Ordena por bpp
    exp_info = []
    for exp_path in experiments:
        name = exp_path.name
        bpp_str = parse_bpp(name)
        try:
            bpp_val = float(bpp_str)
        except Exception:
            bpp_val = float("inf")
        exp_info.append((bpp_val, bpp_str, exp_path))
    exp_info.sort(key=lambda x: x[0])

    # Carga datos
    for _, bpp_str, exp_path in exp_info:
        for ds in args.datasets:
            data = load_distances(exp_path, ds, args.metric, args.FRModel)
            if any(len(v) > 0 for section in data.values() for v in section.values()):
                per_dataset[ds].append((bpp_str, data))

    # Salida
    plots_dir = repo_root() / "evaluation" / "visualizations" / args.algorithm / "plots"
    if args.save:
        plots_dir.mkdir(parents=True, exist_ok=True)

    # Una figura por dataset (vertical)
    for ds, items in per_dataset.items():
        if not items:
            print(f"WARNING: No data found for dataset '{ds}'. Skipping.")
            continue

        fig = plot_dataset_vertical_kde(
            dataset=ds,
            per_experiment_data=items,
            kde_points=args.kde_points,
            bw_scale=args.bw_scale,
            metric=args.metric,
            style=args.style
        )
        if fig is None:
            continue

        if args.save:
            suffix = f"kde_{args.style}_pts{args.kde_points}_bw{args.bw_scale}".replace('.', 'p')
            out_path = plots_dir / f"{ds}_{args.metric}_vertical_{suffix}.png"
            fig.savefig(out_path, dpi=200)
            print(f"Saved: {out_path}")

        if args.show:
            plt.show()
        else:
            plt.close(fig)


if __name__ == "__main__":
    main()

#python visualize_auth_plots_many.py --algorithm stegaformer --metric cosine --datasets CFD facelab_london --style lines --kde_points 512 --bw_scale 0.5 --save