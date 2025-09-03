# visualize_auth_plots.py
# Recorre experiments/output/recognition/<algoritmo>/<experimento>/<dataset>
# y grafica UNA baseline (genuine + impostor) y, por cada bpp, solo las curvas watermarked.
# Estilos:
#   --style lines (default): líneas finas, baseline sólida, bpp con estilos 'dashed'/'dashdot'/'dotted' en ciclo
#   --style fill: baseline línea fina; watermarked con relleno transparente + borde punteado

import argparse
import re
from typing import Dict, List, Tuple
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


def load_distances(exp_path: Path, dataset: str, metric: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Carga los cuatro CSVs de distancias para un experimento+dataset+métrica."""
    files = {
        ('genuine', 'baseline'):     f"{metric}_genuine_distances_baseline.csv",
        ('genuine', 'watermarked'):  f"{metric}_genuine_distances_watermarked.csv",
        ('impostor', 'baseline'):    f"{metric}_impostor_distances_baseline.csv",
        ('impostor', 'watermarked'): f"{metric}_impostor_distances_watermarked.csv",
    }
    dataset_dir = Path(exp_path) / dataset
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
    return [p for p in algo_dir.iterdir() if p.is_dir() and p.name.lower() != "plots"]


# ---------------- Estilos ----------------

def line_style_cycle():
    """
    Ciclo de estilos de línea seguros para plt.hist (strings, NO tuples).
    Evitamos 'solid' porque ya la usa la baseline.
    """
    return ['dashed', 'dashdot', 'dotted']


# ---------------- Plotting ----------------

def plot_dataset_single_baseline(
    ax,
    dataset: str,
    per_experiment_data: List[Tuple[str, Dict[str, Dict[str, np.ndarray]]]],
    bins: int = 100,
    metric: str = "cosine",
    style: str = "lines"
):
    """
    Dibuja UNA baseline (genuino & impostor) y, para cada bpp, SOLO las curvas watermarked.
    style: "lines" (líneas finas + dash) | "fill" (relleno transparente + borde punteado)
    """
    # Colores por clase:
    color_genuine = "#1f77b4"   # azul
    color_impostor = "#d62728"  # rojo

    lw_all = 1.5  # grosor fino común
    density = True

    # 1) Tomar baseline (primera disponible)
    base_genuine = None
    base_impostor = None
    for _, data in per_experiment_data:
        if base_genuine is None and 'baseline' in data.get('genuine', {}):
            base_genuine = data['genuine']['baseline']
        if base_impostor is None and 'baseline' in data.get('impostor', {}):
            base_impostor = data['impostor']['baseline']
        if base_genuine is not None and base_impostor is not None:
            break

    # 2) Límites X globales
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
    else:
        xmin, xmax = 0.0, 1.0

    # 3) Baseline (una vez)
    if base_genuine is not None and len(base_genuine) > 0:
        ax.hist(base_genuine, bins=bins, density=density, histtype="step",
                color=color_genuine, linewidth=lw_all, alpha=1.0,
                label="Genuino (base)", linestyle='solid')
    if base_impostor is not None and len(base_impostor) > 0:
        ax.hist(base_impostor, bins=bins, density=density, histtype="step",
                color=color_impostor, linewidth=lw_all, alpha=1.0,
                label="Impostor (base)", linestyle='solid')

    # 4) Watermarked por bpp (ordenado por bpp)
    def _try_float(b):
        try:
            return float(b)
        except Exception:
            return float("inf")
    per_experiment_data = sorted(per_experiment_data, key=lambda t: _try_float(t[0]))
    styles = line_style_cycle()
    sidx = 0

    for bpp_label, data in per_experiment_data:
        gwm = data.get('genuine', {}).get('watermarked', None)
        iwm = data.get('impostor', {}).get('watermarked', None)

        if style == "lines":
            ls = styles[sidx % len(styles)]
            if gwm is not None and len(gwm) > 0:
                ax.hist(gwm, bins=bins, density=density, histtype="step",
                        color=color_genuine, linewidth=lw_all, alpha=0.95,
                        label=f"Genuino (wm) bpp={bpp_label}", linestyle=ls)
            if iwm is not None and len(iwm) > 0:
                ax.hist(iwm, bins=bins, density=density, histtype="step",
                        color=color_impostor, linewidth=lw_all, alpha=0.95,
                        label=f"Impostor (wm) bpp={bpp_label}", linestyle=ls)
        elif style == "fill":
            if gwm is not None and len(gwm) > 0:
                ax.hist(gwm, bins=bins, density=density, histtype="stepfilled",
                        color=color_genuine, linewidth=lw_all, alpha=0.25,
                        label=f"Genuino (wm) bpp={bpp_label}")
                ax.hist(gwm, bins=bins, density=density, histtype="step",
                        color=color_genuine, linewidth=lw_all, alpha=0.95,
                        linestyle='dotted')
            if iwm is not None and len(iwm) > 0:
                ax.hist(iwm, bins=bins, density=density, histtype="stepfilled",
                        color=color_impostor, linewidth=lw_all, alpha=0.25,
                        label=f"Impostor (wm) bpp={bpp_label}")
                ax.hist(iwm, bins=bins, density=density, histtype="step",
                        color=color_impostor, linewidth=lw_all, alpha=0.95,
                        linestyle='dotted')
        else:
            raise ValueError("style debe ser 'lines' o 'fill'.")

        sidx += 1

    # Estética
    ax.set_title(f"{dataset} — {metric} distances")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Density")
    ax.set_xlim(xmin, xmax)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)

    # Leyenda compacta (sin duplicados)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    h_clean, l_clean = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            h_clean.append(h)
            l_clean.append(l)
            seen.add(l)
    ax.legend(h_clean, l_clean, ncol=2, fontsize=8)


# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualiza distribuciones: baseline única + watermarked por bpp (líneas finas o relleno)."
    )
    parser.add_argument("--algorithm", type=str, required=True,
                        help="Nombre de la carpeta del algoritmo (ej., stegaformer).")
    parser.add_argument("--metric", type=str, default="cosine",
                        help="Prefijo de métrica usado en los CSVs (ej., cosine).")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["CFD", "facelab_london"],
                        help="Nombres de carpetas de datasets a incluir.")
    parser.add_argument("--bins", type=int, default=100, help="Número de bins para histograma.")
    parser.add_argument("--style", type=str, default="lines", choices=["lines", "fill"],
                        help="Estilo de visualización: líneas finas con guiones o relleno transparente.")
    parser.add_argument("--save", action="store_true",
                        help="Guardar figuras en <repo>/evaluation/visualizations/<algorithm>/plots")
    parser.add_argument("--show", action="store_true",
                        help="Mostrar figuras interactivamente.")
    args = parser.parse_args()

    # Rutas
    recognition_root = repo_root() / "experiments" / "output" / "recognition"
    experiments = discover_experiments(recognition_root, args.algorithm)

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
            data = load_distances(exp_path, ds, args.metric)
            if any(len(v) > 0 for section in data.values() for v in section.values()):
                per_dataset[ds].append((bpp_str, data))

    # Salida
    plots_dir = repo_root() / "evaluation" / "visualizations" / args.algorithm / "plots"
    if args.save:
        plots_dir.mkdir(parents=True, exist_ok=True)

    # Una figura por dataset
    for ds, items in per_dataset.items():
        if not items:
            print(f"WARNING: No data found for dataset '{ds}'. Skipping.")
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_dataset_single_baseline(ax, ds, items, bins=args.bins, metric=args.metric, style=args.style)

        if args.save:
            suffix = "lines" if args.style == "lines" else "fill"
            out_path = plots_dir / f"{ds}_{args.metric}_baseline_once_{suffix}.png"
            fig.tight_layout()
            fig.savefig(out_path, dpi=200)
            print(f"Saved: {out_path}")

        if args.show:
            plt.show()
        else:
            plt.close(fig)


if __name__ == "__main__":
    main()

# python .\visualize_auth_plots.py --algorithm stegaformer --metric cosine --datasets CFD facelab_london --bins 100 --save 