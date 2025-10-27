# coding: utf-8
"""
Plot grid of variation percentages vs BPP for a given watermarking model and train dataset.

- Rows: one per evaluation dataset (CFD, ONOT, LFW, facelab_london, ... found under recognition)
- Columns (4): [Offline-WM, Offline-WM_BOTH, Online-WM, Online-WM_BOTH]
  * WM    = variation due to watermark on probe only (single-watermark)
  * WM_BOTH = variation due to watermark on both template and probe
- Each subplot plots two series: Genuine and Impostor with error bars (std), Y in %
- X axis: BPP derived from experiment name prefix using custom rules:
    - If exp starts "1_1" -> bpp = 1
    - If exp starts "1_3" -> bpp = 3
    - If exp starts "15_2" -> bpp = 8
    - If exp starts "3_3" -> bpp = 6
    - Otherwise: default to 2nd numeric token in the exp name

Inputs:
  --root: Windows base root to 'experiments' (e.g. C:\\Users\\...\\WAFIS-ICAOeval\\experiments)
  --wm_model: watermarking model folder (e.g. stegaformer)
  --train_dataset: training dataset (e.g. celeba_hq)
  --recognizer: recognizer name (default: facenet)
  --metric: cosine|euclidean (default: cosine)
  --out_dir: where to save the figure; will create subdirs <wm_model>/<train_dataset>/<recognizer>/
  --dpi: output dpi (default: 150)

Author: AI Project Partner: Watermarking & Steganography
"""

import re
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

def parse_bpp_from_experiment(exp_name: str) -> Optional[int]:
    """
    Custom BPP parsing per user's rules/examples.
    - Try to read first two integers in the name prefix.
    - Apply special cases: (15,2)->8, (3,3)->6, (1,b)->b
    - Fallback: 2nd numeric token.
    """
    if not exp_name:
        return None
    nums = re.findall(r'\d+', exp_name)
    if len(nums) >= 2:
        a, b = int(nums[0]), int(nums[1])
        if a == 15 and b == 2:
            return 8
        if a == 3 and b == 3:
            return 6
        if a == 1:
            return b
        # default to 2nd token
        return b
    elif len(nums) == 1:
        return int(nums[0])
    return None

def read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def collect_datasets(root: Path, wm_model: str, train_dataset: str, recognizer: str) -> Dict[str, List[Tuple[str, Path]]]:
    """
    Return mapping dataset_name -> list of (experiment_name, facenet_dir)
    Expected layout:
      <root>/output/recognition/<wm_model>/<experiment>/<train_dataset>/<dataset>/<recognizer>/results_summary*.json
    """
    base = root / "output" / "recognition" / wm_model
    out: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)
    if not base.exists():
        return out
    for exp_dir in base.iterdir():
        if not exp_dir.is_dir():
            continue
        exp_name = exp_dir.name
        td_dir = exp_dir / train_dataset
        if not td_dir.exists():
            continue
        for ds_dir in td_dir.iterdir():
            if not ds_dir.is_dir():
                continue
            rec_dir = ds_dir / recognizer
            if rec_dir.exists():
                out[ds_dir.name].append((exp_name, rec_dir))
    return out

def extract_variations(rec_dir: Path, metric: str, both: bool, online: bool) -> Optional[Tuple[float, float, float, float]]:
    """
    Read results_summary(.json|_online.json) and extract:
      Genuine_avg%, Genuine_std%, Impostor_avg%, Impostor_std%
    for (both==False -> single WM) or (both==True -> WM_BOTH), and offline/online choice.
    """
    json_name = "results_summary_online.json" if online else "results_summary.json"
    data = read_json(rec_dir / json_name)
    if not data:
        return None
    avg = data.get("average_distances", {})
    std = data.get("std_distances", {})
    suffix = "_both" if both else ""

    g_avg = avg.get(f"facenet_avg_var_dist_{metric}_genuine_due_watermark{suffix}")
    g_std = std.get(f"facenet_std_var_dist_{metric}_genuine_due_watermark{suffix}")
    i_avg = avg.get(f"facenet_avg_var_dist_{metric}_impostor_due_watermark{suffix}")
    i_std = std.get(f"facenet_std_var_dist_{metric}_impostor_due_watermark{suffix}")

    # Values are expected in %, but we accept floats; leave as-is
    if g_avg is None and i_avg is None:
        return None
    try:
        return float(g_avg) if g_avg is not None else np.nan, \
               float(g_std) if g_std is not None else np.nan, \
               float(i_avg) if i_avg is not None else np.nan, \
               float(i_std) if i_std is not None else np.nan
    except Exception:
        return None

def aggregate_by_bpp(entries: List[Tuple[int, Tuple[float,float,float,float]]]):
    """
    entries: list of (bpp, (g_avg,g_std,i_avg,i_std))
    return per bpp:
      bpp -> (means[2], stds[2]) where means = [genuine_mean, impostor_mean]
           and stds  = [genuine_std,  impostor_std] (averaged over experiments)
    We average means; stds are averaged as a rough summary (not pooled).
    """
    from collections import defaultdict
    by_bpp = defaultdict(list)
    for bpp, vals in entries:
        by_bpp[bpp].append(vals)
    out = {}
    for bpp, vals_list in by_bpp.items():
        arr = np.array(vals_list, dtype=float)  # shape (n,4)
        g_means = arr[:,0]
        g_stds  = arr[:,1]
        i_means = arr[:,2]
        i_stds  = arr[:,3]
        means = np.array([np.nanmean(g_means), np.nanmean(i_means)])
        stds  = np.array([np.nanmean(g_stds),  np.nanmean(i_stds)])
        out[bpp] = (means, stds)
    return out

def plot_grid(datasets: List[str],
              data_off_wm, data_off_both, data_on_wm, data_on_both,
              wm_model: str, train_dataset: str, recognizer: str, metric: str,
              out_path: Path, dpi: int = 150):
    rows = len(datasets)
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 2.8*rows), squeeze=False, sharex=True, sharey=True)
    col_titles = ["Offline - WM", "Offline - WM_BOTH", "Online - WM", "Online - WM_BOTH"]

    for r, ds in enumerate(datasets):
        ds_maps = [data_off_wm.get(ds, {}), data_off_both.get(ds, {}), data_on_wm.get(ds, {}), data_on_both.get(ds, {})]
        for c in range(cols):
            ax = axes[r][c]
            mapping = ds_maps[c]
            if not mapping:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(col_titles[c])
                continue
            xs = sorted(mapping.keys())
            gm = [mapping[b][0][0] for b in xs]
            gs = [mapping[b][1][0] for b in xs]
            im = [mapping[b][0][1] for b in xs]
            istd = [mapping[b][1][1] for b in xs]

            ax.errorbar(xs, gm, yerr=gs, marker='o', linestyle='-', label='Genuine')
            ax.errorbar(xs, im, yerr=istd, marker='^', linestyle='--', label='Impostor')
            ax.set_xlabel("BPP")
            #ax.set_yscale('symlog', linthresh=0.02) 
            if c == 0:
                ax.set_ylabel(f"{ds} — Variation (%)")
            ax.grid(True, alpha=0.3)
            if r == 0:
                ax.set_title(col_titles[c])
            if r == 0 and c == 0:
                ax.legend(loc='best')

    fig.suptitle(f"Variation vs BPP — {wm_model} / {train_dataset} / {recognizer} / {metric}", y=0.995, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    print(f"[OK] Saved figure -> {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=False, default=r"C:\Users\JotaR\Documents\Github\WAFIS-ICAOeval\experiments",
                    help=r"Base path to 'experiments' (e.g., C:\Users\JotaR\Documents\Github\WAFIS-ICAOeval\experiments)")
    ap.add_argument("--wm_model", type=str, default="stegaformer")
    ap.add_argument("--train_dataset", type=str, default="coco")
    ap.add_argument("--recognizer", type=str, default="facenet")
    ap.add_argument("--metric", type=str, choices=["cosine", "euclidean"], default="cosine")
    ap.add_argument("--out_dir", type=str, required=False, default=r"C:\Users\JotaR\Documents\Github\WAFIS-ICAOeval\evaluation\visualizations",
                    help=r"Directory to store the figure; subfolders <wm_model>/<train_dataset>/<recognizer> will be created")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) / args.wm_model / "plots" / args.train_dataset / args.recognizer
    out_path = out_dir / f"variation_grid_{args.metric}.png"

    mapping = collect_datasets(root, args.wm_model, args.train_dataset, args.recognizer)
    if not mapping:
        print("[ERROR] No datasets found under the given root/model/train_dataset/recognizer")
        return

    datasets = sorted(mapping.keys())

    data_off_wm = {}
    data_off_both = {}
    data_on_wm = {}
    data_on_both = {}

    for ds in datasets:
        entries_off_wm = []
        entries_off_both = []
        entries_on_wm = []
        entries_on_both = []

        for exp_name, rec_dir in mapping[ds]:
            bpp = parse_bpp_from_experiment(exp_name)
            if bpp is None:
                continue

            # OFFLINE
            vals_wm_off = extract_variations(rec_dir, args.metric, both=False, online=False)
            if vals_wm_off is not None:
                entries_off_wm.append((bpp, vals_wm_off))

            vals_both_off = extract_variations(rec_dir, args.metric, both=True, online=False)
            if vals_both_off is not None:
                entries_off_both.append((bpp, vals_both_off))

            # ONLINE
            vals_wm_on = extract_variations(rec_dir, args.metric, both=False, online=True)
            if vals_wm_on is not None:
                entries_on_wm.append((bpp, vals_wm_on))

            vals_both_on = extract_variations(rec_dir, args.metric, both=True, online=True)
            if vals_both_on is not None:
                entries_on_both.append((bpp, vals_both_on))

        data_off_wm[ds]   = aggregate_by_bpp(entries_off_wm)   if entries_off_wm   else {}
        data_off_both[ds] = aggregate_by_bpp(entries_off_both) if entries_off_both else {}
        data_on_wm[ds]    = aggregate_by_bpp(entries_on_wm)    if entries_on_wm    else {}
        data_on_both[ds]  = aggregate_by_bpp(entries_on_both)  if entries_on_both  else {}

    any_data = any([bool(d) for d in [data_off_wm, data_off_both, data_on_wm, data_on_both]])
    if not any_data:
        print("[ERROR] No data points to plot. Check JSON keys/metric or paths.")
        return

    plot_grid(datasets, data_off_wm, data_off_both, data_on_wm, data_on_both,
              args.wm_model, args.train_dataset, args.recognizer, args.metric, out_path, dpi=args.dpi)

if __name__ == "__main__":
    main()

#python plot_variation_grid.py `
#  --root "C:\Users\JotaR\Documents\Github\WAFIS-ICAOeval\experiments" `
#  --wm_model stegaformer `
#  --train_dataset celeba_hq `
#  --recognizer facenet `
#  --metric cosine `
#  --out_dir "C:\Users\JotaR\Documents\Github\WAFIS-ICAOeval\evaluation\plots" `
#  --dpi 180