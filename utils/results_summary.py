# coding: utf-8
"""
Consolidate watermarking + recognition metrics to Excel and HTML.

- Dynamically crawls under multiple output roots.
- Normalizes folder names to lowercase for robust grouping.
- Builds two tables:
  1) Watermarking metrics (ACC, PSNR, SSIM) with ± std and their "offline" counterparts.
  2) Recognition metrics (Facenet/Cosine for now) for
     Original-Original (baseline), Watermarked-Original (watermarked),
     Watermarked-Watermarked (watermarked_both).

Author: AI Project Partner: Watermarking & Steganography
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

# ---------- helpers ----------

def fmt_pm(val, std, scale=1.0, ndigits=3) -> str:
    """Format as 'value ± std' with optional scaling (e.g., 0..1 to %)."""
    if val is None:
        return ""
    v = float(val) * scale
    if std is None:
        return f"{v:.{ndigits}f}"
    s = float(std) * scale
    return f"{v:.{ndigits}f} ± {s:.{ndigits}f}"

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def lower(s: str) -> str:
    return s.lower() if isinstance(s, str) else s

def parse_bpp_from_experiment(exp_name: Optional[str]) -> Optional[int]:
    """
    Extract BPP as the 2nd numeric token in the experiment name.
    Examples:
      '1_1_clamp' -> 1
      '1_3_255_w16_learn_im' -> 3
    """
    if not exp_name:
        return None
    parts = exp_name.split("_")
    # Fast path: second field is purely digits
    if len(parts) >= 2 and parts[1].isdigit():
        return int(parts[1])
    # Fallback: collect numeric tokens and take the 2nd one
    nums = [int(p) for p in parts if p.isdigit()]
    return nums[1] if len(nums) >= 2 else (nums[0] if nums else None)

# ---------- discovery: watermarking ----------

def find_watermarking_jsons(roots: List[Path]) -> List[Path]:
    """
    Looks for results_summary.json produced by watermarking inference.
    Skips recognition paths (contain 'recognition/facenet').
    """
    out = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("results_summary.json"):
            low = str(p).lower()
            if "recognition" in low and "facenet" in low:
                continue
            out.append(p)
    return out

def parse_watermarking_json(p: Path) -> Dict[str, Any]:
    try:
        data = json.loads(p.read_text())
    except Exception:
        return {}

    parts_low = [lower(x) for x in p.parts]
    dataset = data.get("inference_dataset")
    train_dataset = data.get("training_dataset")
    model = data.get("model_name")
    bpp = data.get("bpp")
    exp_name = data.get("experiment_name")

    # Fallback inference from path .../{model}/{exp}/inference/{train_dataset}/{dataset}/results_summary.json
    if "inference" in parts_low:
        i = parts_low.index("inference")
        if train_dataset is None and i + 1 < len(p.parts):
            train_dataset = p.parts[i + 1]
        if dataset is None and i + 2 < len(p.parts):
            dataset = p.parts[i + 2]
        if model is None and i - 2 >= 0:
            model = p.parts[i - 2]
        if exp_name is None and i - 1 >= 0:
            exp_name = p.parts[i - 1]
        

    return {
        "train_dataset": lower(train_dataset or "unknown"),
        "dataset": lower(dataset or "unknown"),
        "experiment": lower(exp_name or "unknown"),
        "model": lower(model or "unknown"),
        "bpp": bpp,
        "acc": safe_float(data.get("accuracy")),
        "acc_std": safe_float(data.get("accuracy_std")),
        "psnr": safe_float(data.get("psnr")),
        "psnr_std": safe_float(data.get("psnr_std")),
        "ssim": safe_float(data.get("ssim")),
        "ssim_std": safe_float(data.get("ssim_std")),
        "acc_offline": safe_float(data.get("accuracy_offline")),
        "acc_offline_std": safe_float(data.get("accuracy_offline_std")),
        "psnr_offline": safe_float(data.get("psnr_offline")),
        "psnr_offline_std": safe_float(data.get("psnr_offline_std")),
        "ssim_offline": safe_float(data.get("ssim_offline")),
        "ssim_offline_std": safe_float(data.get("ssim_offline_std")),
    }

def build_watermarking_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append({
            "Train Dataset": r["train_dataset"],
            "Dataset": r["dataset"],
            "Experiment": r["experiment"],
            "Model": r["model"],
            "BPP": r["bpp"],
            "ACC %": fmt_pm(r["acc"], r["acc_std"], scale=100.0),
            "PSNR": fmt_pm(r["psnr"], r["psnr_std"]),
            "SSIM": fmt_pm(r["ssim"], r["ssim_std"]),
            "ACC % offline": fmt_pm(r["acc_offline"], r["acc_offline_std"], scale=100.0),
            "PSNR offline": fmt_pm(r["psnr_offline"], r["psnr_offline_std"]),
            "SSIM offline": fmt_pm(r["ssim_offline"], r["ssim_offline_std"]),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["BPP"] = pd.to_numeric(df["BPP"], errors="coerce")
        df = df.sort_values(["Train Dataset","Dataset", "Experiment","Model", "BPP"], na_position="last").reset_index(drop=True)
    return df

# ---------- discovery: recognition ----------

def find_recognition_jsons(roots: List[Path]) -> List[Path]:
    """
    Recognition summary path (Facenet):
    output/recognition/{wm_model}/{experiment}/{dataset}/facenet/results_summary.json
    """
    out = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("results_summary.json"):
            low = str(p).lower()
            if "recognition" in low and "facenet" in low:
                out.append(p)
    return out

def parse_recognition_json(p: Path) -> List[Dict[str, Any]]:
    """Parse one recognition results_summary.json and expand to 3 rows (scenarios)."""
    try:
        data = json.loads(p.read_text())
    except Exception:
        return []

    parts = [lower(x) for x in p.parts]
    dataset = "unknown"
    train_dataset = "unknown"
    wm_model = "unknown"
    exp_name = None

    # Expected: .../recognition/{wm_model}/{exp}/{train_dataset}/{dataset}/facenet/results_summary.json
    if "recognition" in parts:
        i = parts.index("recognition")
        if i + 1 < len(parts):
            wm_model = parts[i + 1]
        if i + 2 < len(parts):
            exp_name = parts[i + 2]
        if i + 3 < len(parts):
            train_dataset = parts[i + 3]
        if i + 4 < len(parts):
            dataset = parts[i + 4]

    bpp = parse_bpp_from_experiment(exp_name)

    metrics: Dict[str, Any] = data.get("recognition_metrics", {})

    # Determine metric (cosine/euclidean) from key suffix
    metric = "Cosine"
    for k in metrics.keys():
        m = re.search(r'_(cosine|euclidean)$', k)
        if m:
            metric = m.group(1).capitalize()
            break
    metric_suffix = metric.lower()

    def key(stat: str, tag: str) -> str:
        # Build the exact key, e.g.:
        # 'facenet_EER_baseline_cosine' or 'facenet_TAR_at_FAR_watermarked_both_cosine'
        return f"facenet_{stat}_{tag}_{metric_suffix}"

    def get(stat: str, tag: str):
        return metrics.get(key(stat, tag), None)

    # Map scenarios to tags in keys
    scenarios = [
        ("Original-Original", "baseline"),
        ("Watermarked-Original", "watermarked"),
        ("Watermarked-Watermarked", "watermarked_both"),
    ]

    rows = []
    for probe_ref, tag in scenarios:
        rows.append({
            "Train Dataset": train_dataset,
            "Dataset": dataset,
            "Experiment": exp_name,
            "Model": wm_model,
            "BPP": bpp,
            "Model_recog": "Facenet",
            "Metric": metric,
            "Probe-Reference": probe_ref,
            "EER": get("EER", tag),
            "FAR at EER": get("FAR_at_EER", tag),
            "FRR at EER": get("FRR_at_EER", tag),
            "TAR at FAR ~0.01%": get("TAR_at_FAR", tag),
            "Actual_FAR": get("Actual_FAR", tag),
            "AUC": get("AUC", tag),
        })
    return rows

def build_recognition_df(json_paths: List[Path]) -> pd.DataFrame:
    rows = []
    for p in json_paths:
        rows.extend(parse_recognition_json(p))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Train Dataset", "Dataset", "Model", "Experiment", "BPP", "Probe-Reference"]).reset_index(drop=True)
    return df

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+",
                    default=[Path(r"C:\Users\JotaR\Documents\Github\WAFIS-ICAOeval\experiments\output")],
                    help="Root directories to search for results (recursively).")
    ap.add_argument("--out_xlsx", default=Path(r"C:\Users\JotaR\Documents\Github\WAFIS-ICAOeval\evaluation\results\consolidated_metrics.xlsx"))
    ap.add_argument("--out_html", default=Path(r"C:\Users\JotaR\Documents\Github\WAFIS-ICAOeval\evaluation\results\consolidated_metrics.html"))
    args = ap.parse_args()

    roots = [Path(r) for r in args.roots]

    # Watermarking
    wm_jsons = find_watermarking_jsons(roots)
    wm_records = [parse_watermarking_json(p) for p in wm_jsons]
    wm_records = [r for r in wm_records if r]  # drop empty
    wm_df = build_watermarking_df(wm_records)

    # Recognition
    rec_jsons = find_recognition_jsons(roots)
    rec_df = build_recognition_df(rec_jsons)

    # Export
    out_xlsx = Path(args.out_xlsx)
    out_html = Path(args.out_html)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        wm_df.to_excel(writer, sheet_name="Watermarking metrics", index=False)
        rec_df.to_excel(writer, sheet_name="Recognition metrics", index=False)

    html = f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Consolidated Metrics</title>
      <style>
        body {{ font-family: Arial, sans-serif; }}
        table {{ border-collapse: collapse; font-size: 13px; }}
        th, td {{ border: 1px solid #ccc; padding: 6px 8px; }}
        th {{ background: #f5f5f5; }}
      </style>
    </head>
    <body>
      <h2>Watermarking metrics</h2>
      {wm_df.to_html(index=False, escape=False)}
      <h2>Recognition metrics</h2>
      {rec_df.to_html(index=False, escape=False)}
    </body>
    </html>"""
    out_html.write_text(html, encoding="utf-8")

    print(f"[OK] Excel -> {out_xlsx}")
    print(f"[OK] HTML  -> {out_html}")

if __name__ == "__main__":
    main()

