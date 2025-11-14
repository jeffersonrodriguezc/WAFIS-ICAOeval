#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolidate watermarking results_summary.json into a single Excel file (.xlsx)

Folder pattern (relative to --experiments-root):
  <algo>/<experiment>/inference/<train_ds>/<test_ds>/results_summary.json

Extracted columns (required):
  model_name, training_dataset, bpp,
  accuracy, accuracy_std, accuracy_offline, accuracy_offline_std,
  psnr, psnr_std, psnr_offline, psnr_offline_std,
  ssim, ssim_std, ssim_offline, ssim_offline_std

Additional identifiers added for traceability:
  algorithm, experiment_name, inference_dataset, timestamp, json_path
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd


REQUIRED_FIELDS = [
    "model_name",
    "training_dataset",
    "bpp",
    "accuracy",
    "accuracy_std",
    "accuracy_offline",
    "accuracy_offline_std",
    "psnr",
    "psnr_std",
    "psnr_offline",
    "psnr_offline_std",
    "ssim",
    "ssim_std",
    "ssim_offline",
    "ssim_offline_std",
]

EXTRA_ID_FIELDS = [
    "algorithm",
    "experiment_name",
    "inference_dataset",
    "timestamp",
    "json_path",
]

EXCEL_COLUMNS = EXTRA_ID_FIELDS + REQUIRED_FIELDS


def safe_get(d: Dict[str, Any], key: str, default: Any = None):
    """Return d[key] if present and not None; otherwise default."""
    val = d.get(key, default)
    return default if val is None else val


def parse_path_tokens(json_path: Path) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    From: <root>/<algo>/<experiment>/inference/<train_ds>/<test_ds>/results_summary.json
    Return: (algo, experiment, train_ds, test_ds)
    """
    parts = json_path.parts
    try:
        idx = len(parts) - 1 - parts[::-1].index("inference")
        algo = parts[idx - 2] if idx - 2 >= 0 else None
        experiment = parts[idx - 1] if idx - 1 >= 0 else None
        train_ds = parts[idx + 1] if idx + 1 < len(parts) else None
        test_ds = parts[idx + 2] if idx + 2 < len(parts) else None
        return algo, experiment, train_ds, test_ds
    except ValueError:
        return None, None, None, None


def collect_jsons(root: Path) -> List[Path]:
    """Find all results_summary.json files."""
    return list(root.rglob("inference/*/*/results_summary.json"))


def load_json(fp: Path) -> Optional[Dict[str, Any]]:
    try:
        with fp.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not parse JSON: {fp} -> {e}", file=sys.stderr)
        return None


def build_row(fp: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    algo, exp, train_ds_from_path, test_ds_from_path = parse_path_tokens(fp)
    row: Dict[str, Any] = {}

    # Identifiers
    row["algorithm"] = algo
    row["experiment_name"] = safe_get(payload, "experiment_name", exp)
    # Prefer JSON field; fallback to folder name after 'inference/<train>/<test>/'
    row["inference_dataset"] = safe_get(payload, "inference_dataset", test_ds_from_path)
    row["timestamp"] = safe_get(payload, "timestamp", "")
    row["json_path"] = str(fp.resolve())

    # Core metrics
    row["model_name"] = safe_get(payload, "model_name", "")
    row["training_dataset"] = safe_get(payload, "training_dataset", train_ds_from_path)
    row["bpp"] = safe_get(payload, "bpp", "")

    # Metrics
    for key in REQUIRED_FIELDS:
        if key not in row:
            row[key] = safe_get(payload, key, "")

    return row


def write_excel(rows: List[Dict[str, Any]], out_xlsx: Path) -> None:
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=EXCEL_COLUMNS)
    df.to_excel(out_xlsx, index=False)
    print(f"[OK] Excel file saved: {out_xlsx}")


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate watermarking results_summary.json into a single Excel file."
    )
    parser.add_argument(
        "--experiments-root",
        type=str,
        required=False,
        default=r"C:\Users\JotaR\Documents\Github\WAFIS-ICAOeval\experiments\output\watermarking",
        help="Root directory containing <algo>/<experiment>/inference/... trees.",
    )
    parser.add_argument(
        "--output-xlsx",
        type=str,
        required=False,
        default=r"C:\Users\JotaR\Documents\Github\WAFIS-ICAOeval\evaluation\results\consolidated_results_wm.xlsx",
        help="Path to write the consolidated Excel file.",
    )
    args = parser.parse_args()

    root = Path(args.experiments_root).expanduser().resolve()
    out_xlsx = Path(args.output_xlsx).expanduser().resolve()

    if not root.exists():
        print(f"[ERROR] Experiments root not found: {root}", file=sys.stderr)
        sys.exit(1)

    json_files = collect_jsons(root)
    if not json_files:
        print(f"[WARN] No results_summary.json files found under {root}", file=sys.stderr)
        write_excel([], out_xlsx)
        sys.exit(0)

    rows: List[Dict[str, Any]] = []
    missing_any_required = 0

    for fp in sorted(json_files):
        payload = load_json(fp)
        if payload is None:
            continue
        row = build_row(fp, payload)  # <-- FIX: solo esta llamada
        if any(str(row.get(k, "")).strip() == "" for k in REQUIRED_FIELDS):
            missing_any_required += 1
        rows.append(row)

    write_excel(rows, out_xlsx)

    print(f"[OK] Consolidated {len(rows)} file(s) into: {out_xlsx}")
    if missing_any_required > 0:
        print(f"[NOTE] {missing_any_required} row(s) had missing required fields; blanks were left as-is.")


if __name__ == "__main__":
    main()
