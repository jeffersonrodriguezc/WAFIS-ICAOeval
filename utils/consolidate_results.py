#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Consolida automáticamente resultados de múltiples archivos:
  - results_summary.json (offline)
  - results_summary_online.json (online)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
import re

KNOWN_DATASETS = {"CFD", "ONOT", "facelab_london", "LFW", "celeba_hq", "coco", "SCface", "ONOT_set1"}
RE_ALGO_GUESS = re.compile(r"(facenet|arcface|cosface|euc(?:lidean)?|cosine)", re.I)
RE_DIST_GUESS = re.compile(r"(cosine|euclidean|euc)", re.I)

def parse_bpp_from_tokens(exp_name: str, wm_model: Optional[str]) -> Optional[str]:
    """
    BPP según las primeras dos partes del nombre de experimento.
    - Para 'stegformer': si los dos primeros tokens son iguales (p.ej. 1_1) => 1 bpp; en otros casos, en blanco.
    - Para 'stegaformer': mapeo específico:
        1_1 -> 1
        1_3 -> 3
        3_3 -> 6
        15_2 -> 8
      Otros casos -> en blanco.
    - Si no hay modelo de watermark, devuelve en blanco.
    """
    try:
        parts = exp_name.split("_")
        if len(parts) < 2:
            return None
        a = int(parts[0])
        b = int(parts[1])
    except Exception:
        return None

    wm = (wm_model or "").lower()
    if wm == "stegformer":
        return "1" if a == b else None
    if wm == "stegaformer":
        mapping = {(1,1): "1", (1,3): "3", (3,3): "6", (15,2): "8"}
        return mapping.get((a,b))
    # Otros modelos: sin regla -> en blanco
    return None

def guess_model_from_keys(keys: List[str]) -> Optional[str]:
    for k in keys:
        m = re.match(r"([a-zA-Z0-9]+)_", k)
        if m:
            return m.group(1)
    return None

def find_first_parent_in_set(path: Path, names: set) -> Optional[str]:
    for part in path.parts[::-1]:
        if part in names:
            return part
    return None

def infer_train_dataset(path: Path) -> Optional[str]:
    """
    NUEVO (pedido): el dataset de entrenamiento es el directorio inmediatamente
    posterior al nombre del experimento en la ruta.
    Estructura esperada (ejemplo):
      .../<wm_model>/recognition/<rec_model>/<EXPERIMENTO>/<TRAIN_DATASET>/<TEST_DATASET>/results_summary*.json
    Si no se encuentra, caemos al comportamiento previo (buscar por nombres conocidos).
    """
    exp = infer_experiment_name(path)
    if exp:
        parts = list(path.parts)
        if exp in parts:
            idx = parts.index(exp)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    # Fallback
    return find_first_parent_in_set(path, KNOWN_DATASETS)

def infer_test_dataset(path: Path) -> Optional[str]:
    return find_first_parent_in_set(path, KNOWN_DATASETS)

def infer_experiment_name(path: Path) -> Optional[str]:
    parents = list(path.parents)
    for p in parents:
        if re.search(r"\d+_\d+_255_", p.name) or re.search(r"\d+_\d+_", p.name):
            return p.name
    return parents[1].name if len(parents) > 1 else None

def infer_model_name(path: Path, json_data: Dict[str, Any]) -> Optional[str]:
    for section in ("recognition_metrics", "average_distances", "std_distances"):
        if section in json_data and isinstance(json_data[section], dict):
            g = guess_model_from_keys(list(json_data[section].keys()))
            if g:
                return g
    m = RE_ALGO_GUESS.search(str(path))
    if m:
        return m.group(1).lower()
    return None

def infer_dist_metric(json_data: Dict[str, Any]) -> Optional[str]:
    for section in ("recognition_metrics", "average_distances", "std_distances"):
        sec = json_data.get(section) or {}
        for k in sec.keys():
            m = re.search(r"(cosine|euclidean)", k)
            if m:
                return m.group(1).lower()
    return None

def infer_watermark_model(path: Path) -> Optional[str]:
    """
    Heurística para nombre de modelo de watermarking:
    """
    parts = list(path.parts)
    if "recognition" in parts:
        idx = parts.index("recognition")
        return parts[idx + 1]

    return None


ORIG_ORIG = "Original - Original"
ORIG_WM   = "Original - Watermark"
WM_WM     = "Watermark - Watermark"

SUBCOLS_ORIG = ["EER","FAR at EER","FRR at EER","TAR at FAR ~0.01%","AUC"]
SUBCOLS_WM   = ["genuine avg % of change","genuine std % of change",
                "impostor avg % of change","impostor std % of change",
                "EER","FAR at EER","FRR at EER","TAR at FAR ~0.01%","AUC"]

def logical_schema() -> List[str]:
    cols = ["model","watermark model","train dataset","test dataset","bpp"]
    cols += [f"{ORIG_ORIG}:{c}" for c in SUBCOLS_ORIG]
    cols += [f"{ORIG_WM}:{c}" for c in SUBCOLS_WM]
    cols += [f"{WM_WM}:{c}" for c in SUBCOLS_WM]
    return cols

def header_label_row() -> Dict[str, Any]:
    """Fila 0 con los rótulos de subcolumnas dentro de cada bloque."""
    row = {k: "" for k in logical_schema()}
    row["model"] = "model"
    row["watermark model"] = "watermark model"
    row["train dataset"] = "train dataset"
    row["test dataset"] = "test dataset"
    row["bpp"] = "bpp"
    for c in SUBCOLS_ORIG:
        row[f"{ORIG_ORIG}:{c}"] = c
    for c in SUBCOLS_WM:
        row[f"{ORIG_WM}:{c}"] = c
    for c in SUBCOLS_WM:
        row[f"{WM_WM}:{c}"] = c
    return row

def block_title_row() -> Dict[str, Any]:
    """Fila 1 con los nombres de bloque replicados encima de las métricas."""
    row = {k: "" for k in logical_schema()}
    row["model"] = "model"
    row["watermark model"] = "watermark model"
    row["train dataset"] = "train dataset"
    row["test dataset"] = "test dataset"
    row["bpp"] = "bpp"
    for c in SUBCOLS_ORIG:
        row[f"{ORIG_ORIG}:{c}"] = ORIG_ORIG
    for c in SUBCOLS_WM:
        row[f"{ORIG_WM}:{c}"] = ORIG_WM
    for c in SUBCOLS_WM:
        row[f"{WM_WM}:{c}"] = WM_WM
    return row

def extract_row(json_path: Path, data: Dict[str, Any], dist_metric_filter: Optional[str]) -> Dict[str, Any]:
    row: Dict[str, Any] = {k: np.nan for k in logical_schema()}

    exp_name = infer_experiment_name(json_path) or ""
    wm_model = infer_watermark_model(json_path) or ""
    bpp = parse_bpp_from_tokens(exp_name, wm_model)
    model = infer_model_name(json_path, data) or ""
    test_ds = infer_test_dataset(json_path) or ""
    train_ds = infer_train_dataset(json_path) or ""
    dist_metric = infer_dist_metric(data) or ""

    # Filtro por métrica de distancia si se especifica
    if dist_metric_filter and dist_metric and dist_metric_filter.lower() != dist_metric.lower():
        return {}  # descartar

    row["model"] = model
    row["watermark model"] = wm_model
    row["train dataset"] = train_ds
    row["test dataset"] = test_ds
    row["bpp"] = bpp

    rec = data.get("recognition_metrics", {}) or {}
    avg = data.get("average_distances", {}) or {}
    std = data.get("std_distances", {}) or {}

    def gv(d: Dict[str, Any], key: str) -> Optional[float]:
        v = d.get(key)
        try:
            return float(v) if v is not None else np.nan
        except Exception:
            return np.nan

    prefix = f"{model}_" if model else ""
    dist_suffix = dist_metric if dist_metric else "cosine"  # por defecto cosine si no se infiere

    # ---- Reconocimiento: Baseline (Original - Original) ----
    row[f"{ORIG_ORIG}:EER"]                = gv(rec, f"{prefix}EER_baseline_{dist_suffix}")
    row[f"{ORIG_ORIG}:FAR at EER"]         = gv(rec, f"{prefix}FAR_at_EER_baseline_{dist_suffix}")
    row[f"{ORIG_ORIG}:FRR at EER"]         = gv(rec, f"{prefix}FRR_at_EER_baseline_{dist_suffix}")
    row[f"{ORIG_ORIG}:TAR at FAR ~0.01%"]  = gv(rec, f"{prefix}TAR_at_FAR_baseline_{dist_suffix}")
    row[f"{ORIG_ORIG}:AUC"]                = gv(rec, f"{prefix}AUC_baseline_{dist_suffix}")

    # ---- Original - Watermark (watermark) ----
    row[f"{ORIG_WM}:genuine avg % of change"]  = gv(avg, f"{prefix}avg_var_dist_{dist_suffix}_genuine_due_watermark")
    row[f"{ORIG_WM}:genuine std % of change"]  = gv(std, f"{prefix}std_var_dist_{dist_suffix}_genuine_due_watermark")
    row[f"{ORIG_WM}:impostor avg % of change"] = gv(avg, f"{prefix}avg_var_dist_{dist_suffix}_impostor_due_watermark")
    row[f"{ORIG_WM}:impostor std % of change"] = gv(std, f"{prefix}std_var_dist_{dist_suffix}_impostor_due_watermark")

    row[f"{ORIG_WM}:EER"]                   = gv(rec, f"{prefix}EER_watermarked_{dist_suffix}")
    row[f"{ORIG_WM}:FAR at EER"]            = gv(rec, f"{prefix}FAR_at_EER_watermarked_{dist_suffix}")
    row[f"{ORIG_WM}:FRR at EER"]            = gv(rec, f"{prefix}FRR_at_EER_watermarked_{dist_suffix}")
    row[f"{ORIG_WM}:TAR at FAR ~0.01%"]     = gv(rec, f"{prefix}TAR_at_FAR_watermarked_{dist_suffix}")
    row[f"{ORIG_WM}:AUC"]                   = gv(rec, f"{prefix}AUC_watermarked_{dist_suffix}")

    # ---- Watermark - Watermark (watermark both) ----
    row[f"{WM_WM}:genuine avg % of change"]  = gv(avg, f"{prefix}avg_var_dist_{dist_suffix}_genuine_due_watermark_both")
    row[f"{WM_WM}:genuine std % of change"]  = gv(std, f"{prefix}std_var_dist_{dist_suffix}_genuine_due_watermark_both")
    row[f"{WM_WM}:impostor avg % of change"] = gv(avg, f"{prefix}avg_var_dist_{dist_suffix}_impostor_due_watermark_both")
    row[f"{WM_WM}:impostor std % of change"] = gv(std, f"{prefix}std_var_dist_{dist_suffix}_impostor_due_watermark_both")

    row[f"{WM_WM}:EER"]                   = gv(rec, f"{prefix}EER_watermarked_both_{dist_suffix}")
    row[f"{WM_WM}:FAR at EER"]            = gv(rec, f"{prefix}FAR_at_EER_watermarked_both_{dist_suffix}")
    row[f"{WM_WM}:FRR at EER"]            = gv(rec, f"{prefix}FRR_at_EER_watermarked_both_{dist_suffix}")
    row[f"{WM_WM}:TAR at FAR ~0.01%"]     = gv(rec, f"{prefix}TAR_at_FAR_watermarked_both_{dist_suffix}")
    row[f"{WM_WM}:AUC"]                   = gv(rec, f"{prefix}AUC_watermarked_both_{dist_suffix}")

    return row

# ----------------- Recolección -----------------

def collect_json_files(roots: List[Path], mode: str) -> List[Path]:
    pattern = "results_summary_online.json" if mode == "online" else "results_summary.json"
    out: List[Path] = []
    for root in roots:
        out.extend(root.rglob(pattern))
    return out

# ----------------- Main -----------------

def main():
    parser = argparse.ArgumentParser(description="Consolidate results_summary*.json into a single Excel (no template reading)")
    parser.add_argument("--roots", nargs="+", required=False, help="Root folders to scan recursively", 
                        default=[r"C:\Users\JotaR\Documents\Github\WAFIS-ICAOeval\experiments\output\recognition"])
    parser.add_argument("--output", required=False, help="Output Excel path", 
                default=r"C:\Users\JotaR\Documents\Github\WAFIS-ICAOeval\evaluation\results\consolidated_results_face_recognition.xlsx")
    parser.add_argument("--mode", choices=["online","offline"], required=False, help="Select which summaries to consolidate",
                        default="online")
    parser.add_argument("--models", nargs="*", default=None, help="Filter by recognition models (e.g., facenet arcface). If omitted, include all found.")
    parser.add_argument("--algorithms", nargs="*", default=None, help="Filter by distance metrics (e.g., cosine euclidean). If omitted, include all found.")
    args = parser.parse_args()

    roots = [Path(p).resolve() for p in args.roots]
    json_files = collect_json_files(roots, args.mode)

    if not json_files:
        print("WARNING: No matching JSON files found. Exiting.")
        return

    rows = []
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"WARNING: Could not parse {jf}: {e}")
            continue

        # Filtrado por modelo
        inferred_model = infer_model_name(jf, data) or ""
        if args.models and inferred_model and (inferred_model.lower() not in [m.lower() for m in args.models]):
            continue

        # Filtrado por algoritmo (métrica de distancia)
        dist_metric = infer_dist_metric(data) or ""
        dist_filter = None
        if args.algorithms:
            algos_norm = [a.lower() for a in args.algorithms]
            if dist_metric and dist_metric.lower() not in algos_norm:
                continue
            if len(algos_norm) == 1:
                dist_filter = algos_norm[0]

        row = extract_row(jf, data, dist_filter)
        if row:
            rows.append(row)

    if not rows:
        print("WARNING: No rows after filtering. Exiting.")
        return

    df_data = pd.DataFrame(rows, columns=logical_schema())
    df_header1 = pd.DataFrame([header_label_row()])
    df_header2 = pd.DataFrame([block_title_row()])
    out_df = pd.concat([df_header1, df_header2, df_data], ignore_index=True)

    out_path = Path(args.output.replace('.xlsx', f'_{args.mode}.xlsx')).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        out_df.to_excel(writer, index=False, sheet_name="Hoja1")

    print(f"✅ Consolidation completed: {out_path}")

if __name__ == "__main__":
    main()
