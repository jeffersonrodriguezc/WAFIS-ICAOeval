
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


METRIC_ROWS = [
    ("EER", "EER"),
    ("FAR", "FAR at EER"),
    ("FRR", "FRR at EER"),
    ("TAR", "TAR at FAR ~0.01%"),
    ("AUC", "AUC"),
]
DISTANCE_ROW_NAME = "Distances"

TRACE_PREFIXES_3 = [
    "Original - Original",
    "Original - Watermark",
    "Watermark - Watermark",
]

TRACE_PREFIXES_2 = [
    "Original - Watermark",
    "Watermark - Watermark",
]

EXPECTED_MAIN_COLS = {
    "recognizer": "model",
    "watermark": "watermark model",
    "train_dataset": "train dataset",
    "test_dataset": "test dataset",
    "bpp": "bpp",
}

def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in EXPECTED_MAIN_COLS.values():
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in the Excel sheet.")
    def _is_bad_row(r):
        try:
            int(r[EXPECTED_MAIN_COLS["bpp"]])
            return False
        except Exception:
            return True
    mask_bad = df.apply(_is_bad_row, axis=1)
    df = df[~mask_bad].copy()
    df[EXPECTED_MAIN_COLS["bpp"]] = df[EXPECTED_MAIN_COLS["bpp"]].astype(int)
    for k in ("recognizer", "watermark", "train_dataset", "test_dataset"):
        col = EXPECTED_MAIN_COLS[k]
        df[col] = df[col].astype(str).str.strip()
    return df

def _subset(df: pd.DataFrame, recognizer: str, watermark: str, train_dataset: str) -> pd.DataFrame:
    m = (
        (df[EXPECTED_MAIN_COLS["recognizer"]].str.lower() == recognizer.lower()) &
        (df[EXPECTED_MAIN_COLS["watermark"]].str.lower() == watermark.lower()) &
        (df[EXPECTED_MAIN_COLS["train_dataset"]].str.lower() == train_dataset.lower())
    )
    sub = df[m].copy()
    if sub.empty:
        raise ValueError(
            f"No rows found for recognizer='{recognizer}', watermark='{watermark}', train_dataset='{train_dataset}'"
        )
    return sub

def _get_test_datasets(df: pd.DataFrame, max_cols: int = 4) -> list:
    tests = sorted([t for t in df[EXPECTED_MAIN_COLS["test_dataset"]].unique().tolist() if t and t.lower() != "test dataset"])
    if not tests:
        raise ValueError("No valid test datasets found in the filtered data.")
    if len(tests) > max_cols:
        tests = tests[:max_cols]
    return tests

def _available_columns(df: pd.DataFrame) -> set:
    return set(map(str, df.columns))

def _column_for_metric(trace_prefix: str, metric_label: str) -> str:
    return f"{trace_prefix}:{metric_label}"

def _column_for_distance(trace_prefix: str, distance_kind: str) -> str:
    return f"{trace_prefix}:{distance_kind} avg % of change"

ALLOWED_BPPS = {1, 3, 6, 8}

def _filter_bpps(x: np.ndarray, y: np.ndarray):
    if x.size == 0:
        return x, y
    mask = np.isin(x, list(ALLOWED_BPPS))
    return x[mask], y[mask]

ALLOWED_BPPS = {1, 3, 6, 8}

def _filter_bpps(x: np.ndarray, y: np.ndarray):
    if x.size == 0:
        return x, y
    mask = np.isin(x, list(ALLOWED_BPPS))
    return x[mask], y[mask]

def _extract_xy(df: pd.DataFrame, test_dataset: str, y_col: str) -> tuple:
    dsub = df[df[EXPECTED_MAIN_COLS["test_dataset"]].str.lower() == test_dataset.lower()].copy()
    if dsub.empty:
        return np.array([]), np.array([])
    dsub = dsub.sort_values(by=EXPECTED_MAIN_COLS["bpp"])
    x = dsub[EXPECTED_MAIN_COLS["bpp"]].to_numpy()
    y = pd.to_numeric(dsub.get(y_col, np.nan), errors="coerce").to_numpy()
    # keep only the requested BPPs while preserving numeric spacing
    x, y = _filter_bpps(x, y)
    x, y = _filter_bpps(x, y)
    return x, y

def _style_mpl(ax, title: str):
    ax.set_title(title, fontsize=11, pad=6, loc="center")
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.15)
    ax.tick_params(axis="both", labelsize=9)
    ax.minorticks_on()

def _ensure_outdirs(base: Path, recognizer: str, watermark: str, train_dataset: str):
    (base / watermark / "plots" / train_dataset / recognizer / "individual").mkdir(parents=True, exist_ok=True)
    
def plot_all(
    excel_path: str,
    mode: str,
    recognizer: str,
    watermark: str,
    train_dataset: str,
    outdir: str,
    max_cols: int = 6,
) -> dict:
    excel_path = Path(excel_path.replace("_.xlsx", f"_{mode}.xlsx"))
    outdir = Path(outdir)
    _ensure_outdirs(outdir, recognizer, watermark, train_dataset)

    xls = pd.ExcelFile(excel_path)
    sheet = xls.sheet_names[0]
    df = pd.read_excel(excel_path, sheet_name=sheet)
    df = _clean_dataframe(df)
    df_f = _subset(df, recognizer, watermark, train_dataset)

    test_datasets = _get_test_datasets(df_f, max_cols=max_cols)
    cols_avail = _available_columns(df_f)

    sns.set_theme(style="whitegrid", context="talk")
    width = 20
    height = 22
    fig, axes = plt.subplots(
        nrows=6, ncols=len(test_datasets),
        figsize=(width, height),
        sharex=True
    )
    if len(test_datasets) == 1:
        axes = np.atleast_2d(axes).reshape(6, 1)

    color_map = {
        "Original - Original": "#4C78A8",
        "Original - Watermark": "#F58518",
        "Watermark - Watermark": "#54A24B",
    }
    marker_map = {
        "Original - Original": "o",
        "Original - Watermark": "s",
        "Watermark - Watermark": "D",
    }

    for c, test_ds in enumerate(test_datasets):
        for r, (row_name, metric_label) in enumerate(METRIC_ROWS, start=0):
            ax = axes[r, c]
            for trace in TRACE_PREFIXES_3:
                colname = _column_for_metric(trace, metric_label)
                if colname not in cols_avail:
                    continue
                x, y = _extract_xy(df_f, test_ds, colname)
                if x.size == 0:
                    continue
                ax.plot(
                    x, y,
                    label=trace,
                    linewidth=2.0,
                    marker=marker_map.get(trace, "o"),
                    markersize=5,
                    color=color_map.get(trace, None),
                )
            _style_mpl(ax, f"{metric_label} — {test_ds}")
            ax.set_xticks([1, 3, 6, 8])
            if r == len(METRIC_ROWS)-1:
                ax.set_xlabel("bpp", fontsize=10)
            if c == 0:
                ax.set_ylabel(metric_label, fontsize=11)
            if r == 0:
                ax.legend(loc='upper center',
                    bbox_to_anchor=(0.5, 1.55),  # arriba, centrado
                    ncol=1,
                    fontsize=9,
                    frameon=True)

        axd = axes[5, c]
        dk_markers = {"genuine": "^", "impostor": "X"}
        for trace in TRACE_PREFIXES_2:
            for dk in ("genuine", "impostor"):
                colname = _column_for_distance(trace, dk)
                if colname not in cols_avail:
                    continue
                x, y = _extract_xy(df_f, test_ds, colname)
                if x.size == 0:
                    continue
                axd.plot(
                    x, y,
                    label=f"{trace} — {dk}",
                    linewidth=2.0,
                    marker=dk_markers.get(dk, "o"),
                    markersize=5,
                    color=color_map.get(trace, None),
                )
        _style_mpl(axd, f"{DISTANCE_ROW_NAME} — {test_ds}")
        axd.set_xticks([1, 3, 6, 8])
        axd.set_xlabel("bpp", fontsize=10)
        if c == 0:
            axd.set_ylabel("avg % change", fontsize=11)
        axd.legend( loc='lower center',
    bbox_to_anchor=(0.5, -0.75),
                    ncol=1,
                    fontsize=9,
                    frameon=True)

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    title = f"{recognizer} × {watermark} — train: {train_dataset}"
    fig.suptitle(title, fontsize=16, y=0.995)

    static_full_png = outdir / watermark / "plots" / train_dataset / recognizer / f"grid_{recognizer}_{watermark}_{train_dataset}_{mode}.png"
    static_full_pdf = outdir / watermark / "plots" / train_dataset / recognizer / f"grid_{recognizer}_{watermark}_{train_dataset}_{mode}.pdf"
    fig.savefig(static_full_png, dpi=300, bbox_inches="tight")
    fig.savefig(static_full_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    for test_ds in test_datasets:
        for row_name, metric_label in METRIC_ROWS:
            fig_i, ax_i = plt.subplots(figsize=(6, 4.2))
            for trace in TRACE_PREFIXES_3:
                colname = _column_for_metric(trace, metric_label)
                if colname not in cols_avail:
                    continue
                x, y = _extract_xy(df_f, test_ds, colname)
                if x.size == 0:
                    continue
                ax_i.plot(
                    x, y,
                    label=trace,
                    linewidth=2.2,
                    marker=marker_map.get(trace, "o"),
                    markersize=6,
                    color=color_map.get(trace, None),
                )
            _style_mpl(ax_i, f"{metric_label} — {test_ds}")
            ax_i.set_xticks([1, 3, 6, 8])
            ax_i.set_xlabel("bpp", fontsize=10)
            ax_i.set_ylabel(metric_label, fontsize=11)
            ax_i.legend(fontsize=9, loc="best", frameon=True)
            plt.tight_layout()
            outp = outdir / watermark / "plots" / train_dataset / recognizer / "individual" / f"{test_ds}_{row_name}_{recognizer}_{watermark}_{train_dataset}_{mode}.png"
            fig_i.savefig(outp, dpi=300, bbox_inches="tight")
            plt.close(fig_i)

        fig_d, ax_d = plt.subplots(figsize=(6, 4.2))
        dk_markers = {"genuine": "^", "impostor": "X"}
        for trace in TRACE_PREFIXES_2:
            for dk in ("genuine", "impostor"):
                colname = _column_for_distance(trace, dk)
                if colname not in cols_avail:
                    continue
                x, y = _extract_xy(df_f, test_ds, colname)
                if x.size == 0:
                    continue
                ax_d.plot(
                    x, y,
                    label=f"{trace} — {dk}",
                    linewidth=2.2,
                    marker=dk_markers.get(dk, "o"),
                    markersize=6,
                    color=color_map.get(trace, None),
                )
        _style_mpl(ax_d, f"{DISTANCE_ROW_NAME} — {test_ds}")
        ax_d.set_xticks([1, 3, 6, 8])
        ax_d.set_xlabel("bpp", fontsize=10)
        ax_d.set_ylabel("avg % change", fontsize=11)
        ax_d.legend(fontsize=9, loc="best", frameon=True)
        plt.tight_layout()
        outp = outdir / watermark / "plots" / train_dataset / recognizer / "individual" / f"{test_ds}_{DISTANCE_ROW_NAME}_{recognizer}_{watermark}_{train_dataset}_{mode}.png"
        fig_d.savefig(outp, dpi=300, bbox_inches="tight")
        plt.close(fig_d)

    return {
        "static_full_png": str(static_full_png),
        "static_full_pdf": str(static_full_pdf),
        "individual_dir": str(outdir / watermark / "plots" / train_dataset / recognizer / "individual"),
        "tests_used": test_datasets,
    }

def main():
    ap = argparse.ArgumentParser(description="Generate recognition plots (static + interactive) from Excel.")
    ap.add_argument("--excel", required=False, help="Path to consolidated_results_face_recognition excel file",
                    default=r"C:\Users\JotaR\Documents\Github\WAFIS-ICAOeval\evaluation\results\consolidated_results_face_recognition_.xlsx")
    ap.add_argument("--mode", default="offline")
    ap.add_argument("--watermark", required=False, help="Watermarking algorithm (e.g., 'stegaformer')",
                    default="stegaformer")
    ap.add_argument("--recognizer", required=False, help="Face recognition algorithm (e.g., 'facenet')",
                    default="facenet")
    ap.add_argument("--train_dataset", required=False, help="Training dataset name (e.g., 'coco')",
                    default="coco")
    ap.add_argument("--outdir", default=r"C:\Users\JotaR\Documents\Github\WAFIS-ICAOeval\evaluation\visualizations", 
                    help="Output directory base")
    args = ap.parse_args()

    paths = plot_all(
        excel_path=args.excel,
        mode=args.mode,
        recognizer=args.recognizer,
        watermark=args.watermark,
        train_dataset=args.train_dataset,
        outdir=args.outdir,
    )
    print("Generated files:")
    for k, v in paths.items():
        print(f" - {k}: {v}")
    print("Done.")

if __name__ == "__main__":
    main()
