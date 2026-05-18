#!/usr/bin/env python3
"""
Baseline L∞ sweep analysis v2 — ArcFace black-box runs
Generates:
  1. Pareto front (PSNR vs cos_sim) — white-box and black-box
  2. acc_after vs rec_weight per epsilon — CFD and FaceLab London
  3. Violin plots of similarity distributions (rec=0) — CFD and FaceLab London
  4. Threshold sweep (ASR vs threshold) — CFD and FaceLab London x both models
  5. ASR vs acc_after scatter — operational realism view
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

# ── Configuration ────────────────────────────────────────────────────
BASE_DIR = Path("../experiments/output/attacks/adversarial/attack_embeddings/"
                "stegaformer/1_1_255_w16_learn_im/celeba_hq/CFD/arcface/baseline")

THRESHOLDS = np.arange(0.05, 0.81, 0.01)
ASR_REF_TH  = 0.35   # reference threshold for Plot 5

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.dpi": 200,
})

ATTACK_TYPE = "l2"  
#WHITE_BOX = "facenet"
#BLACK_BOX = "arcface"

WHITE_BOX = "arcface"
BLACK_BOX = "facenet"

max_attack_capacity = False

if ATTACK_TYPE == "linf":
    EPS_COLORS = {2: "#3B8BD4", 4: "#1D9E75", 6: "#D85A30", 8: "#993556"}
    EPS_LABEL  = lambda e: f"ε={e}/255"
else:
    EPS_COLORS = {0.5: "#3B8BD4", 1: "#1D9E75", 2: "#D85A30", 3: "#993556"}
    EPS_LABEL  = lambda e: f"ε={e}.0"

REC_MARKERS = {0: "o", 1: "s", 10: "D", 50: "^"}
REC_COLORS  = {0: "#2c2c2a", 1: "#3B8BD4", 10: "#1D9E75", 50: "#D85A30"}

OUTPUT_DIR = Path(f"../evaluation/attacks/plots/baseline_{ATTACK_TYPE}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────────
def asr_at_threshold(sim_wm, sim_att, th):
    sim_wm  = np.array(sim_wm)
    sim_att = np.array(sim_att)
    recognized_wm = sim_wm > th
    successful    = (recognized_wm & (sim_att <= th)).sum()
    return (successful / recognized_wm.sum() * 100) if recognized_wm.sum() > 0 else 0


def load_runs(base_dir):
    runs = []
    for folder in sorted(base_dir.iterdir(),
                         key=lambda x: int(x.name) if x.name.isdigit() else 999):
        if not folder.is_dir() or not folder.name.isdigit():
            continue
        run = {"id": int(folder.name), "path": str(folder)}
        try:
            with open(folder / "hyperparameters.json")              as f: run["hparams"] = json.load(f)
            with open(folder / "train_results_last_epoch.json")     as f: run["cfd"]     = json.load(f)
            with open(folder / "test_results.json")                 as f: run["facelab"] = json.load(f)
            with open(folder / "all_ep0_face_recognition_results.json") as f: run["fr"] = json.load(f)
        except FileNotFoundError as e:
            print(f"  [SKIP] {folder.name}: {e}"); continue

        hp              = run["hparams"]
        run["type_attack"] = hp.get("type_attack", "linf")
        if run["type_attack"] == "linf":
            run["eps_int"]  = round(float(hp.get("epsilon",    0) * 255))
        else:
            run["eps_int"]  = float(hp.get("epsilon",    0))
        run["rec"]      = float(hp.get("rec_weight", 0))
        run["adv"]      = float(hp.get("adv_weight", 1))
        run["freq"]     = float(hp.get("freq_weight", 0))
        
        runs.append(run)
    return runs


def discover_fr_keys(fr_dict):
    """Auto-discover FR result keys, return (split_key, model) -> key mapping."""
    mapping = {}
    for k in fr_dict.keys():
        parts = k.rsplit("_", 1)
        if len(parts) == 2:
            mapping[tuple(parts)] = k
    return mapping


def display_name(key):
    """Map internal split key to display label."""
    if "train" in key.lower() or "cfd" in key.lower():
        return "CFD"
    if "test" in key.lower() or "facelab" in key.lower() or "london" in key.lower():
        return "FaceLab London"
    return key


def get_fr(run, fr_key_map, split_key, model):
    """Safe get of FR result dict for a split/model combo."""
    key = fr_key_map.get((split_key, model))
    if key and key in run["fr"]:
        return run["fr"][key]
    return None


# ── Load ─────────────────────────────────────────────────────────────
print("[*] Loading runs...")
runs = load_runs(BASE_DIR)
runs = [r for r in runs if r["type_attack"] == ATTACK_TYPE]
print(f"[*] Filtered to {len(runs)} runs with type_attack='{ATTACK_TYPE}'")

fr_key_map  = discover_fr_keys(runs[0]["fr"])
all_splits  = sorted({s for s, _ in fr_key_map})
all_models  = sorted({m for _, m in fr_key_map})
print(f"    FR splits : {all_splits}")
print(f"    FR models : {all_models}\n")

# Identify CFD (train) and FaceLab (test) split keys
SPLIT_CFD     = next((s for s in all_splits if "train" in s.lower() or "cfd"     in s.lower()), all_splits[0])
SPLIT_FACELAB = next((s for s in all_splits if "test"  in s.lower() or "facelab" in s.lower()), all_splits[-1])
SPLITS = [(SPLIT_CFD, "CFD"), (SPLIT_FACELAB, "FaceLab London")]

# Summary
print(f"{'ID':>3} | {'ε':>3} | {'rec':>5} | {'freq':>5} | "
      f"{'PSNR_CFD':>9} | {'acc_CFD':>8} | {'PSNR_FL':>8} | {'acc_FL':>7}")
print("-" * 70)
for r in runs:
    print(f"{r['id']:3d} | {r['eps_int']:5.2f} | {r['rec']:5.0f} | {r['freq']:5.1f} | "
          f"{r['cfd']['psnr']:9.2f} | {r['cfd']['acc_after']:8.4f} | "
          f"{r['facelab']['psnr']:8.2f} | {r['facelab']['acc_after']:7.4f}")


# ─────────────────────────────────────────────────────────────────────
# PLOT 1: Pareto front (PSNR vs cos_sim)
# ─────────────────────────────────────────────────────────────────────
print("\n[*] Plot 1: Pareto front...")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

for row, (split_key, split_label) in enumerate(SPLITS):
    for col, (model_label, model_name) in enumerate([
        (f"White-box ({WHITE_BOX})", WHITE_BOX),
        (f"Black-box ({BLACK_BOX})",  BLACK_BOX),
    ]):
        ax = axes[row][col]
        results_key = "cfd" if row == 0 else "facelab"

        for r in runs:
            psnr = r[results_key]["psnr"]
            fr   = get_fr(r, fr_key_map, split_key, model_name)
            sim  = fr["avg_sim_attacked"] if fr else r[results_key]["cos_sim"]
            ax.scatter(psnr, sim,
                       c=EPS_COLORS.get(r["eps_int"], "gray"),
                       marker=REC_MARKERS.get(int(r["rec"]), "o"),
                       s=100, edgecolors="black", linewidths=0.5, zorder=3)

        for th, ls in [(0.2, ":"), (0.3, "--"), (0.4, ":")]:
            ax.axhline(th, color="gray", ls=ls, lw=0.8, alpha=0.5)
            ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] > 30 else 40,
                    th + 0.01, f"th={th}", fontsize=8, color="gray")

        ax.set_xlabel("PSNR (dB)")
        ax.set_ylabel("Cosine similarity (post-attack)")
        ax.set_title(f"{split_label} — {model_label}")
        ax.grid(True, alpha=0.2)

# Legends
for eps, c in EPS_COLORS.items():
    axes[0][0].scatter([], [], c=c, s=60, label=EPS_LABEL(eps),
                       edgecolors="black", linewidths=0.5)
for rec, mk in REC_MARKERS.items():
    axes[0][0].scatter([], [], c="gray", marker=mk, s=60,
                       label=f"rec={rec}", edgecolors="black", linewidths=0.5)
axes[0][0].legend(loc="upper left", ncol=2, framealpha=0.9)

fig.suptitle("Pareto front: PSNR vs cosine similarity", fontsize=13)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "plot1_pareto_front.png")
print(f"    Saved: plot1_pareto_front.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────
# PLOT 2: acc_after vs rec_weight — CFD and FaceLab London
# ─────────────────────────────────────────────────────────────────────
print("[*] Plot 2: acc_after vs rec_weight...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax_idx, (split_label, results_key) in enumerate([("CFD", "cfd"), ("FaceLab London", "facelab")]):
    ax = axes[ax_idx]
    eps_groups = defaultdict(list)
    for r in runs:
        eps_groups[r["eps_int"]].append(r)

    for eps in sorted(eps_groups.keys()):
        group = sorted(eps_groups[eps], key=lambda x: x["rec"])
        recs  = [r["rec"] for r in group]
        accs  = [r[results_key]["acc_after"] for r in group]
        ax.plot(recs, accs, "o-", color=EPS_COLORS[eps],
                label=EPS_LABEL(eps), markersize=7, linewidth=1.5)

    ax.set_xlabel("rec_weight")
    ax.set_ylabel("Watermark bit accuracy (acc_after)")
    ax.set_title(split_label)
    ax.set_xticks([0, 1, 10, 50])
    ax.grid(True, alpha=0.2)
    ax.legend()

fig.suptitle("Watermark preservation vs rec_weight", fontsize=13)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "plot2_acc_after_vs_rec.png")
print(f"    Saved: plot2_acc_after_vs_rec.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────
# PLOT 3: Violin distributions (rec=0) — CFD and FaceLab London
# ─────────────────────────────────────────────────────────────────────
print("[*] Plot 3: Violin distributions...")

runs_rec0 = sorted([r for r in runs if r["rec"] == 0], key=lambda x: x["eps_int"])
runs_rec1 = sorted([r for r in runs if r["rec"] == 1], key=lambda x: x["eps_int"])

if max_attack_capacity:
    runs_rec = runs_rec0
else:
    runs_rec = runs_rec1


if runs_rec:
    n_eps  = len(runs_rec)
    n_rows = 4   # CFD-WB, CFD-BB, FaceLab-WB, FaceLab-BB
    row_labels = [
        f"CFD — White-box ({WHITE_BOX})",
        f"CFD — Black-box ({BLACK_BOX})",
        f"FaceLab — White-box ({WHITE_BOX})",
        f"FaceLab — Black-box ({BLACK_BOX})",
    ]
    row_configs = [
        (SPLIT_CFD,     WHITE_BOX),
        (SPLIT_CFD,     BLACK_BOX),
        (SPLIT_FACELAB, WHITE_BOX),
        (SPLIT_FACELAB, BLACK_BOX),
    ]

    fig, axes = plt.subplots(n_rows, n_eps, figsize=(3.5 * n_eps, 3.5 * n_rows), squeeze=False)

    for col, r in enumerate(runs_rec):
        for row, (split_key, model_name) in enumerate(row_configs):
            ax  = axes[row][col]
            fr  = get_fr(r, fr_key_map, split_key, model_name)

            if fr:
                sim_wm  = np.array(fr["all_sim_wm"])
                sim_att = np.array(fr["all_sim_attacked"])

                vp_wm  = ax.violinplot([sim_wm],  positions=[0], showmeans=True)
                vp_att = ax.violinplot([sim_att], positions=[1], showmeans=True)
                for pc in vp_wm["bodies"]:
                    pc.set_facecolor("#3B8BD4"); pc.set_alpha(0.65)
                for pc in vp_att["bodies"]:
                    pc.set_facecolor("#D85A30"); pc.set_alpha(0.65)

                for th in [0.2, 0.3, 0.4]:
                    ax.axhline(th, color="gray", ls="--", lw=0.7, alpha=0.4)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, color="gray")

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Before\n(wm)", "After\n(att)"], fontsize=8)
            ax.set_title(f"ε={r['eps_int']}" if row == 0 else "", fontsize=10)
            if col == 0:
                ax.set_ylabel(row_labels[row] + "\nCosine sim", fontsize=8)
            ax.grid(True, alpha=0.15, axis="y")

    rec_label = "0" if max_attack_capacity else "1"
    fig.suptitle(f"Similarity distributions before/after attack (rec={rec_label})", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot3_violin_distributions.png")
    print(f"    Saved: plot3_violin_distributions.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────
# PLOT 4: Threshold sweep (ASR vs threshold) — CFD and FaceLab London
# ─────────────────────────────────────────────────────────────────────
print("[*] Plot 4: Threshold sweep...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for row, (split_key, split_label) in enumerate(SPLITS):
    for col, (model_name, model_label) in enumerate([
        (WHITE_BOX, f"White-box ({WHITE_BOX})"),
        (BLACK_BOX, f"Black-box ({BLACK_BOX})"),
    ]):
        ax = axes[row][col]

        for r in runs_rec:
            fr = get_fr(r, fr_key_map, split_key, model_name)
            if fr:
                asr_curve = [asr_at_threshold(fr["all_sim_wm"], fr["all_sim_attacked"], th)
                             for th in THRESHOLDS]
                ax.plot(THRESHOLDS, asr_curve,
                        color=EPS_COLORS[r["eps_int"]], lw=2,
                        label=EPS_LABEL(r["eps_int"]))

        ax.axvline(ASR_REF_TH, color="black", ls=":", lw=1, alpha=0.5,
                   label=f"th={ASR_REF_TH}")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("ASR (%)")
        ax.set_title(f"{split_label} — {model_label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(-5, 105)

rec_label = "0" if max_attack_capacity else "1"
fig.suptitle(f"ASR vs threshold (rec={rec_label}, varying ε)", fontsize=13)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "plot4_threshold_sweep.png")
print(f"    Saved: plot4_threshold_sweep.png")
plt.close()


#─────────────────────────────────────────────────────────────────────
# PLOT 5: ASR vs acc_after — operational realism (rec=1 only)
# ─────────────────────────────────────────────────────────────────────
print("[*] Plot 5: ASR vs acc_after (rec=1, operational realism)...")

runs_rec1 = [r for r in runs if r["rec"] == 1]

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

for row, (model_name, model_label) in enumerate([
    (WHITE_BOX, f"White-box ({WHITE_BOX})"),
    (BLACK_BOX, f"Black-box ({BLACK_BOX})"),
]):
    for col, (split_key, split_label, results_key) in enumerate([
        (SPLIT_CFD,     "CFD",            "cfd"),
        (SPLIT_FACELAB, "FaceLab London", "facelab"),
    ]):
        ax = axes[row][col]

        for r in runs_rec1:
            fr = get_fr(r, fr_key_map, split_key, model_name)
            if fr is None:
                continue
            asr = asr_at_threshold(fr["all_sim_wm"], fr["all_sim_attacked"], ASR_REF_TH)
            acc = r[results_key]["acc_after"]

            ax.scatter(asr, acc,
                       c=EPS_COLORS.get(r["eps_int"], "gray"),
                       marker="s", s=160,
                       edgecolors="black", linewidths=0.7, zorder=3)
            ax.annotate(EPS_LABEL(r['eps_int']),
                        xy=(asr, acc), xytext=(5, 5),
                        textcoords="offset points", fontsize=8, color="gray")

        # Ideal zone
        ax.axvline(50,   color="green", ls="--", lw=0.8, alpha=0.4)
        ax.axhline(0.90, color="green", ls="--", lw=0.8, alpha=0.4)
        ax.fill_between([50, 105], 0.90, 1.01,
                        color="green", alpha=0.05, label="Ideal zone")

        ax.set_xlabel(f"ASR at threshold={ASR_REF_TH} (%)")
        ax.set_ylabel("Watermark bit accuracy (acc_after)")
        ax.set_title(f"{split_label} — {model_label}")
        ax.set_xlim(-5, 105)
        ax.set_ylim(0.80, 1.01)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)

# Epsilon legend
for eps, c in EPS_COLORS.items():
    axes[0][0].scatter([], [], c=c, marker="s", s=80,
                       label=EPS_LABEL(eps), edgecolors="black", linewidths=0.5)
axes[0][0].legend(loc="lower left", framealpha=0.9)

fig.suptitle(f"Operational realism: ASR vs watermark preservation (rec=1, th={ASR_REF_TH})", fontsize=13)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "plot5_asr_vs_acc_after.png")
print(f"    Saved: plot5_asr_vs_acc_after.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────
# Threshold sweep JSON
# ─────────────────────────────────────────────────────────────────────
print("[*] Saving threshold sweep JSON...")
sweep = {}
for r in runs:
    key = f"id{r['id']}_eps{r['eps_int']}_rec{int(r['rec'])}"
    entry = {
        "run_id": r["id"], "eps_int": r["eps_int"],
        "rec_weight": r["rec"], "adv_weight": r["adv"],
        "psnr_cfd": r["cfd"]["psnr"],
        "acc_after_cfd": r["cfd"]["acc_after"],
        "psnr_facelab": r["facelab"]["psnr"],
        "acc_after_facelab": r["facelab"]["acc_after"],
        "thresholds": THRESHOLDS.tolist(),
    }
    for (split_key, split_label) in SPLITS:
        for model_name in [WHITE_BOX, BLACK_BOX]:
            fr = get_fr(r, fr_key_map, split_key, model_name)
            if fr:
                label = f"asr_{split_label.replace(' ', '_')}_{model_name}"
                entry[label] = [asr_at_threshold(fr["all_sim_wm"],
                                                 fr["all_sim_attacked"], th)
                                for th in THRESHOLDS]
    sweep[key] = entry

with open(OUTPUT_DIR / "threshold_sweep_results.json", "w") as f:
    json.dump(sweep, f, indent=2)
print(f"    Saved: threshold_sweep_results.json")

print("\n[*] Done. Files saved to /app/")
print("    plot1_pareto_front.png")
print("    plot2_acc_after_vs_rec.png")
print("    plot3_violin_distributions.png")
print("    plot4_threshold_sweep.png")
print("    plot5_asr_vs_acc_after.png")
print("    threshold_sweep_results.json")
