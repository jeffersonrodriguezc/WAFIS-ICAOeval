# run_baseline_sweep.py
import subprocess
import json
from pathlib import Path
from datetime import datetime

BASE = [
    "python", "workflow.py",
    "--baseline", "True",
    "--pgd_steps", "180",
    "--dataset", "CFD",
    "--dataset_test", "facelab_london",
    "--loss_mode", "l1",
    "--freq_weight", "1.0",
]

EPSILONS = [0.5, 1, 2, 3]
#REC_WEIGHTS = [0, 1, 10, 50]
REC_WEIGHTS = [1]
k = 180

manifest = []
run_id = 21
type_attack = "l2" #linf or l2

print(f"[*] Baseline {type_attack} sweep: {len(EPSILONS)} ε × {len(REC_WEIGHTS)} rec × 2 models = {len(EPSILONS) * len(REC_WEIGHTS) * 2} runs")
print(f"[*] Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ── FaceNet white-box ──────────────────────────────────────────
for eps_int in EPSILONS:
    for rec_weight in REC_WEIGHTS:
        if type_attack == "linf":
            eps = eps_int / 255.0
        else:
            eps = eps_int
        
        if type_attack == "linf":
            step = 0.001
        else:
            step = eps * 15 / k

        
        tag = f"{type_attack}_fn_eps{eps_int}_rec{rec_weight}"
        
        cmd = BASE + [
            "--epsilon", f"{eps:.8f}",
            "--step_size", f"{step:.8f}",
            "--adv_weight", "1.0",
            "--rec_weight", f"{rec_weight}.0",
            "--facenet_mode", "facenet",
            "--facenet_mode_test", "arcface",
        ]
        
        print(f"[{run_id:02d}] Launching {tag}")
        subprocess.run(cmd, check=True)
        
        manifest.append({
            "run_id": run_id,
            "tag": tag,
            "norm": type_attack,
            "white_box": "facenet",
            "black_box": "arcface",
            "eps": eps_int,
            "adv_weight": 1.0,
            "rec_weight": float(rec_weight),
            "step_size": step,
            "pgd_steps": k
        })
        run_id += 1

# Guardar manifest
date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
output_path = Path(f"baseline_{type_attack}_manifest_facenet_{date}.json")

print(f"\n[*] Completed {run_id} runs")
print(f"[*] Manifest saved to {output_path}")
print(f"[*] Finished at {date}")

with output_path.open("w") as f:
    json.dump(manifest, f, indent=2)

manifest = []
run_id = 9
type_attack = "l2"

# ── ArcFace white-box ──────────────────────────────────────────
for eps_int in EPSILONS:
    for rec_weight in REC_WEIGHTS:
        if type_attack == "linf":
            eps = eps_int / 255.0
        else:
            eps = eps_int

        if type_attack == "linf":
            step = 0.001
        else:
            step = eps * 15 / k 
        
        tag = f"{type_attack}_arc_eps{eps_int}_rec{rec_weight}"
        
        cmd = BASE + [
            "--epsilon", f"{eps:.8f}",
            "--step_size", f"{step:.8f}",
            "--adv_weight", "1.0",
            "--rec_weight", f"{rec_weight}.0",
            "--facenet_mode", "arcface",
            "--facenet_dir", "./weights/arcface/ms1mv3_arcface_r100_fp16_backbone.pth",
            "--facenet_mode_test", "facenet",
        ]
        
        print(f"[{run_id:02d}] Launching {tag}")
        subprocess.run(cmd, check=True)
        
        manifest.append({
            "run_id": run_id,
            "tag": tag,
            "norm": type_attack,
            "white_box": "arcface",
            "black_box": "facenet",
            "eps": eps_int,
            "adv_weight": 1.0,
            "rec_weight": float(rec_weight),
            "step_size": step,
            "pgd_steps": k
        })
        run_id += 1

# Guardar manifest
date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
output_path = Path(f"baseline_{type_attack}_manifest_arcface_{date}.json")

print(f"\n[*] Completed {run_id} runs")
print(f"[*] Manifest saved to {output_path}")
print(f"[*] Finished at {date}")

with output_path.open("w") as f:
    json.dump(manifest, f, indent=2)

