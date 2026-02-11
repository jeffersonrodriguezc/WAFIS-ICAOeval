# Guideline to read the repo (Branch adversarial-attacl-poc)

## 1) Repository Structure:
```text
modules/attacks/adversarial/attack_embeddings/

├── workflow.py                  # Main orchestrator (train, validate, test, save results)
├── criteria/
│   └── loss_functions.py        # L_adv, L_rec (MSE/LPIPS/Cosine)
├── network/
│   ├── AAD.py                   # AAD Generator + FusionModule + Gaussian spatial mask
│   ├── MAE.py                   # Attribute Encoder (MLAttrEncoder)
│   └── face_modules.py          # Face recognition backbone (ArcFace)
├── weights/                     # Pretrained models weights 
├── utils.py                     # PGD steps/projections (L2 and L∞), normalizations, and image utilities
├── dataset.py                   # Datasets for attacks and face recognition evaluation
├── options/options.py           # Experiment hyperparameters (attack, losses, paths, dataloaders)
└── requirements.txt
```

## 2) Mapping Course Concepts:

## 2.1 Objective Formulation (Total Loss)

**Key File:** `criteria/loss_functions.py`

- `AdvLoss`: Implements the adversarial component using `CosineEmbeddingLoss`.
  - In `evasion` mode, it uses a target of `-1` to minimize similarity with the reference identity.
- `RecLoss`: Implements the reconstruction term.
  - Supports `l2` (MSE), `lpips` (perceptual), and `combined`.

**Where they are combined in practice:** `workflow.py` → `attack_batch_pipeline(...)`

- The loss is calculated as `loss = (adv_weight * ladv ) + (rec_weight * lrec ) + (mask_reg * mean(mask))`.

---

## 2.2 Adversarial Optimization (PGD)

**Optimization Step in Embeddings (L2):**

- `utils.py`
  - `pgd_step(...)`: PGD update with normalized gradient.
  - `l2_project(...)`: Projection to the L2 feasible set (`||delta||_2 <= epsilon`).

**Direct Usage in the Main Flow:**

- `workflow.py` → `attack_batch_pipeline(...)`
  - Initializes `delta` in the embedding.
  - Iterates through `pgd_steps`.
  - Updates `delta` and projects it in each iteration.

**Comparative Baseline (Pixel + L∞):**

- `workflow.py` → `attack_batch_baseline_linf(...)`
- `utils.py` → `pgd_step_linf(...)`, `linf_project(...)`

This allows for a comparison between "latent space attack" vs. "direct pixel space attack."

---

## 2.3 Network Pipeline (Attack Architecture)

### Networks and Conceptual Roles

1. **FaceNet (ArcFace backbone)**
   - Extracts identity embeddings (`z_id`, `z_adv`) to measure evasion.
2. **Attribute Encoder (`MLAttrEncoder`)**
   - Extracts multi-scale attributes (`z_att`) that preserve structure/non-identity features.
3. **AAD Generator (`AADGenerator`)**
   - Generates the adversarial image conditioned by `z_att` and the perturbed embedding (`z_id + delta`).
4. **Fusion Module (`FusionModule`, optional)**
   - Learns a mask to merge the original and adversarial images.
   - Allows for regularization of the modified region.

### Implementation by File

- `network/MAE.py` → `MLAttrEncoder`
- `network/AAD.py` → `AADGenerator`, `FusionModule`, `get_spatial_weights_gauss`
- `network/face_modules.py` → Face recognition backbone
- `workflow.py` (`__init__`) → Loading, freezing, and orchestration of modules

**Notes:**
- 1. The watermarking network is imported from other module in this repository.
- 2. The weights for these architectures are loaded from a folder that is located locally.
---

## 3) Guided Reading of `workflow.py` 

`workflow.py` contains the entire execution logic. It is recommended to read in this order:

1. **`AttackEmbeddings.__init__(...)`**
   - Loads pre-trained networks.
   - Freezes base components and prepares losses/metrics.
2. **`attack_batch_pipeline(...)`** ← Core of the method
   - Implements the embedding attack + PGD + loss composition.
3. **`attack_batch_baseline_linf(...)`**
   - Pixel-based baseline for methodological comparison.
4. **`run_attack(tag)`**
   - Main loop (train/val/test), logging, and metrics reporting.
5. **`run_eval_face_recognition(...)`**
   - Evaluates the impact of the attack on face recognition similarity.

---

## 4) Metrics for Result Discussion

The module reports relevant metrics to validate the attack trade-off:

- **Cosine similarity** of embeddings (effect on identity).
- **PSNR / SSIM** (visual quality).
- **Bit/message accuracy of watermark** (integrity of hidden content).
- **Attack Success Rate (ASR)** to measure the number of successful evasion attacks (before the attack they were recognized as the same person and after they were not)

These metrics are calculated and recorded in `run_attack(...)`, and the face recognition evaluation is concentrated in `run_eval_face_recognition(...)`.

## 5) Critical Hyperparameters for Experimental Analysis

**File:** `options/options.py`

For sensitivity analysis and methodological justification, the following are the most important parameters:

- **Attack:** `pgd_steps`, `epsilon`, `step_size`.
- **Losses:** `adv_weight`, `rec_weight`, `mse_weight`, `lpips_weight`, `mask_reg`.
- **Architecture:** `use_fusion_module`, `use_weight_mask`.

These parameters directly control the balance between:
**evasion success** ↔ **visual fidelity** ↔ **information preservation**.
