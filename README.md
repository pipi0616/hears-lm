# HEARS-LM

**Hereditary EAR Sequence Language Model** — a hearing-loss–specific protein language model for missense variant pathogenicity, thermodynamics, mechanism, and clinical interpretation.

> A hearing-loss–specific protein language model learns transferable variant biophysics, aligns with expert clinical decisions, and prospectively predicts ClinVar reclassifications. Preprint: forthcoming.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## Overview

Hearing loss is the most common sensory disability worldwide, and a significant fraction of cases have a genetic basis — yet translating a genetic finding into a clinical action remains hard. The central bottleneck is the **variant of uncertain significance (VUS)**: a missense change detected in a patient whose effect on protein function is unknown. Clinicians cannot confidently counsel families, guide treatment decisions, or determine eligibility for emerging gene therapies (such as the active DB-OTO programme for *OTOF*, NCT05788536) without knowing whether a VUS is truly pathogenic.

General-purpose variant effect predictors such as AlphaMissense were trained on the broad proteome. They lack the domain-specific supervision — curated hearing-loss variant databases, gene-therapy target landscapes, clinical expert panel decisions — needed to resolve the hard cases in audiology.

**HEARS-LM** is a protein language model built specifically for the hearing-loss domain. It couples a domain-adapted 30-layer ESM-2 backbone with one hundred curated hearing-loss–specific biological descriptors through cross-attention fusion. Four task-specific heads are trained jointly:

- **Pathogenicity** — variant-level binary classification
- **Thermodynamic ΔΔG** — predicted folding free-energy change
- **Mechanism** — mechanism-of-disease probability vector
- **Ordinal severity** — predicted clinical severity grade

A thermodynamic auxiliary objective, supervised by FoldX folding free-energy estimates, allows biophysical signal to propagate across all heads and generalize to unseen genes.

---

## Why HEARS-LM?

| Capability | HEARS-LM | AlphaMissense (general) |
|---|---|---|
| Domain-specific training data | Hearing-loss curated variants | Broad proteome |
| Expert panel alignment (HL-VCEP PP3/BP4) | Yes (Mann–Whitney p < 10⁻⁶) | Not evaluated |
| Thermodynamic generalization to unseen genes | Yes (Spearman ρ = 0.817, 10 held-out genes) | No dedicated head |
| Prospective VUS reclassification | Yes (AUROC 0.748, 16 months ahead) | Not demonstrated |
| Hearing-loss gene benchmarks | Outperforms on 12/26 actionable genes | Baseline |
| Gene-therapy target landscapes | OTOF tolerance map (AUROC 0.950) | Not available |
| Mechanism prediction | Yes | No |

---

## Key Results

| Result | Value |
|---|---|
| Genes where tolerance maps outperform AlphaMissense | 12 of 26 (no significant losses; paired bootstrap) |
| Thermodynamic generalization — Spearman ρ (10 held-out genes) | 0.817 |
| Agreement with HL-VCEP PP3/BP4 evidence codes | Mann–Whitney p < 10⁻⁶ |
| Prospective VUS-to-pathogenic reclassification (16 months ahead) | AUROC 0.748 |
| OTOF tolerance landscape (1,997 residues; DB-OTO gene therapy target) | AUROC 0.950 |

---

## Installation

```bash
git clone https://github.com/pipi0616/hears-lm.git
cd hears-lm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

GPU recommended (training was performed on a single NVIDIA A100; inference works on consumer GPUs and CPU). For Apple Silicon, install the MPS-compatible build of PyTorch.

---

## Quick Start

### Inference on a list of variants

```python
from src.model.inference import HearsLMPredictor

predictor = HearsLMPredictor(checkpoint='checkpoints/hears_lm_t30_ep2.pt')
scores = predictor.predict(
    gene='OTOF',
    protein_change=['Tyr730Cys', 'Arg1583His'],
)
# Returns dict: {variant: {'pathogenicity': float, 'pred_ddg': float, 'mechanism': ndarray, 'severity': int}}
```

### Reproducing the manuscript figures

```bash
# Figure 1 — architecture, splits, features, ablation
python src/figures/01_make_fig1.py

# Figures 2 & 3 — thermodynamic generalization + per-residue landscape
python src/figures/02_make_fig2_fig3.py

# Figures 4–6 — KCNQ4 biology, prospective ACMG/HL-VCEP, OTOF clinical
python src/figures/03_make_fig4_fig5_fig6.py

# Supplementary figures S1–S6
python src/figures/04_make_supp_figures.py
```

### Training from scratch

Training requires preprocessed feature files and FoldX ΔΔG targets (released via Zenodo; see *Data Availability* below).

```bash
# 1. Build splits + features
python src/data/01_build_splits.py
python src/data/02_build_pervariant_features.py
python src/data/03_build_celltype_features.py
python src/data/04_build_complex_features.py
python src/data/05_build_hhl_clinical_features.py
python src/data/06_build_external_evidence.py

# 2. Train
python src/model/train.py --epochs 3 --batch-size 32 --lr 1e-4

# 3. Generate held-out predictions
python src/model/predict_splits.py --checkpoint checkpoints/hears_lm_t30_ep2.pt
```

---

## Data Availability

This repository contains source code only. Pre-trained model weights and curated feature data (~4 GB processed; ~6 GB checkpoints) will be deposited at Zenodo on paper acceptance. Released artifacts will include:

- HEARS-LM checkpoint (`hears_lm_t30_ep2.pt`)
- Three held-out evaluation cohorts with computed predictions
- Per-residue tolerance landscapes for 36 hearing-loss genes
- VUS reclassification catalogue and OTOF candidate map

Raw ClinVar, gnomAD, AlphaFold, and DVD data should be obtained from the respective primary sources (all open-access). See `docs/data_dictionary.md` for per-feature provenance.

---

## Repository Layout

```
hears-lm/
├── src/
│   ├── data/               Data partitioning + 100-feature engineering
│   ├── model/              Training + inference (ESM-2 + LoRA + cross-attention)
│   ├── evaluation/         Benchmarks, ESM-2 zero-shot, ACMG analysis, HL-VCEP validation
│   └── figures/            Publication figure generation scripts
├── requirements.txt
├── LICENSE                 MIT
└── CITATION.cff
```

---

## Citation

```bibtex
@article{hears_lm_2026,
  title   = {A hearing-loss--specific protein language model learns
             transferable variant biophysics, aligns with expert clinical
             decisions, and prospectively predicts ClinVar reclassifications},
  author  = {[Author list]},
  journal = {[Journal]},
  year    = {2026},
  doi     = {[DOI]}
}

@software{hears_lm_software_2026,
  title   = {HEARS-LM: Hereditary EAR Sequence Language Model},
  author  = {[Author list]},
  year    = {2026},
  url     = {https://github.com/pipi0616/hears-lm},
  doi     = {[Zenodo DOI]}
}
```

---

## License

MIT (see [LICENSE](LICENSE)). Pretrained models and data files released on Zenodo will be distributed under the same license unless otherwise stated.

## Contact

For questions about the model, contact: shaopeiye55@gmail.com
