# HEARS-LM

**Hereditary EAR Sequence Language Model** — a hearing-loss–specific protein language model for missense variant pathogenicity, thermodynamics, mechanism, and clinical interpretation.

> A hearing-loss–specific protein language model learns transferable variant biophysics, aligns with expert clinical decisions, and prospectively predicts ClinVar reclassifications. Preprint: forthcoming.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## What this is

HEARS-LM couples a domain-adapted 30-layer ESM-2 protein language model with one hundred curated hearing-loss–specific biological descriptors through cross-attention fusion. Four task-specific heads—pathogenicity, thermodynamic ΔΔG, mechanism, and ordinal severity—are trained jointly under a thermodynamic auxiliary objective supervised by FoldX folding free-energy estimates.

Key results from the accompanying paper:

- Per-residue tolerance maps significantly outperform **AlphaMissense** on twelve of twenty-six clinically actionable syndromic hearing-loss genes (paired bootstrap; no significant losses).
- Auxiliary thermodynamic head generalizes to ten genes never seen during folding-energy supervision (aggregate Spearman ρ = 0.817).
- Predictions agree with ClinGen Hearing Loss Variant Curation Expert Panel (HL-VCEP) PP3/BP4 evidence codes (Mann–Whitney p < 10⁻⁶).
- Anticipates ClinVar VUS-to-pathogenic upgrades sixteen months ahead of expert reclassification (AUROC 0.748).
- Releases a 1,997-residue OTOF tolerance landscape (AUROC 0.950) directly relevant to the active DB-OTO gene-therapy programme (NCT05788536).

## Repository layout

```
hears-lm/
├── src/
│   ├── data/               Data partitioning + 100-feature engineering
│   ├── model/              Training + inference (ESM-2 + LoRA + cross-attention)
│   ├── evaluation/         Benchmarks, ESM-2 zero-shot, ACMG analysis, HL-VCEP validation
│   └── figures/            Publication figure generation
├── figures/                Publication figures (PNG + PDF, 300 dpi)
├── tables/                 Result tables (CSV)
├── docs/                   Manuscript + supplementary material
├── requirements.txt
├── LICENSE                 MIT
└── CITATION.cff
```

## Installation

```bash
git clone https://github.com/pipi0616/hears-lm.git
cd hears-lm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

GPU recommended (training was performed on a single NVIDIA A100; inference works on consumer GPUs and CPU). For Apple Silicon, install the MPS-compatible build of PyTorch.

## Quick start

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

Training requires preprocessed feature files and FoldX ΔΔG targets (released via Zenodo separately; see *Data availability* below).

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

## Data availability

This repository contains source code only. The pre-trained model weights and curated feature data (~4 GB processed; ~6 GB checkpoints) are too large for GitHub and will be deposited at Zenodo on paper acceptance. The released artifacts will include:

- HEARS-LM checkpoint (`hears_lm_t30_ep2.pt`)
- Three held-out evaluation cohorts with computed predictions
- Per-residue tolerance landscapes for 36 hearing-loss genes
- VUS reclassification catalogue and OTOF candidate map

Raw ClinVar, gnomAD, AlphaFold, and DVD data should be obtained from the respective primary sources, all of which are open-access. See `docs/data_dictionary.md` for the per-feature provenance.

## Reproducing the paper

The figures (`figures/*.png`, `figures/*.pdf`) and tables (`tables/*.csv`) in this repository correspond exactly to the publication. To regenerate them from scratch, the scripts in `src/figures/` read pre-computed prediction CSVs from `eval_results/`, which are released alongside the checkpoint on Zenodo.

## Citation

If you use HEARS-LM or any of the released resources in your work, please cite the manuscript and this software repository:

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

## Acknowledgements

This work uses ESM-2 sequence representations (Lin et al., 2023), AlphaFold2 monomer structures (Jumper et al., 2021), FoldX folding free-energy estimates (Schymkowitz et al., 2005), ClinVar (Landrum et al., 2020), the Deafness Variation Database (Azaiez et al., 2018), the ClinGen Hearing Loss Variant Curation Expert Panel, and the broader hearing-loss research community whose curated annotations underpin the feature panel.

## License

MIT (see [LICENSE](LICENSE)). Pretrained models and any data files released on Zenodo will be distributed under the same license unless otherwise stated.

## Contact

For questions about the model, contact: shaopeiye55@gmail.com. For clinical reclassification submissions to ClinGen HL-VCEP based on HEARS-LM predictions, follow the panel's standard submission process; this software is provided as a research tool and is not a clinical decision-support device.
