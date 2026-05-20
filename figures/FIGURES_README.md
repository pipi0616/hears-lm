# HHL Paper — Figures & Tables Index

All figures support the 4-act narrative: Biology → Landscape → Expert alignment → Clinical translation.

---

## Main Figures (6)

### Figure 1 — Model architecture, data, features, ablations
- **a**: Model architecture diagram (ESM-2 t30 + LoRA + cross-attention + 4 multi-task heads)
- **b**: Dataset splits (train/val_db/val_time/val_gene/val_func_mave) with n + P-rate
- **c**: 100 HHL features grouped (cell-type 44, AF 10, IMPC 8, etc.)
- **d**: Multi-task loss weighting (P/B primary, ΔΔG/mech/ordinal auxiliary)
- **e**: HHL feature group ablation — structural features dominate (Δval_db = −0.122)
- **f**: PLM + features contribution (ESM-zs 0.66 → LGBM-features 0.99* → Ours 0.97; *AF-driven on val_time)

### Figure 2 — ΔΔG generalization to UNSEEN HHL genes (Story 1: STRONG)
- **a**: Predicted vs FoldX ΔΔG hexbin, val_gene (10 unseen genes), ρ=0.817
- **b**: Per-gene |Spearman| vs FoldX (Ours_t30 vs REVEL / AM / ESM-zs / phylop100) — bar chart
- **c**: Paired Δ vs baselines, 95% bootstrap CI — all positive, all p<10⁻⁴
- **d**: Predicted |ΔΔG| violin P vs B (Mann-Whitney p=2.2×10⁻³⁵)
- **e**: Story 1 summary table (gene overlap 0/10 verified)

### Figure 3 — Per-residue mutational landscape outperforms ESM-zs and AlphaMissense (Story 2: STRONG)
- **a**: Per-gene Δ AUC on 26 HHL genes — Ours vs AM (blue) + Ours vs ESM-zs (green), paired bootstrap 95% CI
- **b**: Top-K@residue precision curves (P@5/10/20/50), Wilcoxon p<10⁻⁵
- **c**: Multi-baseline P@10 across 10 methods — Ours #1 (0.669 vs MutPred 0.494)
- **d**: Genes with perfect P@10 (Ours 13/36; AM 2/36; ESM-zs 2/36)
- **e**: Inheritance stratification — syndromic HHL is strongest niche (Δ +0.192)
- **f**: 12 SIG WIN genes labeled with clinical context

### Figure 4 — Biological plausibility (Story 2 mechanistic validation)
- **a**: KCNQ4 mutational landscape — disease residues cluster in pore + S4 voltage sensor
- **b**: KCNQ4 disease-residue detection violin (AUC 0.843, MW p=1.1×10⁻¹⁵)
- **c**: TM domain enrichment in K⁺ channels (KCNQ1 p=2.4×10⁻⁷, KCNQ4 p=4.9×10⁻⁶)
- **d**: 3D spatial clustering in AlphaFold structures (K⁺ channels p<10⁻⁴; not over-clustering on multi-domain proteins)

### Figure 5 — Expert alignment + 16-month prospective ClinVar prediction (Story 3: STRONG)
- **a**: ROC for prospective VUS upgrade (5 methods compared)
- **b**: Violin — pred_prob for stable VUS vs future-upgrades (MW p=7.4×10⁻⁷)
- **c**: AUC comparison bar chart — Ours best (0.748 vs METARNN 0.732, AM 0.722)
- **d**: ClinGen ACMG concordance — PP3 (p=1.2×10⁻⁶), BP4 (p=8.4×10⁻⁷)
- **e**: ClinGen HL-VCEP expert validation (66 variants, AUC 0.901)
- **f**: HL-VCEP KCNQ4 specific-residue rule recovery (5/6, OR=88, p=4.7×10⁻⁶)

### Figure 6 — OTOF gene therapy candidate map + clinical VUS catalog (Story 4: clinical deliverable)
- **a**: OTOF mutational landscape (1997 residues, ClinVar P/LP positions marked)
- **b**: OTOF disease-residue detection AUC (Ours 0.950 vs AM 0.752)
- **c**: VUS catalog summary by gene (227 reclassifications across 12 syndromic HHL genes)
- **d**: VUS novelty audit pie (181 at known P/LP, 26 truly novel, 20 ultra-novel)
- **e**: 18 ultra-novel positions table (>5 aa from any known disease residue — testable hypotheses)

---

## Supplementary Figures (6)

### Supp Fig S1 — Honest framing
- AF dominance on val_time/val_gene (single feature 0.987/0.900)
- val_db ceiling analysis (t30 hits non-stacking limit; METARNN +0.06 via stacking)
- Class balance per split

### Supp Fig S2 — Per-gene Top-10 residue precision table (all 36 HHL genes)
Perfect-P@10 genes highlighted in green.

### Supp Fig S3 — Multi-baseline P@10 (10 methods × 42 genes)
Bar chart: Ours_t30 #1 with 0.669; MutPred #2 with 0.494; ESM1B last.

### Supp Fig S4 — Inheritance stratification + 3D clustering detail
- Syndromic 7/13 SIG WIN, AR 5/10, AD 0/2, XL 0/1
- 3D clustering significant only on K⁺ channels (KCNQ4/Q1 p<10⁻⁴)

### Supp Fig S5 — Calibration plots
- val_db: under-confident (Brier 0.237, ECE 0.290) — Platt scaling recommended for deployment
- val_time/val_gene: better calibrated

### Supp Fig S6 — DVD-expanded retraining preliminary results
- Ep0 partial: val_db +0.005, val_time/val_gene slight regression (not converged)
- Per-gene change on val_db (OPA1/TECTA/MYO7A improved; SLC4A11/COL11A1 worsened)

---

## Main Tables (4)

### Table 1 — Per-split AUC vs baselines
Three rows (val_db / val_time / val_gene) × Ours + 6 baselines.

### Table 2 — 26-gene paired bootstrap (S6) — full detail
Per-gene: n_pos, n_P/LP, Ours_AUC, AM_AUC, Δ AUC, 95% CI, SIG_WIN flag.

### Table 3 — Statistical summary of 11 strong findings
Metric, effect size, 95% CI, p-value for each anchor finding.

### Table 4 — VUS reclassification catalog summary by gene

---

## Supplementary Tables (4)

### Supp Table S1 — Best method by gene (Top-K count)
### Supp Table S2 — 18-20 ultra-novel disease-residue positions for prospective testing
### Supp Table S3 — Tool coverage per split (HHL coverage %)
### Supp Table S4 — Full HHL feature ablation (LightGBM leave-one-group-out)

---

## File listing

```
figures/
├── Fig1_architecture.{png,pdf}
├── Fig2_ddg.{png,pdf}
├── Fig3_landscape.{png,pdf}
├── Fig4_biology.{png,pdf}
├── Fig5_prospective.{png,pdf}
├── Fig6_otof_clinical.{png,pdf}
├── FigS1_af_ceiling.{png,pdf}
├── FigS2_topk_per_gene.{png,pdf}
├── FigS3_multi_baseline.{png,pdf}
├── FigS4_inheritance_3d.{png,pdf}
├── FigS5_calibration.{png,pdf}
└── FigS6_dvd_retrain.{png,pdf}

tables/
├── Table1_per_split_AUC.csv
├── Table2_S6_paired_bootstrap.csv
├── Table3_statistical_summary.csv
├── Table4_VUS_catalog.csv
├── TableS1_best_method_count.csv
├── TableS2_ultra_novel_positions.csv
├── TableS3_tool_coverage.csv
└── TableS4_feature_ablation.csv
```

---

## Key statistical highlights (for figure legends)

- **S1 ΔΔG**: per-gene ρ=0.826 on 10 UNSEEN genes; +0.547 vs ESM-zs [bootstrap 95% CI +0.36, +0.65]
- **S6 vs AM**: 12/26 SIG WIN, median Δ +0.151, binom p=2.4×10⁻⁴
- **S6 vs ESM-zs**: 22/26 SIG WIN, median Δ +0.274, binom p=2.4×10⁻⁷
- **Top-K@residue**: P@10 mean 0.758; 13/36 perfect; Wilcoxon p=3.1×10⁻⁶
- **TM enrichment**: KCNQ1 47% vs 20% (p=2.4×10⁻⁷), KCNQ4 50% vs 20% (p=4.9×10⁻⁶)
- **3D clustering**: KCNQ4 34.5 Å vs 62.0 random (p<10⁻⁴)
- **S7 prospective ClinVar**: AUC 0.787, MW p=7.4×10⁻⁷; **beats METARNN/AM/REVEL**
- **ACMG concordance**: PP3 p=1.2×10⁻⁶, BP4 p=8.4×10⁻⁷
- **HL-VCEP rules**: KCNQ4 5/6 specific residues; OR=88, p=4.7×10⁻⁶
- **ClinGen HL-VCEP held-out (n=66)**: AUC 0.901

---

## Reviewer-defensible claims

1. We are the **only HHL-specific method** with 100% coverage AND **22/26 SIG WIN over ESM-2 zero-shot** baseline at the per-residue level.
2. We **beat all general baselines** (METARNN, AM, REVEL, ESM-zs) on **prospective ClinVar VUS upgrade prediction**, achieved without external predictor stacking.
3. We **align with ClinGen HL-VCEP expert ACMG decisions** at p<10⁻⁶ for both PP3 and BP4 codes.
4. We provide **three clinical resources**: VUS catalog (227 reclassifications), OTOF gene therapy candidate map, per-residue HHL atlas.

## Honest limitations (in Discussion)

- Not the SOTA on raw per-variant val_db AUC (METARNN 0.898 vs ours 0.840, due to predictor stacking which we prohibit)
- AF dominates val_time/val_gene splits
- Mechanism head doesn't generalize
- VUS catalog 9% truly novel positions (mostly within-hotspot prioritization)
- Calibration weak on val_db (needs Platt scaling for deployment)
