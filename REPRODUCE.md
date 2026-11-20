# Reproducing HEARS-LM v2

Python environment: `conda env create -f environment.yml`. Paths below are relative to this folder.

**What runs from the release alone.** Steps 11-18 (training, evaluation, tables, figures) and the
command-line tool use only files in this release: `data/` (labelled variants, annotation and descriptor tables,
reference proteins), `runs/` (weights and predictions) and `results/`.

**What needs the raw downloads.** Steps 1-10 rebuild the data tables from the raw inputs (ClinVar monthly
`variant_summary` releases 2025-01 and 2026-04, the Deafness Variation Database export, the PanelApp panel, UniProt
entries, AlphaFold models, Tollefson et al. 2023 Table S4, the gnomAD v4 constraint table, the descriptor sources of
Supplementary Table S3). They read these from the directory given by the environment variable `HEARSLM_LEGACY`
(`code/data/paths.py`); sources and access dates are in Supplementary Table S3. Every table these steps produce is
included in the release.

| step | script | output |
|---|---|---|
| 1 | `code/data/01_extract_sources.py` | ClinVar / DVD missense records of the panel genes |
| 2 | `code/data/02_fetch_transcript_proteins.py` | RefSeq protein sequences of all source transcripts (NCBI E-utilities) |
| 3 | `code/data/03_canonicalize.py` | one reference protein per gene; every variant re-mapped and verified |
| 4 | `code/data/04_build_splits.py`, `04b_cv_folds_by_family.py` | `data/variants_v8.csv` (labels, provenance, partitions, family folds) |
| 5 | `code/data/05_annotate_dbnsfp.py` | dbNSFP 4.8a scores, conservation, gnomAD AF by GRCh38 key (MyVariant.info; cached) |
| 6 | `code/data/06_structure_features.py` | AlphaFold pLDDT and UniProt features per residue |
| 7 | `code/data/07_map_ddgun.py` | DDGun ddG (Tollefson et al. 2023) on reference numbering |
| 8 | `code/data/08_esm2_zeroshot.py`, `13_esm2_wt_marginals.py`, `14_refresh_zeroshot_summary.py` | ESM-2 150M zero-shot scores; full-context log-probabilities; per-partition summaries |
| 9 | `code/data/09_build_features.py` | descriptor table and groups (labelled variants; `--variants/--dbnsfp/--out` for the VUS table) |
| 10 | `code/data/10_clingen_expert.py`, `11_vus_table.py`, `12_mave_tables.py` | expert classifications, VUS table, MaveDB tables |
| 11 | `code/model/run_queue.py queue.txt` | every training run listed in `queue.txt` (`code/model/train.py`) |
| 12 | `code/eval/evaluate_fixed.py --stage dev --final LOCK` | model selection on the development set (locks `results/final_model.json`) |
| 13 | `code/eval/evaluate_fixed.py --stage test`, `benchmark_common_pairwise.py`, `arm_pair_contrasts.py` | test-set results; comparison panel by the coverage rule (`code/eval/common.py`); paired tests; component contrasts |
| 14 | `code/eval/evaluate_cv.py --arm cv_<arm>` | leave-gene-family-out cross-validation (same panel) |
| 15 | `code/model/predict.py --runs runs/full/seed0 runs/full/seed1 runs/full/seed2 --variants data/vus_2026-04_variants.csv --features-csv data/features/features_vus.csv --out results/vus_predictions_full.csv` | scores of the unresolved ClinVar variants (catalogue) |
| 16 | `code/eval/predict_external.py`, `evaluate_mave.py`, `thermo_tables.py`, `code/figures/figS3_kcnq4.py` | assay variants and the KCNQ4 scan scored by the sequence-only cross-validation models; assay correlations; tables behind Figure 5a-b |
| 17 | `code/eval/evaluate_clinical.py`, `evidence_by_partition.py`, `calibrate_arm.py seq_plm_ddg`, `calibrate_arm.py feat_plm_only` | calibration, cross-fitted yields, thresholds by population, expert panel, reclassification, catalogue; thresholds and yields of the two other released configurations |
| 18 | `code/eval/data_tables.py`, `table_components.py`, `make_tables.py`, `make_supplement.py`, `code/figures/*.py` | tables, supplementary workbook and figures |

The manuscript text and the scripts that insert the numbers into it are not part of this release; every number in the
article is in the results tables (`results/`) and the supplementary workbook produced by step 18.

Random seeds: data partition hash seed 20261006; model seeds 0-2 (see `queue.txt`); bootstrap seed 0.
`02_ANALYSIS_PLAN.md` records the pre-specified analysis plan and its dated amendments (English version:
Supplementary Note S1).

Check of the released tool: `python code/tests/check_cli_reproduces_evaluation.py` scores test variants through
`code/hearslm_score.py` and compares them with the stored test-set predictions (needs network access for dbNSFP).
