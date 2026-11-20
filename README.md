# HEARS-LM v2

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20311406.svg)](https://doi.org/10.5281/zenodo.20311406)

> Release v2.0 is the version described and evaluated in the paper. Releases v1.0.x contain an earlier version of
> the model that has been superseded.

Hearing-loss–adapted protein language model for scoring missense variants in the 163 genes of the Genomics England
PanelApp hearing-loss panel, with ACMG/AMP evidence-strength calibration (PP3/BP4).

> HEARS-LM provides computational evidence (PP3/BP4) for use within ACMG/AMP–ClinGen classification. It does not
> classify variants. It covers missense substitutions in coding sequence only.

## Quick start

```bash
conda env create -f environment.yml && conda activate hearslm
python code/hearslm_score.py my_variants.csv -o scores.csv
```

`my_variants.csv` needs `gene` and `protein_change` (`p.Arg143Trp`, `Arg143Trp` or `R143W`, numbered on the RefSeq
transcript ClinVar uses for the gene, normally MANE Select). Rows without a coordinate are scored by the
sequence-only model. Optional `chrom,pos,ref,alt` (GRCh38) switch to the selected (full) model, whose conservation
inputs are then fetched from dbNSFP through MyVariant.info; add `--no-finetune` to score those rows with the
configuration without fine-tuning instead, which runs quickly on a CPU. A coordinate that dbNSFP does not resolve to the
stated protein change is not used: that row is scored by the sequence-only model and a note is printed.

`python code/tests/check_cli_reproduces_evaluation.py` verifies that the tool reproduces the evaluated models.

Output columns: `prob` (ensemble probability), `model` (`full`, `no-finetune` or `sequence-only`), `evidence`
(calibrated ACMG/AMP level: `PP3_moderate`, `PP3_supporting`, `indeterminate`, `BP4_supporting`, `BP4_moderate`;
Supporting and Moderate are the strengths evaluated in the paper), `n_training_variants_gene`, `ddg_pred` (DDGun-like folding free-energy change, kcal/mol, positive = destabilising; models with
the auxiliary head only).

## Models

| model | inputs | thresholds | use |
|---|---|---|---|
| `runs/full/seed{0,1,2}` | sequence + full-protein ESM-2 values + conservation + structure + gene-level descriptors | `results/calibration_thresholds.csv` | selected configuration; used by the tool when a coordinate is given |
| `runs/seq_plm_ddg/seed{0,1,2}` | sequence + full-protein ESM-2 values | `results/calibration_thresholds_seq_plm_ddg.csv` | no external annotation required |
| `runs/feat_plm_only/seed{0,1,2}` | full-protein ESM-2 values + the three descriptor groups; no fine-tuning | `results/calibration_thresholds_feat_plm_only.csv` | no graphics processor needed (`--no-finetune`) |
| `runs/release_full/`, `runs/release_seq_plm_ddg/` | as the first two rows, retrained on every labelled variant | none | additional download for research use; score scale not calibrated |

These are the models evaluated in the paper: all reported performance and calibration, the command-line tool and the
precomputed scores refer to them. Thresholds were derived on the pooled test sets (temporal, literature-evidence and
withheld-gene variants; see Supplementary Tables S9-S9e), which makes them conservative for panel genes.

**Genes without evidence levels.** The evaluated models were trained without ten genes withheld for evaluation (OTOF,
CDH23, TECTA, WFS1, TMC1, LOXHD1, STRC, CLDN14, TBC1D24, ACTG1). For these genes, and for panel genes without any
training label, the tool and the catalogue report the score but no evidence level: on genes absent from training
the model is less accurate than general-purpose predictors, and its evidence is less reliable (paper, Sections 3.5
and 3.7). `n_training_variants_gene` gives the number of training variants of the gene so that sparsely represented
genes can be recognised. Use a general-purpose predictor for genes without curated variants.

## Precomputed scores

`results/vus_catalogue.csv`: every ClinVar (April 2026) missense variant of uncertain or conflicting significance in
the panel genes that carries no label in the study, scored by the selected model, with the evidence level (genes with
training variants only), the number of training variants of the gene and whether a pathogenic missense variant is
known at or near the residue.

## Reproducing the paper

See `REPRODUCE.md` (pipeline order) and `02_ANALYSIS_PLAN.md` / Supplementary Note S1 (pre-specified analysis plan
and dated amendments).

## Citation

Ye S, Chen P, Han Y, Wang L, Chen K. HEARS-LM: a hearing-loss–adapted protein language model for missense variant
interpretation, evaluated with leakage-controlled benchmarks. (submitted)

Code, models and data: https://doi.org/10.5281/zenodo.20311406 (all versions; v2.0 is the version used in the paper).

## Licence

MIT (code). Model weights: MIT. Upstream data keep their original licences (ClinVar, Deafness Variation Database,
dbNSFP, gnomAD, AlphaFold DB, UniProt, MaveDB).
