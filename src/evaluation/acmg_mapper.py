#!/usr/bin/env python3
"""
ACMG/AMP criteria mapping for HHL variants — translate model output to clinical evidence codes.

Output per variant:
  PVS1: null variant (Ter/frameshift) in LoF-intolerant HHL gene
  PM1: variant in functional domain (in_domain==1) AND high constraint
  PM2_supporting: absent from gnomAD (has_af==0)
  PM5: novel missense at residue where another pathogenic missense seen (proxy: high HPO + AM>0.7)
  PP3: multiple lines computational pathogenic (≥3 of: our_prob>0.7, AM>0.564, phylop100>4, ddg_abs>1.5)
  PS3_supporting: functional MAVE evidence (mean_score_z < -2)
  BS1: popmax AF > 0.005
  BS2: high AF in expected unaffected population (popmax > 0.001 for AR)
  BP4: multiple lines computational benign (≥3 of: our_prob<0.3, AM<0.34, phylop100<2, ddg_abs<0.5)
  BP6: no other criteria + benign predictions

Final classification by ACMG combining rules:
  Pathogenic: 1 PVS1 + ≥1 PS + (PM*1 or PP*≥2)
  Likely Pathogenic: 1 PVS1 + 1 PM* | 2 PS | 1 PS + ≥3 PM | ...
  (Simplified Bayesian point system: PVS=8, PS=4, PM=2, PP=1, BS=-4, BP=-1; total > 10 → P)
"""
import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix,
                              classification_report)

ROOT = Path('/Users/pipi/Projects/QAFI_Paper/hearing_v2')
EVAL = ROOT/'eval_results'

# Load val_db with full features + our model predictions
def load_with_predictions(split_name):
    df = pd.read_csv(ROOT/f'data/splits/{split_name}_with_features.csv', low_memory=False)
    # Add v2 predictions (best balanced model)
    for tag in ['', 'v2', 'v3']:
        suffix = '_ep2' if tag == '' else f'_{tag}_ep2'
        if split_name in ('val_db','val_func_mave'):
            pname = f'{split_name}_predictions{suffix}.csv'
        else:
            vtag = 'ep2' if tag == '' else f'{tag}_ep2'
            pname = f'{split_name}_predictions_{vtag}.csv'
        p = EVAL/pname
        col = f'Ours_v{1 if tag=="" else tag[-1]}'
        if p.exists():
            d = pd.read_csv(p)
            df = df.merge(d[['gene','protein_change','pred_prob']].rename(columns={'pred_prob':col}),
                          on=['gene','protein_change'], how='left')
    return df


def is_null_variant(pc):
    s = str(pc)
    m = re.match(r'^([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$', s)
    if not m: return False
    return m.group(3) in ('Ter','X')


def apply_acmg(row, our_prob_col='Ours_v2'):
    """Apply ACMG criteria for a single variant. Returns dict of triggered criteria + final class."""
    criteria = []
    points = 0  # ACMG-Bayes points

    pc = row.get('protein_change','')
    our_p = row.get(our_prob_col, np.nan)
    am_p = row.get('am_score_best', np.nan)
    phylop = row.get('phylop100', np.nan)
    plddt = row.get('plddt', np.nan)
    ddg_abs = row.get('abs_ddg_fold', np.nan)
    in_domain = row.get('in_domain', 0)
    has_af = row.get('has_af', 0)
    af_popmax = row.get('af_popmax', 0)
    af_high_bs1 = row.get('af_high_bs1', 0)
    loeuf_constrained = row.get('gnomad_loeuf_constrained', 0)
    mis_z = row.get('gnomad_mis_z', np.nan)
    mean_score_z = row.get('mean_score_z', np.nan)
    is_AR = row.get('cgd_AR', 0)

    # ===== Pathogenic criteria =====

    # PVS1: null variant in LoF-intolerant HHL gene
    if is_null_variant(pc) and (loeuf_constrained == 1 or row.get('gnomad_pLI_high', 0) == 1):
        criteria.append('PVS1')
        points += 8

    # PS1: same AA change as established pathogenic (approximate via high prob from training similarity — skip)
    # PS2/PS4: de novo / case-control (need extra data — skip)
    # PS3: well-established functional studies
    if pd.notna(mean_score_z) and mean_score_z < -2.0:
        criteria.append('PS3_supporting')
        points += 2  # supporting

    # PM1: in functional domain AND high missense constraint
    if in_domain == 1 and pd.notna(mis_z) and mis_z > 3.09:
        criteria.append('PM1')
        points += 2

    # PM2_supporting: absent from gnomAD
    if has_af == 0 and not is_null_variant(pc):
        criteria.append('PM2_supporting')
        points += 1

    # PM5: novel missense, computational strongly pathogenic
    if not is_null_variant(pc) and pd.notna(am_p) and am_p > 0.7 and our_p > 0.7:
        criteria.append('PM5')
        points += 2

    # PP3: multiple lines computational
    pp3_count = sum([
        pd.notna(our_p) and our_p > 0.7,
        pd.notna(am_p) and am_p > 0.564,  # AM 'likely_pathogenic' threshold
        pd.notna(phylop) and phylop > 4.0,
        pd.notna(ddg_abs) and ddg_abs > 1.5,
    ])
    if pp3_count >= 3:
        criteria.append('PP3_strong')
        points += 4  # strong evidence (recent ACMG update)
    elif pp3_count >= 2:
        criteria.append('PP3')
        points += 1

    # ===== Benign criteria =====

    # BS1: allele frequency > expected
    if af_high_bs1 == 1:
        criteria.append('BS1')
        points -= 4

    # BS2: in healthy controls (proxy: high AF for AR)
    if is_AR == 1 and pd.notna(af_popmax) and af_popmax > 0.001:
        if 'BS1' not in criteria:
            criteria.append('BS2')
            points -= 4

    # BP4: multiple lines computational benign
    bp4_count = sum([
        pd.notna(our_p) and our_p < 0.3,
        pd.notna(am_p) and am_p < 0.34,
        pd.notna(phylop) and phylop < 2.0,
        pd.notna(ddg_abs) and ddg_abs < 0.5,
    ])
    if bp4_count >= 3:
        criteria.append('BP4_strong')
        points -= 4
    elif bp4_count >= 2:
        criteria.append('BP4')
        points -= 1

    # ===== Final classification (ACMG-Bayes points system, Tavtigian et al 2018) =====
    if points >= 10:
        acmg_class = 'Pathogenic'
    elif points >= 6:
        acmg_class = 'Likely_Pathogenic'
    elif points <= -7:
        acmg_class = 'Benign'
    elif points <= -1:
        acmg_class = 'Likely_Benign'
    else:
        acmg_class = 'VUS'

    return {
        'acmg_criteria': '|'.join(criteria) if criteria else 'None',
        'acmg_points': points,
        'acmg_class': acmg_class,
        'n_criteria': len(criteria),
    }


# ============ Run on val_db ============
print('='*60)
print('Applying ACMG to val_db')
print('='*60)
db = load_with_predictions('val_db')
print(f'Loaded val_db: {db.shape}')

acmg_results = db.apply(lambda r: apply_acmg(r, 'Ours_v2'), axis=1, result_type='expand')
db = pd.concat([db, acmg_results], axis=1)

print(f'\n--- ACMG class distribution ---')
print(db.acmg_class.value_counts())
print(f'\n--- ACMG class vs ClinVar truth (y) ---')
print(pd.crosstab(db.acmg_class, db.y, margins=True, margins_name='Total'))

# Classification accuracy when reducing to binary (P+LP=1, B+LB=0, VUS=ignore)
db['acmg_binary'] = db.acmg_class.map({
    'Pathogenic': 1, 'Likely_Pathogenic': 1,
    'Benign': 0, 'Likely_Benign': 0,
    'VUS': np.nan
})

m = db.acmg_binary.notna()
y_true = db.loc[m, 'y'].astype(int)
y_acmg = db.loc[m, 'acmg_binary'].astype(int)
print(f'\n--- ACMG binary (P/LP vs B/LB) accuracy ---')
print(f'  Non-VUS coverage: {m.sum()}/{len(db)} ({100*m.sum()/len(db):.0f}%)')
print(f'  Accuracy: {(y_true == y_acmg).mean():.3f}')
print(f'  Concordance with ClinVar: {((y_true == y_acmg)).mean():.3f}')
cm = confusion_matrix(y_true, y_acmg)
print(f'  Confusion: {cm.tolist()}')
print(classification_report(y_true, y_acmg, target_names=['Benign','Path'], digits=3))

# AUC of acmg_points
m_pts = db.acmg_points.notna()
auc_pts = roc_auc_score(db.loc[m_pts, 'y'], db.loc[m_pts, 'acmg_points'])
print(f'\n--- ACMG points as continuous score AUC: {auc_pts:.3f} (vs our_prob 0.821, AM 0.814) ---')

# Compare ACMG class to raw model probability at threshold 0.5
db['model_binary'] = (db.Ours_v2 >= 0.5).astype(int)
acc_model = (db.y.astype(int) == db.model_binary).mean()
print(f'\n--- Raw model @ thresh 0.5: accuracy = {acc_model:.3f} on full set ---')

# VUS analysis
n_vus = (db.acmg_class == 'VUS').sum()
print(f'\n--- VUS analysis ---')
print(f'  ACMG-VUS: {n_vus}/{len(db)} ({100*n_vus/len(db):.0f}%)')
print(f'  Compare ClinVar baseline: ClinVar marks all val_db with definitive labels')

# Save
OUT = EVAL/'acmg_output'
OUT.mkdir(exist_ok=True)
db_out = db[['gene','protein_change','y','Ours_v2','am_score_best','acmg_criteria','acmg_points','acmg_class','acmg_binary']]
db_out.to_csv(OUT/'val_db_acmg.csv', index=False)
print(f'\n✅ Saved → {OUT}/val_db_acmg.csv')

# Same for val_gene
print('\n' + '='*60)
print('Applying ACMG to val_gene')
print('='*60)
vg = load_with_predictions('val_gene')
acmg_vg = vg.apply(lambda r: apply_acmg(r, 'Ours_v2'), axis=1, result_type='expand')
vg = pd.concat([vg, acmg_vg], axis=1)
print(f'\nACMG class distribution:')
print(vg.acmg_class.value_counts())
print(f'\nvs ClinVar y:')
print(pd.crosstab(vg.acmg_class, vg.y, margins=True, margins_name='Total'))

vg['acmg_binary'] = vg.acmg_class.map({
    'Pathogenic': 1, 'Likely_Pathogenic': 1,
    'Benign': 0, 'Likely_Benign': 0, 'VUS': np.nan
})
m = vg.acmg_binary.notna()
print(f'\nval_gene binary classifications: {m.sum()}/{len(vg)}, accuracy: {((vg.loc[m,"y"].astype(int) == vg.loc[m,"acmg_binary"].astype(int))).mean():.3f}')
print(f'val_gene VUS rate: {(vg.acmg_class == "VUS").sum()/len(vg)*100:.0f}%')

vg[['gene','protein_change','y','Ours_v2','am_score_best','acmg_criteria','acmg_points','acmg_class','acmg_binary']].to_csv(OUT/'val_gene_acmg.csv', index=False)
print(f'✅ Saved → {OUT}/val_gene_acmg.csv')
