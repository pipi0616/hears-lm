#!/usr/bin/env python3
"""
Step 02 - Fetch the protein sequence of every source transcript used by ClinVar (NM_ accession in the
variant Name) and by the Deafness Variation Database (NP_ accession in hgvs_p), so that each reported
protein change can be re-mapped onto the UniProt canonical sequence used by the model (step 03).

Source: NCBI E-utilities efetch (nuccore fasta_cds_aa for NM_, protein fasta for NP_).
Output: data/sources/transcript_proteins.csv  (accession, kind, protein_id, length, sequence)
Cached: re-running only fetches missing accessions.
"""
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]                       # the release folder
SRC = ROOT / 'data/sources'
OUT = SRC / 'transcript_proteins.csv'
EUTILS = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
SNAPSHOTS = ['2025-01', '2025-06', '2026-01', '2026-04']


def efetch(db, ids, rettype):
    for attempt in range(5):
        try:
            r = requests.post(EUTILS, data={'db': db, 'id': ','.join(ids), 'rettype': rettype, 'retmode': 'text',
                                            'tool': 'hearslm_revision', 'email': 'shaopeiye55@gmail.com'}, timeout=120)
            if r.status_code == 200 and r.text.strip():
                return r.text
            print(f'  efetch status {r.status_code}, retry {attempt}', file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f'  efetch error {e!r}, retry {attempt}', file=sys.stderr)
        time.sleep(3 * (attempt + 1))
    return ''


def parse_fasta(text):
    recs, header, seq = [], None, []
    for line in text.splitlines():
        if line.startswith('>'):
            if header is not None:
                recs.append((header, ''.join(seq)))
            header, seq = line[1:], []
        elif line.strip():
            seq.append(line.strip())
    if header is not None:
        recs.append((header, ''.join(seq)))
    return recs


def main():
    nms = set()
    for snap in SNAPSHOTS:
        d = pd.read_csv(SRC / f'clinvar_{snap}.csv', usecols=['nm', 'vclass'], dtype=str)
        nms |= set(d.loc[d.vclass == 'missense', 'nm'].dropna())
    dvd = pd.read_csv(SRC / 'dvd_missense.csv', usecols=['np', 'vclass'], dtype=str)
    nps = set(dvd.loc[dvd.vclass == 'missense', 'np'].dropna())
    nms = {x for x in nms if x.startswith('NM_')}
    print(f'{len(nms)} NM accessions (ClinVar), {len(nps)} NP accessions (DVD)')

    done = pd.read_csv(OUT, dtype=str) if OUT.exists() else pd.DataFrame(
        columns=['accession', 'kind', 'protein_id', 'length', 'sequence'])
    have = set(done.accession)
    rows = done.to_dict('records')

    todo_nm = sorted(nms - have)
    for i in range(0, len(todo_nm), 40):
        batch = todo_nm[i:i + 40]
        text = efetch('nuccore', batch, 'fasta_cds_aa')
        for header, seq in parse_fasta(text):
            m = re.match(r'lcl\|(N[MR]_\d+\.\d+)_prot_([A-Z]P_\d+\.\d+)', header)
            if m:
                rows.append({'accession': m.group(1), 'kind': 'NM', 'protein_id': m.group(2),
                             'length': len(seq), 'sequence': seq})
        print(f'  NM {i + len(batch)}/{len(todo_nm)}', flush=True)
        time.sleep(0.4)

    todo_np = sorted(nps - have)
    for i in range(0, len(todo_np), 40):
        batch = todo_np[i:i + 40]
        text = efetch('protein', batch, 'fasta')
        for header, seq in parse_fasta(text):
            acc = header.split()[0]
            rows.append({'accession': acc, 'kind': 'NP', 'protein_id': acc, 'length': len(seq), 'sequence': seq})
        print(f'  NP {i + len(batch)}/{len(todo_np)}', flush=True)
        time.sleep(0.4)

    out = pd.DataFrame(rows).drop_duplicates('accession')
    out.to_csv(OUT, index=False)
    got = set(out.accession)
    print(f'fetched {len(out)} sequences; missing NM: {sorted(nms - got)[:20]} ({len(nms - got)}); '
          f'missing NP: {sorted(nps - got)[:20]} ({len(nps - got)})')


if __name__ == '__main__':
    main()
