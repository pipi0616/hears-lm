"""
Location of the upstream data of the original study.

Those files live in a folder that has been moved at least once ('hearing/' -> '_archive/hearing/'), so the
path is resolved at import time from a list of candidates and from the environment variable HEARSLM_LEGACY.
Derived tables that the revision needs are cached inside this project (data/, data/features/,
data/annotations/), so a complete re-run of the analysis does not require the upstream files.
"""
import os
from pathlib import Path

CANDIDATES = [
    Path(os.environ['HEARSLM_LEGACY']) if os.environ.get('HEARSLM_LEGACY') else None,
    Path('/Users/pipi/Projects/QAFI_Paper/hearing'),
    Path('/Users/pipi/Projects/QAFI_Paper/_archive/hearing'),
]
MARKER = 'data/features/gene_sequences.csv'


def legacy_root(required=True):
    for c in CANDIDATES:
        if c and (c / MARKER).exists():
            return c
    if required:
        raise FileNotFoundError(
            'upstream data of the original study not found; set HEARSLM_LEGACY to the folder that contains '
            f'{MARKER} (tried: ' + ', '.join(str(c) for c in CANDIDATES if c) + ')')
    return None


LEGACY = legacy_root(required=False)
