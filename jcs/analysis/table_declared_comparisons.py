"""The comparisons a run declared before it was simulated, with their intervals.

Reads results/<run>/report/confirmatory_contrasts.csv. Every declared outcome is a share, printed in
percent (stock in percent of capacity), and differences are percentage points; the interval shares one 5%
allowance over all comparisons of the run.

Usage: table_declared_comparisons.py [run]   (default: local_measures)
"""
import os
import sys
from pathlib import Path
import pandas as pd

RESULTS = Path(os.environ.get('JCS_RESULTS', Path(__file__).resolve().parents[1]/'results'))
run = sys.argv[1] if len(sys.argv) > 1 else 'local_measures'
table = pd.read_csv(RESULTS/run/'report/confirmatory_contrasts.csv')
print(f'{run}: {len(table)} declared comparisons, {int(table.worlds.min())} to {int(table.worlds.max())} worlds each')
for question, part in table.groupby('question', sort=False):
    print(f'\n{question}')
    for row in part.itertuples():
        print(f'  {row.name:<44s} {row.outcome:<16s} {100*row.second_mean:7.2f} -> {100*row.first_mean:7.2f}  '
              f'difference {100*row.difference:+6.2f} ({100*row.lower:+6.2f} to {100*row.upper:+6.2f})  higher in {row.worlds_positive} of {row.worlds} worlds')
