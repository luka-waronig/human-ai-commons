"""Numbers behind the statements on capacity and source density: what lasts beyond the earlier paper's 100 steps.

Reads results/original_game/report/continuity_conditions.csv and prints cooperation and the
share of demand met in the first 100 steps and in steps 1500 to 2000, for full starts.
"""
import os
from pathlib import Path
import pandas as pd

RESULTS = Path(os.environ.get('JCS_RESULTS', Path(__file__).resolve().parents[1]/'results'))
table = pd.read_csv(RESULTS/'original_game/report/continuity_conditions.csv')
table = table[(table.initial_stock == 'full') & (table.group != 'ladder') & table.window.isin(['first_100', 'late'])]
names = {'fixed_absolute': 'fixed requests', 'scaled_to_accessible_renewal': 'requests scaled to renewal'}

for requests, part in table.groupby('requests', sort=False):
    for varied, fixed, value in [('capacity', 'density', .3), ('density', 'capacity', 20)]:
        rows = part[part[fixed] == value].drop_duplicates([varied, 'window']).sort_values([varied, 'window'])
        print(f'\n{names[requests]}, {fixed} {value:g}: {varied} | window | pressure | cooperation | humans | AI (share of demand met, %)')
        for row in rows.itertuples():
            window = 'first 100 steps' if row.window == 'first_100' else 'steps 1500 to 2000'
            print(f'  {getattr(row, varied):>5g} | {window:<18s} | {row.relative_load:4.2f} | {100*row.cooperation:5.1f} | {100*row.human_activity:5.1f} | {100*row.ai_activity:5.1f}')
