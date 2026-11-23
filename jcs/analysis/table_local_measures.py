"""Numbers behind the tables on the local measures: the share of demand met for both groups under each measure.

Reads results/local_measures/report and prints one table per setting. Levels are percentages;
changes are percentage points over the 1500 steps after the measure starts, with a 95% interval over
worlds. Siting rows compare the same 1500 steps of the original game under sited and under random
placement. The share of requests delivered separates receiving less from asking less.

Usage: table_local_measures.py [setting ...]   (default: every setting of the run)
"""
import os
import sys
from pathlib import Path
import pandas as pd

RESULTS = Path(os.environ.get('JCS_RESULTS', Path(__file__).resolve().parents[1]/'results'))
report = RESULTS/'local_measures/report'
effects = pd.read_csv(report/'intervention_effects.csv')
effects = effects[(effects.timing == 'after_development') & effects.forced.isna()]
effects = effects[effects.window_stop-effects.window_start == (effects.window_stop-effects.window_start).max()]
placement = pd.read_csv(report/'intervention_placement_effects.csv')
placement = placement[placement.window.isin(set(effects.window))]


def line(name, row, before):
    cells = [f'{name:<46s}']
    for group in ['human', 'ai']:
        level, change = 100*row[f'{group}_activity_{before}'], 100*row[f'{group}_activity_difference_mean']
        low, high = 100*row[f'{group}_activity_difference_lower'], 100*row[f'{group}_activity_difference_upper']
        cells.append(f'{level:5.1f} -> {level+change:5.1f} ({change:+5.1f}, {low:+5.1f} to {high:+5.1f})')
    for group in ['human', 'ai']:
        key = f'{group}_request_met_{before}'
        if key in row and pd.notna(row[key]):
            cells.append('{:5.1f} -> {:5.1f}'.format(100*row[key], 100*(row[key]+row[f'{group}_request_met_difference_mean'])))
    level, change = 100*row[f'empty_share_{before}'], 100*row['empty_share_difference_mean']
    cells.append(f'{level:5.1f} -> {level+change:5.1f}')
    return ' | '.join(cells)


for setting in sys.argv[1:] or list(dict.fromkeys(effects.cell)):
    part = effects[effects.cell == setting]
    sited = placement[placement.random_placement_cell == setting]
    if part.empty and sited.empty:
        continue
    print(f'\n{setting} ({int(part.worlds.iloc[0]) if len(part) else int(sited.worlds.iloc[0])} worlds)')
    print(' '*46+' | humans, share of demand met (change, interval) | AI, share of demand met (change, interval)'
          ' | humans, share of requests delivered | AI, share of requests delivered | sources nearly empty')
    for row in part.to_dict('records'):
        print(line(row['measure'], row, 'unchanged'))
    for row in sited.to_dict('records'):
        print(line('siting: '+row['cell'].replace('_', ' '), row, 'random_placement'))
