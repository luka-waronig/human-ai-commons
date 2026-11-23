"""Numbers behind the statements on the local measures under changed assumptions.

Reads results/local_measures_assumptions/report. For each changed assumption it prints the share of demand
met under the original game and its change under the price, the cap, both together and siting, in
percentage points over the 1500 steps after the change. The first interval is the 95% interval over
worlds; the second is the corrected interval of the declared comparison. The last column is the share of
reachable sources that are nearly empty.
"""
import os
from pathlib import Path
import pandas as pd

RESULTS = Path(os.environ.get('JCS_RESULTS', Path(__file__).resolve().parents[1]/'results'))
report = RESULTS/'local_measures_assumptions/report'
effects = pd.read_csv(report/'intervention_effects.csv')
effects = effects[(effects.timing == 'after_development') & (effects.window == '0-1500')].set_index(['cell', 'design'])
placement = pd.read_csv(report/'intervention_placement_effects.csv')
placement = placement[placement.window == '0-1500'].set_index('random_placement_cell')
declared = pd.read_csv(report/'confirmatory_contrasts.csv').set_index('name')
measures = {'price_4': 'price', 'cap_25': 'cap', 'price_4_cap_25': 'price and cap'}


def line(name, row, before, contrast):
    cells = [f'{name:<14s}']
    for group in ['human', 'ai']:
        change = 100*row[f'{group}_activity_difference_mean']
        low, high = 100*row[f'{group}_activity_difference_lower'], 100*row[f'{group}_activity_difference_upper']
        corrected = declared.loc[contrast.format(group=group)]
        cells.append(f'{change:+5.1f} ({low:+5.1f} to {high:+5.1f}; {100*corrected.lower:+5.1f} to {100*corrected.upper:+5.1f})')
    level = 100*row[f'empty_share_{before}']
    cells.append(f"{level:5.1f} -> {level+100*row['empty_share_difference_mean']:5.1f}")
    return ' | '.join(cells)


for cell in [cell for cell in dict.fromkeys(effects.index.get_level_values('cell')) if not cell.startswith('sited_')]:
    first = effects.loc[(cell, 'price_4')]
    print(f"\n{cell.replace('_', ' ')}: original game meets {100*first.human_activity_unchanged:.1f}% of human and "
          f"{100*first.ai_activity_unchanged:.1f}% of AI demand ({int(first.worlds)} worlds)")
    print(' '*14+' | humans: change (95% interval; corrected) | AI: change (95% interval; corrected) | sources nearly empty')
    for design, name in measures.items():
        print(line(name, effects.loc[(cell, design)], 'unchanged', f'{design}_{{group}}_{cell}'))
    print(line('siting', placement.loc[cell], 'random_placement', f'sited_{{group}}_{cell}'))
