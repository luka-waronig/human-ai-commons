"""Numbers behind the statements on siting: the original game with AI agents sited against randomly placed, in every setting.

Reads the placement tables of results/local_measures and results/siting_across_settings. Shares are
percentages and changes are percentage points over the same 1500 steps of both placements, with a 95%
interval over worlds. The delivery limit, what is delivered and the renewal lost at full sources are in
units per step; they show how much of the change is a higher limit and how much is use of it.
"""
import os
from pathlib import Path
import pandas as pd

RESULTS = Path(os.environ.get('JCS_RESULTS', Path(__file__).resolve().parents[1]/'results'))
table = pd.concat([pd.read_csv(RESULTS/run/'report/intervention_placement_effects.csv') for run in ['local_measures', 'siting_across_settings']])
table = table[table.window == '0-1500']

print('setting, share sited | humans (change, interval) | AI (change, interval) | sources nearly empty | delivery limit | delivered | renewal lost | sources an AI agent reaches | reachable renewal')
for row in table.to_dict('records'):
    cells = [f"{row['random_placement_cell']:<15s} {row['sited_share']:.1f}"]
    for group in ['human', 'ai']:
        level, change = 100*row[f'{group}_activity_random_placement'], 100*row[f'{group}_activity_difference_mean']
        cells.append(f"{level:5.1f} -> {level+change:5.1f} ({change:+5.1f}, {100*row[f'{group}_activity_difference_lower']:+5.1f} to {100*row[f'{group}_activity_difference_upper']:+5.1f})")
    empty = 100*row['empty_share_random_placement']
    cells.append(f"{empty:5.1f} -> {empty+100*row['empty_share_difference_mean']:5.1f}")
    cells.append(f"{row['delivery_ceiling_baseline_renewal_under_random_placement']:5.1f} -> {row['delivery_ceiling_baseline_renewal_sited']:5.1f}")
    delivered = row['human_delivered_random_placement']+row['ai_delivered_random_placement']
    cells.append(f"{delivered:5.1f} -> {delivered+row['human_delivered_difference_mean']+row['ai_delivered_difference_mean']:5.1f}")
    lost = row['spilled_renewal_random_placement']
    cells.append(f"{lost:5.1f} -> {lost+row['spilled_renewal_difference_mean']:5.1f}")
    cells.append(f"{row['ai_reachable_sources_under_random_placement']:4.2f} -> {row['ai_reachable_sources_sited']:4.2f}")
    cells.append(f"{row['accessible_baseline_renewal_under_random_placement']:5.1f} -> {row['accessible_baseline_renewal_sited']:5.1f}")
    print(' | '.join(cells))
