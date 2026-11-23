"""Numbers behind the statements on the payoff grid: how many combinations of u and v help both groups, one, or neither.

Reads results/payoff_grid/report/search_map.csv and prints, for each setting, the share of demand
met under the original game and the changes over the 1500 steps after the payoffs change. The counts are
of mean changes over worlds, whatever their size; most are far below a tenth of a percentage point.
"""
import os
from pathlib import Path
import pandas as pd

RESULTS = Path(os.environ.get('JCS_RESULTS', Path(__file__).resolve().parents[1]/'results'))
table = pd.read_csv(RESULTS/'payoff_grid/report/search_map.csv')
table = table[table.admissible]
whole = [c for c in table.columns if c.startswith('human_activity_') and c.endswith('_difference_mean')][0][len('human_activity_'):-len('_difference_mean')]

for setting, part in table.groupby('cell', sort=False):
    original = part[(part.r_c == 0) & (part.r_d == 0)].iloc[0]
    others = part[(part.r_c != 0) | (part.r_d != 0)]
    human = 100*others[f'human_activity_{whole}_difference_mean']
    ai = 100*others[f'ai_activity_{whole}_difference_mean']
    print(f'\n{setting}: {len(others)} combinations besides the original game, {int(original.worlds)} worlds, steps {whole} after the change')
    human_level, ai_level = 100*original[f'human_activity_{whole}'], 100*original[f'ai_activity_{whole}']
    print(f'  original game: share of demand met {human_level:.1f}% for humans, {ai_level:.1f}% for AI')
    print(f'  raise both {int(((human > 0) & (ai > 0)).sum())} | raise one and lower the other {int(((human > 0) != (ai > 0)).sum())} | lower both {int(((human <= 0) & (ai <= 0)).sum())}')
    both = others[(human > 0) & (ai > 0)]
    if len(both):
        best = both.loc[(both[f'human_activity_{whole}_difference_mean']+both[f'ai_activity_{whole}_difference_mean']).idxmax()]
        human_gain, ai_gain = 100*best[f'human_activity_{whole}_difference_mean'], 100*best[f'ai_activity_{whole}_difference_mean']
        print(f'  largest gain for both: humans {human_gain:+.3f}, AI {ai_gain:+.3f} percentage points'
              f' (switching levels {best.q_c:.2f} and {best.q_d:.2f})')
    print(f'  range over all combinations: humans {human.min():+.2f} to {human.max():+.2f}, AI {ai.min():+.2f} to {ai.max():+.2f} percentage points')
    equal = others[(others.r_c-others.r_d).abs() < 1e-9].sort_values('q_c')
    print('  when both parameters change equally (switching level: change for humans, for AI):')
    equal = equal.rename(columns={f'human_activity_{whole}_difference_mean': 'human_change', f'ai_activity_{whole}_difference_mean': 'ai_change'})
    print('   ', '; '.join(f'{row.q_c:.2f}: {100*row.human_change:+.2f}, {100*row.ai_change:+.2f}' for row in equal.itertuples()))
