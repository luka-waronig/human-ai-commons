"""Cooperation over the first 600 steps for three source capacities, with the first 100 steps marked.

Reads the step-by-step series of the original game (results/original_game), reference setting. Each
line is the mean over worlds. In the left panel every source starts full, so a larger source starts
with more; in the right panel every source starts with the same volume.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from human_ai_commons.paper_tables import mean_series
from style import RESULTS, LIGHT_BLUE, BLUE, DARK, save

run = RESULTS/'original_game'
config = json.loads((run/'manifest.json').read_text())['configuration']
worlds = range(config['worlds'])
capacities, volume = config['continuity']['capacities'], config['continuity']['equal_initial_volume']
colors = dict(zip(capacities, [LIGHT_BLUE, BLUE, DARK]))
steps = 600

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, layout='constrained')
for ax, (start, title) in zip(axes, [('full', 'every source starts full'), ('equal_volume', f'every source starts with {volume:g} units')]):
    for capacity, color in colors.items():
        values = 100*mean_series(run, 'continuity', 'reference', worlds, f'storage_journal_K{capacity}_{start}', 'cooperation')[:steps]
        ax.plot(np.arange(1, len(values)+1), values, color=color, linewidth=1.5, label=f'{capacity}')
    ax.axvline(100, color='0.4', linewidth=.8, linestyle=':')
    ax.text(104, 22, '100 steps', color='0.3', fontsize=9, va='bottom')
    ax.set(title=title, xlabel='step', xlim=(0, steps), ylim=(20, 90))
axes[0].set_ylabel('cooperation (% of agents that restrain)')
axes[1].legend(frameon=False, loc='lower right', title='source capacity (units)')
save(fig, 'persistence')
