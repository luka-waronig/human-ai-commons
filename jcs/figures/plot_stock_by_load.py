"""Late stock of every source against how much is asked of it.

Reads the saved source states of the original game (results/original_game), reference setting. Each
point is one source in one world. The expected load is what agents would ask of the source per step, at
the lower request and with a random choice of source, divided by its baseline renewal; loads of 3 or
more are drawn at 3. The stock is the mean over steps 1500 to 2000.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from human_ai_commons.paper_experiments import CLASSES
from human_ai_commons.paper_tables import array_file
from style import RESULTS, LIGHT_BLUE, YELLOW, DARK, save

run = RESULTS/'original_game'
worlds = range(json.loads((run/'manifest.json').read_text())['configuration']['worlds'])
colors = dict(zip(CLASSES, [LIGHT_BLUE, YELLOW, DARK]))

fig, ax = plt.subplots(figsize=(7, 4.6), layout='constrained')
for index in worlds:
    with np.load(array_file(run, 'continuity', 'reference', index), allow_pickle=False) as saved:
        load, classes, stock = (saved[f'journal_reference::{key}'] for key in ['load', 'classes', 'late_stock'])
    for code, name in enumerate(CLASSES):
        ax.scatter(np.minimum(load[classes == code], 3), 100*stock[classes == code], s=6, alpha=.3, color=colors[name], linewidths=0)
for name in CLASSES:
    ax.scatter([], [], s=28, color=colors[name], label=name)
ax.legend(frameon=False, loc='upper right', title='source')
ax.axhline(10, color='0.4', linewidth=.8, linestyle=':')
ax.text(0, 11.5, 'nearly empty below this line', color='0.3', fontsize=9, ha='left')
ax.set(xlabel='expected load of the source (requests relative to baseline renewal)', ylabel='stock in steps 1500 to 2000 (% of capacity)',
       xlim=(-.05, 3.05), ylim=(-3, 103), xticks=[0, .5, 1, 1.5, 2, 2.5, 3], xticklabels=['0', '0.5', '1', '1.5', '2', '2.5', '3 or more'])
save(fig, 'stock_by_load')
