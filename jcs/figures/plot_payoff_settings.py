"""Every combination of the two payoff parameters, by what it changes for humans and for AI agents together.

Reads the table of the payoff grid (results/payoff_grid/report/search_map.csv). Each point is one
combination of u and v; its position is the change in the share of demand met over the 1500 steps after
the payoffs change, against the original game in the same worlds. The original game is the cross at the
origin. Both panels have the same axes, one percentage point in every direction.
"""
import matplotlib.pyplot as plt
import pandas as pd
from style import RESULTS, BLUE, both_groups, save

table = pd.read_csv(RESULTS/'payoff_grid/report/search_map.csv')
table = table[table.admissible & ~((table.r_c == 0) & (table.r_d == 0))]
titles = {'reference': r'renewal response $\theta$ = 3', 't10_p1': r'renewal response $\theta$ = 10'}

fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout='constrained')
for ax, (setting, title) in zip(axes, titles.items()):
    part = table[table.cell == setting]
    x, y = 100*part['human_activity_0-1500_difference_mean'], 100*part['ai_activity_0-1500_difference_mean']
    both_groups(ax, list(x), list(y))
    ax.scatter(x, y, s=16, color=BLUE, alpha=.75, linewidths=0)
    ax.set(title=title, xlim=(-3, 3), ylim=(-3, 3))
save(fig, 'payoff_settings')
