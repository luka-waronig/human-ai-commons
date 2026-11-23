"""The three measures on AI agents in all seven settings, by what each changes for humans and for AI agents together.

Reads the tables of the local measures and of siting in further settings (results/local_measures/report,
results/siting_across_settings/report). A point is one measure in one setting; its position is the change
in the share of demand met over 1500 steps against the original game in the same worlds, with a 95%
interval over worlds. Pricing and the cap start after 1000 steps of the original game; siting compares
the same steps under sited and under random placement of the AI agents. The reference setting is drawn
filled and the other six open, so that each measure shows as a cloud with its exceptions named.
"""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from style import RESULTS, BLUE, YELLOW, RED, both_groups, save

effects = pd.read_csv(RESULTS/'local_measures/report/intervention_effects.csv')
effects = effects[(effects.timing == 'after_development') & (effects.window == '0-1500')].set_index(['cell', 'design'])
placement = pd.concat([pd.read_csv(RESULTS/run/'report/intervention_placement_effects.csv') for run in ['local_measures', 'siting_across_settings']])
placement = placement[(placement.window == '0-1500') & (placement.sited_share == 1)].set_index('random_placement_cell')
settings = ['reference', 't10_p1', 'density_02', 'density_05', 'large_share_50', 'large_share_20', 't3_conf']
measures = {'scarcity pricing': (BLUE, 'o', lambda s: effects.loc[(s, 'price_4')]),
            'extraction cap': (YELLOW, 'D', lambda s: effects.loc[(s, 'cap_25')]),
            'capacity-based siting': (RED, 's', lambda s: placement.loc[s])}
# the settings in which a measure leaves its cloud, named beside the point: text, horizontal offset, vertical offset, alignment.
exceptions = {('extraction cap', 'large_share_50'): ('AI 50% of demand', .7, 0, 'left'),
              ('extraction cap', 'large_share_20'): ('AI 20% of demand', .7, 0, 'left'),
              ('capacity-based siting', 't3_conf'): ('fixed requests', -.7, 1.3, 'right'),
              ('scarcity pricing', 't3_conf'): ('fixed requests', -.7, 0, 'right')}

fig, ax = plt.subplots(figsize=(7.6, 6.4), layout='constrained')
both_groups(ax, [0, 1], [0, 1])
for name, (color, marker, row_of) in measures.items():
    for setting in settings:
        row = row_of(setting)
        x, y = 100*row.human_activity_difference_mean, 100*row.ai_activity_difference_mean
        reference = setting == 'reference'
        ax.errorbar(x, y, xerr=[[x-100*row.human_activity_difference_lower], [100*row.human_activity_difference_upper-x]],
                    yerr=[[y-100*row.ai_activity_difference_lower], [100*row.ai_activity_difference_upper-y]],
                    marker=marker, markersize=9 if reference else 7, markerfacecolor=color if reference else 'white',
                    markeredgecolor=color, markeredgewidth=1.6, ecolor=color, elinewidth=.8, capsize=0, linestyle='none', zorder=4 if reference else 3)
        if (name, setting) in exceptions:
            text, dx, dy, align = exceptions[(name, setting)]
            ax.text(x+dx, y+dy, text, ha=align, va='center', fontsize=9.5, color=color)
ax.set(xlim=(-6, 22), ylim=(-19, 30), xticks=[-5, 0, 5, 10, 15, 20], yticks=[-15, -10, -5, 0, 5, 10, 15, 20, 25, 30])
# one legend inside the plot, in the empty area below "both gain": the measures by color and shape, then the
# settings shown with the markers that appear on the plot, filled for the reference setting and open for the others.
filled = tuple(Line2D([], [], marker=marker, color=color, markersize=7, linestyle='none') for color, marker, _ in measures.values())
open_ = tuple(Line2D([], [], marker=marker, color=color, markerfacecolor='white', markeredgewidth=1.4, markersize=6, linestyle='none')
              for color, marker, _ in measures.values())
handles = [Line2D([], [], marker=marker, color=color, markersize=8, linestyle='none') for color, marker, _ in measures.values()]+[filled, open_]
labels = list(measures)+['reference setting', 'other six settings']
legend = ax.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None, pad=.3)}, handlelength=3.4,
                   loc='upper right', bbox_to_anchor=(.99, .91), frameon=False, fontsize=10)
save(fig, 'local_measures')
