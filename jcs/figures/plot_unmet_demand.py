"""What becomes of the demand under the original game and under each measure, at a weak and a strong response of renewal to restraint.

Reads the tables of the local measures (results/local_measures/report), reference setting and theta = 10.
One bar is the demand of every agent as 100%, under the original game, scarcity pricing, the extraction
cap, pricing with the cap, capacity-based siting and siting with pricing, over the same 1500 steps of the
same worlds. What is met is split into what baseline renewal supplies and what renewal above the
baseline rate supplies, which a source produces only when it is neither empty nor full and nearby agents
restrain. Each bar first divides the baseline renewal within reach into what is met from it, what remains up to
the delivery limit (the black mark), and what demand cannot reach; the demand beyond baseline renewal
follows, divided into what renewal above baseline serves and what no renewal serves.
"""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from style import RESULTS, BLUE, LIGHT_BLUE, YELLOW, save

report = RESULTS/'local_measures/report'
placement = pd.read_csv(report/'intervention_placement_effects.csv')
placement = placement[placement.window == '0-1500'].set_index('cell')
effects = pd.read_csv(report/'intervention_effects.csv')
effects = effects[(effects.timing == 'after_development') & (effects.window == '0-1500')].set_index(['cell', 'design'])
panels = {r'renewal response $\theta$ = 3': ('reference', 'sited_all'), r'renewal response $\theta$ = 10': ('t10_p1', 'sited_all_t10')}


def delivered(row, changed=True):
    """Units delivered and units of renewal above baseline per step, under the unchanged game of an effects row or under its measure."""
    total, extra = row.human_delivered_unchanged+row.ai_delivered_unchanged, row.bonus_renewal_unchanged
    if changed:
        total, extra = total+row.human_delivered_difference_mean+row.ai_delivered_difference_mean, extra+row.bonus_renewal_difference_mean
    return total, extra


# baseline renewal within reach comes first: met from it, open up to the delivery limit, out of reach of demand;
# the demand beyond it follows, with the part that renewal above baseline serves.
names = ['met from baseline renewal', 'open to decisions', 'baseline renewal that demand cannot reach', 'met from renewal above baseline', 'beyond all renewal']
colors = [BLUE, YELLOW, LIGHT_BLUE, BLUE, '0.85']
hatches = [None, None, None, '////', None]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained', sharey=True)
for ax, (title, (random_cell, sited_cell)) in zip(axes, panels.items()):
    row = placement.loc[sited_cell]
    bars = [('original game', 'under_random_placement', delivered(effects.loc[(random_cell, 'price_4')], changed=False)),
            ('scarcity pricing', 'under_random_placement', delivered(effects.loc[(random_cell, 'price_4')])),
            ('extraction cap', 'under_random_placement', delivered(effects.loc[(random_cell, 'cap_25')])),
            ('pricing with cap', 'under_random_placement', delivered(effects.loc[(random_cell, 'price_4_cap_25')])),
            ('capacity-based siting', 'sited', delivered(effects.loc[(sited_cell, 'price_4')], changed=False)),
            ('siting with pricing', 'sited', delivered(effects.loc[(sited_cell, 'price_4')]))]
    for position, (label, key, (amount, extra)) in enumerate(bars):
        # demand is the higher request of every agent, 1.5 times the lower requests; siting leaves requests unchanged.
        demand = 1.5*row[f'relative_load_{key}']*row[f'accessible_baseline_renewal_{key}']
        limit, renewal = row[f'delivery_ceiling_baseline_renewal_{key}'], row[f'accessible_baseline_renewal_{key}']
        # renewal above baseline is drawn in steady state, so what baseline renewal supplies is the rest of what is delivered.
        baseline = min(amount-extra, limit)
        parts = [baseline, limit-baseline, renewal-limit, amount-baseline, demand-renewal-(amount-baseline)]
        left = 0
        for part, color, hatch in zip([100*value/demand for value in parts], colors, hatches):
            ax.barh(position, part, left=left, color=color, edgecolor='white', hatch=hatch, height=.62, linewidth=.8)
            if part >= 6:
                # a label on the hatched part gets a solid backing so that it stays legible.
                backing = dict(facecolor=BLUE, edgecolor='none', pad=1.2) if hatch else None
                ax.text(left+part/2, position, f'{part:.0f}%', ha='center', va='center', fontsize=10.5, color='white' if color == BLUE else '0.15', bbox=backing)
            left += part
        ax.plot([100*limit/demand]*2, [position-.4, position+.4], color='k', linewidth=2.2, solid_capstyle='butt')
    ax.set(xlim=(0, 100), ylim=(len(bars)-.45, -.55), yticks=range(len(bars)), xlabel='share of demand (%)', title=title)
    ax.set_yticklabels([label for label, _, _ in bars], fontsize=11)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
handles = [Patch(facecolor=color, edgecolor='white', hatch=hatch, label=name) for name, color, hatch in zip(names, colors, hatches)]
handles.append(Line2D([], [], color='k', linewidth=2.2, label='delivery limit'))
fig.legend(handles=handles, frameon=False, loc='outside upper center', ncols=3, fontsize=10)
save(fig, 'unmet_demand')
