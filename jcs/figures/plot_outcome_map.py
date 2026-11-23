"""The outcomes drawn as a map: where each measure takes both groups from the original game.

Reads the tables of the local measures and of the payoff grid. East is a larger share of demand met for
human agents and north a larger share for AI agents, in percentage points, at one scale in both
directions. The flag is the original game; the small cluster on it is every combination of the payoff
parameters. Each route ends at the measured position of one measure in the reference setting: the two
black routes and the red one are the three measures, a black route continues the cap to the cap combined
with scarcity pricing, and the gray route continues the cap to a floor of half of capacity. The routes are drawn curved for the look of a map; only their end points are data.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Rectangle
from style import RESULTS, RED, save

INK, GRAY, SOFT = '#141414', '#8a8a8a', '#b4b4b4'
effects = pd.read_csv(RESULTS/'local_measures/report/intervention_effects.csv')
effects = effects[(effects.timing == 'after_development') & (effects.window == '0-1500') & (effects.cell == 'reference')].set_index('design')
placement = pd.read_csv(RESULTS/'local_measures/report/intervention_placement_effects.csv')
placement = placement[(placement.window == '0-1500') & (placement.random_placement_cell == 'reference')].set_index('cell')
at = lambda row: (100*float(row.human_activity_difference_mean), 100*float(row.ai_activity_difference_mean))
price, cap, strict, sited = at(effects.loc['price_4']), at(effects.loc['cap_25']), at(effects.loc['cap_50']), at(placement.loc['sited_all'])
combined = at(effects.loc['price_4_cap_25'])
grid = pd.read_csv(RESULTS/'payoff_grid/report/search_map.csv')
grid = grid[(grid.cell == 'reference') & grid.admissible]

with plt.rc_context({'font.family': ['Monaco', 'DejaVu Sans Mono'], 'font.size': 13.5}):
    fig = plt.figure(figsize=(8.1, 7.75))
    ax = fig.add_axes([.02, .02, .96, .96])
    X0, X1, Y0, Y1 = -14.6, 26.6, -14.2, 25.2
    ax.set(xlim=(X0, X1), ylim=(Y0, Y1), aspect='equal')
    ax.axis('off')
    ax.add_patch(Rectangle((X0+.2, Y0+.2), X1-X0-.4, Y1-Y0-.4, fill=False, edgecolor=INK, linewidth=1.4))
    ax.plot([X0+.2, X1-.2], [0, 0], color=SOFT, linewidth=1)
    ax.plot([0, 0], [Y0+.2, Y1-.2], color=SOFT, linewidth=1)
    for v in [10, 20]:
        ax.plot([v, v], [-.4, .4], color=SOFT, linewidth=1)
        ax.text(v, -1.0, f'+{v}', ha='center', va='top', color=GRAY, fontsize=12)
    for v in [-10, 10, 20]:
        ax.plot([-.4, .4], [v, v], color=SOFT, linewidth=1)
        ax.text(-.8, v, f'{v:+d}', ha='right', va='center', color=GRAY, fontsize=12)

    def trail(a, m, b, color):
        t = np.linspace(0, 1, 80)
        ax.plot((1-t)**2*a[0]+2*(1-t)*t*m[0]+t**2*b[0], (1-t)**2*a[1]+2*(1-t)*t*m[1]+t**2*b[1], color=color, linewidth=2.4,
                linestyle=(0, (1, 2.2)), dash_capstyle='butt', zorder=3)

    trail((0, 0), (1.2, 12), sited, RED)
    trail((0, 0), (0, 2.5), price, INK)
    trail((0, 0), (5, .2), cap, INK)
    trail(cap, (20.5, 1.9), strict, GRAY)
    trail(cap, (11.6, 3.9), combined, INK)
    for (x, y), color in [(price, INK), (cap, INK), (combined, INK), (strict, GRAY)]:
        ax.scatter([x], [y], marker='s', s=80, color=color, zorder=5)
    for sign in [1, -1]:
        ax.plot([sited[0]-.6, sited[0]+.6], [sited[1]-.6*sign, sited[1]+.6*sign], color=RED, linewidth=4.5, solid_capstyle='butt', zorder=6)
    ax.scatter(100*grid['human_activity_0-1500_difference_mean'], 100*grid['ai_activity_0-1500_difference_mean'], marker='s', s=9, color=INK, linewidths=0, zorder=6)
    ax.plot([0, 0], [0, 2.0], color=INK, linewidth=2.2, zorder=6)
    ax.fill([0, 1.5, 0], [2.0, 1.5, 1.0], color=INK, zorder=6)

    ink, gray = dict(color=INK, zorder=7), dict(color=GRAY, zorder=7)
    ax.text(price[0]-.9, price[1], 'scarcity pricing', ha='right', va='center', **ink)
    ax.text(cap[0]+.9, cap[1]+.8, 'extraction cap', ha='left', va='bottom', **ink)
    ax.text(sited[0]+1.3, sited[1], 'capacity-based siting', ha='left', va='center', **ink)
    ax.text(strict[0], strict[1]-.9, 'strict cap', ha='center', va='top', **gray)
    ax.text(combined[0]+.9, combined[1]+.2, 'pricing with cap', ha='left', va='center', **ink)
    ax.plot([.5, 1.9], [-.5, -2.7], color=INK, linewidth=1, zorder=5)
    ax.text(1.7, -2.9, 'START HERE', ha='left', va='top', **ink)
    ax.text(X1-.9, Y1-.9, 'BOTH GAIN', ha='right', va='top', **ink)
    ax.text(X1-.9, Y0+.8, 'HUMANS GAIN, AI LOSES', ha='right', va='bottom', **ink)
    ax.text(X0+.8, Y0+.8, 'BOTH LOSE', ha='left', va='bottom', **ink)
    ax.text(X0+.8, Y1-.9, 'AI GAINS,\nHUMANS LOSE', ha='left', va='top', linespacing=1.4, **ink)

    # a compass in the corner: north is a gain for AI, east a gain for humans, each named on its own side.
    cx, cy, r = -10.9, -6.0, 1.7
    ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor=INK, linewidth=1.4))
    ax.fill([cx, cx-.5, cx+.5], [cy+r+1.0, cy, cy], color=INK)
    ax.fill([cx+r+1.0, cx, cx], [cy, cy-.5, cy+.5], facecolor='white', edgecolor=INK, linewidth=1.3)
    ax.text(cx, cy+r+1.5, 'AI gains', ha='center', va='bottom', **ink)
    ax.text(cx+r+1.5, cy, 'humans gain', ha='left', va='center', **ink)
    bar = -8.6
    ax.plot([10, 20], [bar, bar], color=INK, linewidth=1.8)
    for x in [10, 20]:
        ax.plot([x, x], [bar-.4, bar+.4], color=INK, linewidth=1.8)
    ax.text(15, bar-.7, '10 pp', ha='center', va='top', **ink)
    save(fig, 'outcome_map')
