"""Shared look of the figures and the folders they read from and write to."""
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# results of the runs; another folder can be named in JCS_RESULTS.
RESULTS = Path(os.environ.get('JCS_RESULTS', Path(__file__).resolve().parents[1]/'results'))
FIGURES = RESULTS/'figures'
BLUE, LIGHT_BLUE, DARK, YELLOW, RED = '#33658a', '#86bbd8', '#2f4858', '#f6ae2d', '#d1402b'

plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 12, 'axes.labelsize': 12, 'axes.titlesize': 12,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES/f'{name}.pdf', bbox_inches='tight', metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(FIGURES/f'{name}.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print('saved', FIGURES/f'{name}.pdf')


def both_groups(ax, x, y):
    """Frame for a change that concerns both groups: the original game at the origin, the four regions named."""
    limits = []
    for values in [x, y]:
        low, high = min(0, *values), max(0, *values)
        width = max(high-low, 1e-6)
        limits.append((min(low, -.25*width)-.12*width, max(high, .25*width)+.12*width))
    ax.axhline(0, color='0.3', linewidth=.8)
    ax.axvline(0, color='0.3', linewidth=.8)
    ax.plot(0, 0, marker='+', color='k', markersize=13, markeredgewidth=1.8, linestyle='none', label='original game', zorder=5)
    corners = [('both gain', 'right', 'top'), ('humans gain, AI loses', 'right', 'bottom'),
               ('both lose', 'left', 'bottom'), ('AI gains, humans lose', 'left', 'top')]
    for text, h, v in corners:
        ax.text(.985 if h == 'right' else .015, .985 if v == 'top' else .015, text, transform=ax.transAxes, ha=h, va=v, fontsize=9, color='0.3')
    ax.set(xlim=limits[0], ylim=limits[1], xlabel='change in share of demand met for humans\n(% of demand)',
           ylabel='change in share of demand met for AI\n(% of demand)')
