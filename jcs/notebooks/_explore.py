"""Read-only bundle access and plotting conventions for the JCS notebooks."""
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from human_ai_commons.world import World
from human_ai_commons.config import welfare_spec
from human_ai_commons.metrics import supported_activity

BLUE = '#244867'
TEAL = '#278477'
ORANGE = '#bc703a'
GRAY = '#87929c'


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.titlesize': 12, 'axes.labelsize': 11, 'axes.spines.top': False,
        'axes.spines.right': False, 'figure.dpi': 110, 'savefig.dpi': 140,
        'axes.titlepad': 13, 'grid.alpha': .15})


def frame_map(ax, world, stocks, strategies, title):
    field = np.full((world.spec.height, world.spec.width), np.nan)
    field[world.resources[:, 1], world.resources[:, 0]] = stocks/world.capacity
    cmap = plt.get_cmap('YlGnBu').copy()
    cmap.set_bad('#f1f3f4')
    im = ax.imshow(np.ma.masked_invalid(field), origin='lower', cmap=cmap,
                   vmin=0, vmax=1, extent=(-.5, world.spec.width-.5, -.5, world.spec.height-.5))
    # marker offsets separate agents sharing one grid cell; locations remain fixed.
    offsets = (np.arange(len(world.positions)) % 3 - 1)*.14
    for kind, marker in [(False, 'o'), (True, '^')]:
        mask = world.kinds == kind
        colors = np.where(strategies[mask], BLUE, ORANGE)
        ax.scatter(world.positions[mask, 0]+offsets[mask], world.positions[mask, 1]+offsets[mask],
                   c=colors, marker=marker, s=48, edgecolor='white', linewidth=.7, zorder=3)
    ax.set(title=title, xlabel='Grid x', ylabel='Grid y', xticks=range(0,world.spec.width,2),
           yticks=range(0,world.spec.height,2))
    return im


def map_legend():
    return [Line2D([], [], marker='o', color='none', markerfacecolor=GRAY, label='Human', markersize=7),
            Line2D([], [], marker='^', color='none', markerfacecolor=GRAY, label='AI', markersize=7),
            Line2D([], [], marker='s', color='none', markerfacecolor=BLUE, label='Lower request', markersize=7),
            Line2D([], [], marker='s', color='none', markerfacecolor=ORANGE, label='Higher request', markersize=7)]


class Bundle:
    """A JCS result bundle. No method writes to it or executes the model."""
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.manifest = json.loads(self.file('manifest.json').read_text())
        self.config = self.manifest['configuration']
        self.run_id = self.manifest['run_id']
        self._tests = None

    def file(self, relative):
        path = (self.path/relative).resolve()
        if not path.is_relative_to(self.path):
            raise ValueError('Bundle paths must stay inside the selected folder')
        if not path.is_file():
            raise FileNotFoundError(f'Missing saved input: {relative}')
        return path

    def table(self, name):
        return pd.read_csv(self.file('analysis/'+name))

    @property
    def tests(self):
        if self._tests is None:
            self._tests = json.loads(self.file('test_comparisons.json').read_text())
        return self._tests

    def problems(self):
        return [g['problem'] for g in self.tests['problems']]

    def designs(self, problem):
        g = next(g for g in self.tests['problems'] if g['problem'] == problem)
        families = g.get('families')
        candidates = [p for group in families.values() for p in group] if families else list(g['policies'].values())
        return {p['policy_id']: p for p in candidates}

    def matched(self, problem, policy_id, reference_id):
        designs = self.designs(problem)
        candidate, reference = designs[policy_id], designs[reference_id]
        if candidate['world_ids'] != reference['world_ids']:
            raise ValueError('These designs do not use matching worlds in the same order')
        for p in [candidate, reference]:
            if len(p['realization_ids']) != len(p['world_ids']):
                raise ValueError('Inconsistent saved realization inventory')
        return candidate, reference

    @lru_cache(maxsize=4)
    def history(self, evaluation_id, world_id):
        # bounded cache: selected histories, not the full trajectory archive.
        world = World.load(self.file(f'worlds/{world_id}.npz'))
        with np.load(self.file(f'trajectories/{evaluation_id}.npz'), allow_pickle=False) as f:
            data = {key: f[key] for key in f.files}
        if data['stocks'].shape[1] != len(world.resources):
            raise ValueError('World and trajectory source counts disagree')
        return world, data

    def trace(self, data, metric, world):
        if metric == 'stock':
            return (data['stocks'][:, data['accessible']]/world.capacity[data['accessible']]).mean(axis=1)
        if metric == 'cooperation':
            return data['strategies'].mean(axis=1)
        if metric not in ['human', 'ai']:
            raise ValueError('Unknown trajectory measurement')
        mask = ~data['agent_is_ai'] if metric == 'human' else data['agent_is_ai']
        ratio = self.config['world']['defect_to_cooperate_request_ratio']
        spec = welfare_spec(self.config)
        baseline = data['agent_reference_request'][mask]
        value = supported_activity(data['agent_delivered'][:, mask], baseline, ratio*baseline,
                                   spec.additional_use_value if spec else 1)
        if spec and spec.utility == 'log1p':
            value = np.log1p(value)/np.log(2)
        return value.mean(axis=1)
