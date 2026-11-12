from dataclasses import dataclass, replace
import json
from pathlib import Path
import numpy as np
from .provenance import canonical, identifier, rng


def torus_distances(a, b, width, height):
    delta = np.abs(np.asarray(a)[:, None, :]-np.asarray(b)[None, :, :])
    delta = np.minimum(delta, np.array([width, height])-delta)
    return delta.max(axis=-1)


@dataclass(frozen=True)
class WorldSpec:
    width: int = 20
    height: int = 20
    humans: int = 50
    ai: int = 50
    density: float = .3
    capacity: float = 20
    renewal_low: float = .1
    renewal_high: float = .5
    maximum_occupancy: int = 3
    access_radius: int = 1
    feedback_radius: int = 3
    game_radius: int = 3
    initial_cooperation: float = .5
    require_access: bool = True

    def __post_init__(self):
        ints = [self.width, self.height, self.humans, self.ai, self.maximum_occupancy,
                self.access_radius, self.feedback_radius, self.game_radius]
        if any(isinstance(v, bool) or not isinstance(v, int) for v in ints):
            raise ValueError('dimensions, counts, and radii must be integers')
        if min(self.width, self.height, self.maximum_occupancy) < 1 or min(ints[2:4]+ints[5:]) < 0:
            raise ValueError('invalid dimensions, population, or radius')
        n = self.humans+self.ai
        if n < 1 or n > self.maximum_occupancy*self.width*self.height:
            raise ValueError('population must fit the grid and be positive')
        numbers = [self.density, self.capacity, self.renewal_low, self.renewal_high, self.initial_cooperation]
        if not all(np.isfinite(v) for v in numbers) or not 0 < self.density <= 1 or self.capacity <= 0:
            raise ValueError('invalid resource parameters')
        if not 0 < self.renewal_low <= self.renewal_high or not 0 <= self.initial_cooperation <= 1:
            raise ValueError('invalid renewal or initial cooperation')


@dataclass(frozen=True)
class World:
    spec: WorldSpec
    resources: np.ndarray
    capacity: np.ndarray
    renewal: np.ndarray
    positions: np.ndarray
    kinds: np.ndarray
    initial: np.ndarray
    draw_attempts: int = 1

    def __post_init__(self):
        # Owned immutable arrays prevent policies from modifying shared worlds.
        for name, dtype in [('resources', int), ('capacity', float), ('renewal', float),
                            ('positions', int), ('kinds', bool), ('initial', bool)]:
            raw = np.asarray(getattr(self, name))
            if dtype is int and (not np.all(np.isfinite(raw)) or not np.all(raw == np.floor(raw))):
                raise ValueError('coordinates must be finite integers')
            if dtype is bool and not np.all(np.isin(raw, [0, 1])):
                raise ValueError('types and strategies must be binary')
            value = np.array(raw, dtype=dtype, copy=True)
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        m, n = len(self.resources), self.spec.humans+self.spec.ai
        if m < 1 or self.resources.shape != (m, 2) or self.positions.shape != (n, 2):
            raise ValueError('invalid world geometry')
        if self.capacity.shape != (m,) or self.renewal.shape != (m,) or self.kinds.shape != (n,) or self.initial.shape != (n,):
            raise ValueError('inconsistent world arrays')
        for a in [self.resources, self.positions]:
            if np.any(a < 0) or np.any(a >= [self.spec.width, self.spec.height]):
                raise ValueError('coordinate outside torus')
        if len(np.unique(self.resources, axis=0)) != m or self.kinds.sum() != self.spec.ai:
            raise ValueError('duplicate resources or inconsistent type counts')
        if (not np.all(np.isfinite(self.capacity)) or not np.all(np.isfinite(self.renewal))
                or np.any(self.capacity <= 0) or np.any(self.renewal < 0)):
            raise ValueError('invalid stocks or rates')
        _, counts = np.unique(self.positions, axis=0, return_counts=True)
        if counts.max() > self.spec.maximum_occupancy:
            raise ValueError('occupancy limit exceeded')

    @property
    def access(self):
        return torus_distances(self.positions, self.resources, self.spec.width, self.spec.height) <= self.spec.access_radius

    @property
    def id(self):
        return identifier(dict(spec=self.spec, resources=self.resources, capacity=self.capacity,
                               renewal=self.renewal, positions=self.positions, kinds=self.kinds,
                               initial=self.initial, draw_attempts=self.draw_attempts))[:24]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, spec=canonical(self.spec), resources=self.resources, capacity=self.capacity,
                            renewal=self.renewal, positions=self.positions, kinds=self.kinds,
                            initial=self.initial, draw_attempts=self.draw_attempts)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as a:
            return cls(WorldSpec(**json.loads(str(a['spec']))), **{k: a[k] for k in
                       ['resources', 'capacity', 'renewal', 'positions', 'kinds', 'initial']},
                       draw_attempts=int(a['draw_attempts']))


def initial_strategies(kinds, fraction, generator):
    strategies = np.zeros(len(kinds), dtype=bool)
    for kind in [False, True]:
        group = np.flatnonzero(kinds == kind)
        count = int(np.floor(fraction*len(group)+.5))
        strategies[generator.permutation(group)[:count]] = True
    return strategies


def generate_world(spec, world_seed):
    """Placement conditioning on access for a given landscape."""
    generator = rng(world_seed, 'landscape-and-placement')
    cells = np.array([(x, y) for x in range(spec.width) for y in range(spec.height)])
    n = spec.humans+spec.ai
    for attempt in range(1, 10001):
        resources = cells[generator.random(len(cells)) < spec.density]
        if not len(resources):
            continue
        eligible = cells
        if spec.require_access:
            near = torus_distances(cells, resources, spec.width, spec.height) <= spec.access_radius
            eligible = cells[near.any(axis=1)]
        if len(eligible)*spec.maximum_occupancy < n:
            continue
        counts = np.zeros(len(eligible), dtype=int)
        chosen = []
        for _ in range(n):
            # Uniform among positions with remaining occupancy, without a retry cap bias.
            available = np.flatnonzero(counts < spec.maximum_occupancy)
            index = available[generator.integers(len(available))]
            counts[index] += 1
            chosen.append(eligible[index])
        kinds = np.arange(n) >= spec.humans
        initial = initial_strategies(kinds, spec.initial_cooperation, rng(world_seed, 'initial-strategies'))
        renewal = generator.uniform(spec.renewal_low, spec.renewal_high, len(resources))
        return World(spec, resources, np.full(len(resources), spec.capacity), renewal,
                     np.array(chosen), kinds, initial, attempt)
    raise ValueError('could not generate a landscape with sufficient accessible placement capacity')


def transform_world(world, *, initial_fraction=None, cluster_ai=False, transform_seed=0):
    kinds = world.kinds.copy()
    if cluster_ai:
        center = np.array([[world.spec.width/2, world.spec.height/2]])
        distances = torus_distances(world.positions, center, world.spec.width, world.spec.height)[:, 0]
        order = np.argsort(distances, kind='stable')
        kinds[:] = False
        kinds[order[:world.spec.ai]] = True
    fraction = world.spec.initial_cooperation if initial_fraction is None else initial_fraction
    # Original per-position strategies remain fixed when only demand labels change.
    initial = world.initial if initial_fraction is None else initial_strategies(kinds, fraction, rng(transform_seed, 'initial-strategies'))
    return replace(world, spec=replace(world.spec, initial_cooperation=fraction), kinds=kinds, initial=initial)
