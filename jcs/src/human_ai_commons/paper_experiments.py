"""Experiments of the paper: the original game over time, payoff changes, renewal controls, the payoff grid, and local measures on AI agents.

This protocol is separate from the staged payoff search in experiments.py. It
has its own configuration, random-input split and output folder and never
writes to a search bundle. Every condition states what it changes; matched
conditions share the world and the indexed random inputs at the same absolute step.
"""
from dataclasses import asdict, dataclass, replace
import json
from multiprocessing import get_context
import os
from pathlib import Path
import signal
import numpy as np
from .config import load_config, scenario as reference_scenario, world_spec
from .dynamics import Drivers, RenewalControl, Scenario, Simulator
from .payoffs import Policy
from .provenance import append_json, canonical, environment, identifier, read_jsonl, seed, source_hashes, utc_now, write_json
from .world import WorldSpec, generate_world, torus_distances

EXPERIMENTS = ('continuity', 'interventions', 'feedback')
SEARCH_EXPERIMENTS = ('search', 'search_test')
# the local measures run first, so that their results are available early in a full run.
PAPER_RUNS = ('local_measures', 'original_game', 'original_game_long_horizon', 'model_assumptions',
              'payoff_grid', 'agent_assumptions')
SERIES = ('human_activity', 'ai_activity', 'cooperation', 'stock', 'contested_stock', 'renewal', 'bonus_renewal',
          'spilled_renewal', 'human_delivered', 'ai_delivered', 'below_half_share', 'contested_empty_share_per_step',
          'stock_all_sources', 'human_scarcity', 'ai_scarcity',
          'human_requested', 'ai_requested', 'human_cooperation', 'ai_cooperation')
CLASSES = ('underused', 'contested', 'overloaded')


def local_load(world, cooperative_request):
    """Expected ordinary request reaching each source divided by its baseline renewal.

    An agent chooses uniformly among its accessible sources that hold stock, so
    with all of them available its ordinary request is spread equally over its
    access set. The result depends on geometry and requests, never on payoffs.
    """
    access = world.access
    count = access.sum(axis=1)
    share = np.divide(np.asarray(cooperative_request, dtype=float), count, out=np.zeros(len(count)), where=count > 0)
    return (access*share[:, None]).sum(axis=0)/world.renewal


def delivery_ceiling(world, target, multiplier=1.):
    """Largest total delivery per step that any choice of sources and actions could sustain.

    A transport problem on the fixed access network: each agent receives at most
    its target, each source gives at most its renewal times `multiplier`, and
    stocks do not change. No payoff design, source-selection rule or allocation
    rule can sustain more with the same access, requests and renewal.
    """
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    pairs = np.argwhere(world.access)
    if not len(pairs):
        return 0.
    agents, sources, columns = len(world.kinds), len(world.renewal), np.arange(len(pairs))
    limits = coo_matrix((np.ones(2*len(pairs)), (np.r_[pairs[:, 0], agents+pairs[:, 1]], np.r_[columns, columns])),
                        shape=(agents+sources, len(pairs)))
    result = linprog(-np.ones(len(pairs)), A_ub=limits, b_ub=np.r_[np.asarray(target, dtype=float), world.renewal*multiplier],
                     bounds=(0, None), method='highs')
    if result.status != 0:
        raise ArithmeticError('delivery ceiling could not be computed')
    return float(-result.fun)


def maximum_multiplier(scenario):
    gain = scenario.theta if scenario.feedback == 'manuscript' else scenario.theta-1
    return float(np.clip(1+.125*gain, 1 if scenario.feedback == 'repository' else .1, 3))


def load_classes(world, cooperative_request, scenario):
    """0 underused, 1 contested, 2 overloaded; defined before any simulation.

    underused: expected requests stay within baseline renewal even if every user
    intensifies. overloaded: expected ordinary requests exceed the largest renewal
    the feedback law can produce. contested: choices can matter for the stock.
    Action reversals and source-selection noise blur the two boundaries.
    """
    load = local_load(world, cooperative_request)
    classes = np.where(load*scenario.defect_ratio <= 1, 0, np.where(load > maximum_multiplier(scenario), 2, 1))
    return load, classes


@dataclass(frozen=True)
class State:
    """Everything a continuation needs besides the world, scenario and policy."""
    step: int
    water: np.ndarray
    strategies: np.ndarray


@dataclass(frozen=True)
class Disturbance:
    """A common temporary disturbance.

    renewal_scale multiplies every source's renewal during [start, stop).
    stock_scale multiplies every stock once, immediately before step start.
    """
    kind: str
    start: int
    stop: int
    value: float

    def __post_init__(self):
        if self.kind not in ['renewal_scale', 'stock_scale'] or not 0 <= self.value <= 1 or self.stop <= self.start:
            raise ValueError('invalid disturbance')


class Recorder:
    """Per-step ensemble series and per-window source and agent summaries."""
    def __init__(self, simulator, windows, start, stop):
        self.sim, world = simulator, simulator.world
        self.start, self.windows = start, [(str(name), int(a), int(b)) for name, a, b in windows]
        if any(not start <= a < b <= stop for _, a, b in self.windows):
            raise ValueError('measurement windows must lie inside the simulated segment')
        self.accessible = simulator.accessible
        self.load, classes = load_classes(world, simulator.cooperative_request, simulator.scenario)
        self.classes = np.where(self.accessible, classes, -1)
        self.target = simulator.cooperative_request*simulator.scenario.defect_ratio
        self.series = np.full((stop-start, len(SERIES)), np.nan)
        m, n = len(world.renewal), len(world.kinds)
        self.sums = [dict(count=0, stock=np.zeros(m), renewal=np.zeros(m), cooperation=np.zeros(m),
                          activity=np.zeros(n), delivered=np.zeros(n), scarcity=np.zeros(n), chosen=np.zeros(n)) for _ in self.windows]

    def update(self, t, event):
        sim, world = self.sim, self.sim.world
        kinds, acc, contested = world.kinds, self.accessible, self.classes == 1
        n = sim.water/world.capacity
        activity = np.minimum(event['delivered']/self.target, 1)
        baseline = world.renewal*sim.renewal_scale
        # scarcity of the source an agent drew from this step: one minus its stock level after renewal.
        chosen = event['targets'] >= 0
        scarcity = np.zeros(len(kinds))
        scarcity[chosen] = 1-event['source_level'][event['targets'][chosen]]
        self.series[t-self.start] = [
            activity[~kinds].mean() if (~kinds).any() else np.nan, activity[kinds].mean() if kinds.any() else np.nan,
            sim.strategies.mean(), n[acc].mean(), n[contested].mean() if contested.any() else np.nan,
            event['resource_renewal'][acc].sum(), (event['potential_renewal']-baseline)[acc].sum(),
            (event['potential_renewal']-event['resource_renewal'])[acc].sum(),
            event['delivered'][~kinds].sum(), event['delivered'][kinds].sum(), (n[acc] < .5).mean(),
            (n[contested] < .1).mean() if contested.any() else np.nan, n.mean(),
            scarcity[chosen & ~kinds].mean() if (chosen & ~kinds).any() else np.nan,
            scarcity[chosen & kinds].mean() if (chosen & kinds).any() else np.nan,
            # what each group asked for and the share of it on the lower request, so that asking less and receiving less can be told apart.
            event['requested'][~kinds].sum(), event['requested'][kinds].sum(),
            sim.strategies[~kinds].mean() if (~kinds).any() else np.nan, sim.strategies[kinds].mean() if kinds.any() else np.nan]
        local = None
        for (_, a, b), total in zip(self.windows, self.sums):
            if a <= t < b:
                if local is None:
                    local = np.divide(sim.ecological @ sim.strategies.astype(float), sim.ecological_count,
                                      out=np.zeros(len(n)), where=sim.ecological_count > 0)
                total['count'] += 1
                total['stock'] += n
                total['renewal'] += event['resource_renewal']
                total['cooperation'] += local
                total['activity'] += activity
                total['delivered'] += event['delivered']
                total['scarcity'] += scarcity
                total['chosen'] += chosen

    def finish(self):
        world, acc, kinds = self.sim.world, self.accessible, self.sim.world.kinds
        access = world.access
        reach = access.sum(axis=1)
        over = (access & (self.classes == 2)[None, :]).sum(axis=1)
        exposure = np.where(reach == 0, 'no_access', np.where(over == reach, 'only_overloaded',
                            np.where(over == 0, 'no_overloaded', 'mixed')))
        rows, arrays = [], {}
        for (name, a, b), total in zip(self.windows, self.sums):
            if total['count'] != b-a:
                raise ArithmeticError('incomplete measurement window')
            part = self.series[a-self.start:b-self.start]
            seen = np.isfinite(part).sum(axis=0)
            means = np.divide(np.nansum(part, axis=0), seen, out=np.full(len(SERIES), np.nan), where=seen > 0)
            stock, activity = total['stock']/total['count'], total['activity']/total['count']
            # an agent that never drew from a source has no scarcity to report.
            scarcity = np.divide(total['scarcity'], total['chosen'], out=np.full(len(kinds), np.nan), where=total['chosen'] > 0)
            # a mean over no source or no agent does not exist; it is saved as missing.
            row = dict(window=name, window_start=a, window_stop=b, **{key: float(v) if np.isfinite(v) else None for key, v in zip(SERIES, means)})
            row.update(empty_share=float((stock[acc] < .1).mean()), full_share=float((stock[acc] > .9).mean()),
                       intermediate_share=float(((stock[acc] >= .1) & (stock[acc] <= .9)).mean()))
            for code, label in enumerate(CLASSES):
                member = self.classes == code
                row[f'{label}_sources'] = int(member.sum())
                row[f'{label}_stock'] = float(stock[member].mean()) if member.any() else None
                row[f'{label}_empty_share'] = float((stock[member] < .1).mean()) if member.any() else None
                row[f'{label}_renewal_ratio'] = (float((total['renewal'][member]/total['count']).sum()/world.renewal[member].sum())
                                                 if member.any() else None)
            for kind, label in [(False, 'human'), (True, 'ai')]:
                group = kinds == kind
                if group.any():
                    row[f'{label}_activity_p10'] = float(np.quantile(activity[group], .1))
                    row[f'{label}_activity_minimum'] = float(activity[group].min())
                for level in ['only_overloaded', 'mixed', 'no_overloaded', 'no_access']:
                    member = group & (exposure == level)
                    row[f'{label}_{level}_agents'] = int(member.sum())
                    row[f'{label}_{level}_activity'] = float(activity[member].mean()) if member.any() else None
                    row[f'{label}_{level}_scarcity'] = float(np.nanmean(scarcity[member])) if np.isfinite(scarcity[member]).any() else None
            rows.append(row)
            arrays[name] = dict(source_stock=stock, source_cooperation=total['cooperation']/total['count'], agent_activity=activity)
        return rows, arrays


def run_segment(world, scenario, policy, driver_seed, start, stop, *, state=None, cooperative_request=None,
                renewal_control=None, disturbances=(), windows=(), consequence_weight=0., consequence_scope='feedback'):
    """Simulate absolute steps [start, stop) and return the final state and measurements.

    A new run starts at step 0 from the world's initial state unless a state is
    supplied. A continuation must supply the state reached at `start`; random
    inputs are always indexed by the absolute step, so an unchanged continuation
    reproduces the uninterrupted history. Measurements cover only this segment.
    """
    if state is None and start != 0:
        raise ValueError('a continuation requires the state at its first step')
    if state is not None and state.step != start:
        raise ValueError('state step does not match the first simulated step')
    simulator = Simulator(world, scenario, policy, cooperative_request=cooperative_request, renewal_control=renewal_control,
                          consequence_weight=consequence_weight, consequence_scope=consequence_scope)
    if state is not None:
        simulator.restore(state.water, state.strategies)
    drivers = Drivers(driver_seed, len(world.positions))
    recorder = Recorder(simulator, windows, start, stop)
    residual = 0.
    for t in range(start, stop):
        scale = 1.
        for d in disturbances:
            if d.kind == 'renewal_scale' and d.start <= t < d.stop:
                scale *= d.value
            if d.kind == 'stock_scale' and t == d.start:
                simulator.water = simulator.water*d.value
        simulator.renewal_scale = scale
        event = simulator.step(*drivers.step(t))
        residual = max(residual, abs(event['balance_residual']))
        recorder.update(t, event)
    rows, arrays = recorder.finish()
    final = State(stop, simulator.water.copy(), simulator.strategies.copy())
    info = dict(maximum_balance_residual=residual, cooperative_request=simulator.cooperative_request.copy(),
                load=recorder.load, classes=recorder.classes, series=recorder.series)
    return final, rows, arrays, info


# configuration ---------------------------------------------------------------------------

FEEDBACK_PARTS = ('start_dependence', 'controls', 'recovery')


def load_protocol(path):
    """A configuration of the paper extends a study configuration and adds a `paper` section.

    The world, the reference scenario, the payoff family and the base seed are read
    with the study's own loaders, so both protocols describe the same model.
    """
    resolved = load_config(path)
    if 'paper' not in resolved:
        raise ValueError('the configuration has no paper section')
    settings = resolved['paper']
    physical = replace(reference_scenario(resolved, settings['reference_load']), **settings.get('reference_scenario', {}))
    config = dict(settings, world=asdict(world_spec(resolved)), scenario=asdict(physical),
                  payoffs={key: resolved['payoffs'][key] for key in ['t', 's', 'relative_bounds']},
                  base_seed=resolved['validation']['base_seed'])
    validate_protocol(config)
    return config


def experiments_of(config):
    """Experiments a configuration runs; a companion configuration may name only some of them."""
    return list(config.get('experiments', EXPERIMENTS))


def cells_of(config, experiment):
    """Named physical settings an experiment is repeated in. `reference` is the configured scenario."""
    if experiment == 'continuity':
        return ['reference']
    section = 'search' if experiment in SEARCH_EXPERIMENTS else experiment
    return list(config[section].get('cells', ['reference']))


def worlds_of(config, experiment):
    return config['search']['test_worlds'] if experiment == 'search_test' else config['worlds']


def validate_protocol(config):
    WorldSpec(**config['world'])
    base = Scenario(**config['scenario'])
    for name, spec in config.get('cells', {}).items():
        known = {'scenario', 'world_spec', 'requests', 'siting', 'sited_share'}
        siting = spec.get('siting', 'random')
        if (name == 'reference' or set(spec)-known or spec.get('requests', 'scaled') not in ['scaled', 'fixed']
                or siting not in ['random', 'sited'] or not 0 < spec.get('sited_share', 1) <= 1
                or ('sited_share' in spec and siting == 'random')):
            raise ValueError(f'invalid cell {name}')
        replace(base, **spec.get('scenario', {}))
        replace(WorldSpec(**config['world']), **spec.get('world_spec', {}))
    for name, spec in config['policies'].items():
        policy(config, name)
        if set(spec)-{'r_c', 'r_d', 'observation', 'forced', 'epsilon', 'consequence_weight', 'consequence_scope', 'diagnostic', 'scenario'}:
            raise ValueError(f'unknown design option in {name}')
        replace(base, **({'epsilon': spec['epsilon']} if 'epsilon' in spec else {}))
        replace(base, **spec.get('scenario', {}))
        if not 0 <= spec.get('consequence_weight', 0) <= 1:
            raise ValueError('consequence weight must lie in [0,1]')
    if not set(config.get('experiments', EXPERIMENTS)) <= set(EXPERIMENTS+SEARCH_EXPERIMENTS):
        raise ValueError('unknown experiment')
    if set(experiments_of(config)) & set(SEARCH_EXPERIMENTS):
        validate_search(config)
    if 'unchanged' not in config['policies']:
        raise ValueError('the unchanged design must be defined')
    if not isinstance(config['worlds'], int) or config['worlds'] < 1 or not config['split'].startswith('paper'):
        raise ValueError('a positive world count and a paper-specific random-input split are required')
    requests = config['conference_requests']
    if min(requests['human'], requests['ai']) <= 0:
        raise ValueError('positive fixed requests required')
    used = (set(config['interventions']['designs']) | set(config['feedback']['hysteresis_designs'])
            | set(config['feedback']['control_designs']) | set(config['feedback']['recovery_designs']))
    if not used <= set(config['policies']):
        raise ValueError('every named design must be defined under policies')
    for experiment in ['interventions', 'feedback']:
        for cell in cells_of(config, experiment):
            cell_spec(config, cell)
    # a restriction may name cells an extending configuration no longer uses; it must still name defined cells.
    defined = {'reference', *config.get('cells', {})}
    if not set(config['interventions'].get('from_initialization_cells', [])) <= defined:
        raise ValueError('from-initialization cells must be defined cells')
    if not set(config['feedback'].get('parts', FEEDBACK_PARTS)) <= set(FEEDBACK_PARTS):
        raise ValueError('unknown feedback part')
    if not set(config['feedback'].get('hysteresis_design_cells', [])) <= defined:
        raise ValueError('hysteresis design cells must be defined cells')
    for part, names in config['feedback'].get('part_cells', {}).items():
        if part not in FEEDBACK_PARTS or not set(names) <= defined:
            raise ValueError('part_cells must name feedback parts and defined cells')
    for name, d in config['feedback']['disturbances'].items():
        Disturbance(d['kind'], 0, d.get('duration', 1), d['value'])


def feedback_parts(config, cell):
    """Parts of the feedback experiment run in one cell; `part_cells` restricts a part to named cells."""
    f = config['feedback']
    limits = f.get('part_cells', {})
    return [part for part in f.get('parts', FEEDBACK_PARTS) if cell in limits.get(part, cells_of(config, 'feedback'))]


def hysteresis_designs(config, cell):
    f = config['feedback']
    return f['hysteresis_designs'] if cell in f.get('hysteresis_design_cells', cells_of(config, 'feedback')) else []


def policy(config, name):
    spec, p = config['policies'][name], config['payoffs']
    if 'forced' in spec:
        return Policy(t=p['t'], s=p['s'], forced=spec['forced'])
    made = Policy.from_relative(spec['r_c'], spec['r_d'], p['t'], p['s'], spec.get('observation', 'local'))
    low, high = p['relative_bounds']
    # a diagnostic design may leave the admissible box; it is never reported as an admissible design.
    if not spec.get('diagnostic') and not all(low <= r <= high for r in made.relative):
        raise ValueError(f'{name} lies outside the admissible payoff bounds')
    return made


def design_setup(config, name, scenario):
    """The policy, the scenario it runs under, and simulator options for one named design."""
    spec = config['policies'][name]
    changed = replace(scenario, epsilon=spec['epsilon']) if 'epsilon' in spec else scenario
    # a design may also change how sources are chosen or capped, which the payoff family cannot express.
    changed = replace(changed, **spec.get('scenario', {}))
    return policy(config, name), changed, dict(consequence_weight=float(spec.get('consequence_weight', 0.)),
                                               consequence_scope=spec.get('consequence_scope', 'feedback'))


def policy_record(config, name):
    made, spec = policy(config, name), config['policies'][name]
    return dict(design=name, forced=made.forced, r_c=float(made.relative[0]), r_d=float(made.relative[1]),
                q_c=float(made.q_c), q_d=float(made.q_d), cost_l2=made.cost('L2'), policy_id=made.id,
                consequence_weight=float(spec.get('consequence_weight', 0.)), reversal_override=spec.get('epsilon'),
                admissible=not spec.get('diagnostic', False) and not made.forced and 'epsilon' not in spec and 'scenario' not in spec,
                price=spec.get('scenario', {}).get('price'), floor=spec.get('scenario', {}).get('floor'),
                selection=spec.get('scenario', {}).get('selection'), regulated=spec.get('scenario', {}).get('regulated'),
                allocation_override=spec.get('scenario', {}).get('allocation'),
                scenario_changes=json.dumps(spec.get('scenario', {}), sort_keys=True))


def make_world(config, index, **changes):
    spec = replace(WorldSpec(**config['world']), **changes)
    return generate_world(spec, seed(config['base_seed'], config['split'], index, 'world'))


def trajectory_seed(config, index):
    return seed(config['base_seed'], config['split'], index, 'trajectory')


def fixed_requests(config, world):
    r = config['conference_requests']
    return np.where(world.kinds, float(r['ai']), float(r['human']))


def cell_spec(config, cell):
    cells = config.get('cells', {})
    if cell != 'reference' and cell not in cells:
        raise ValueError(f'unknown cell {cell}')
    return cells.get(cell, {})


def site_ai_agents(world, demand_ratio, human_request, sited_share=1.):
    """Place AI agents where the sources can carry them; sources, human agents and initial strategies stay as generated.

    Each moved AI agent goes, in turn, to a free cell with source access. A cell is
    acceptable when the highest expected load on the sources reachable from it stays at
    or below one, which means that lower requests fit within baseline renewal; if no
    cell achieves that, the cells with the lowest such load are acceptable. Among
    acceptable cells the agent takes the one that shares the fewest sources with human
    agents, and among those the one with the lowest load. Loads follow from the
    placement, the demand ratio and the size of a human agent's lower request, so the
    rule depends on nothing that a simulation produces. With a share below one, only that share of the
    AI agents is moved: those whose reachable sources carry the highest expected load.
    """
    if not 0 < sited_share <= 1:
        raise ValueError('the sited share of the ai agents lies in (0,1]')
    spec = world.spec
    cells = np.array([(x, y) for x in range(spec.width) for y in range(spec.height)])
    near = torus_distances(cells, world.resources, spec.width, spec.height) <= spec.access_radius
    reach = near.sum(axis=1)
    share = np.divide(1., reach, out=np.zeros(len(reach)), where=reach > 0)
    spread = near*share[:, None]/world.renewal[None, :]
    index = lambda position: int(position[0])*spec.height+int(position[1])
    occupancy = np.zeros(len(cells), dtype=int)
    load = np.zeros(len(world.renewal))
    weight = np.where(world.kinds, demand_ratio, 1.)
    for agent, position in enumerate(world.positions):
        occupancy[index(position)] += 1
        load = load+spread[index(position)]*weight[agent]
    # loads are counted in human requests per unit of renewal; in those units a source carries its load up to this limit.
    limit = 1/float(human_request)
    # the number of sources reachable from each cell that at least one human agent also reaches.
    used_by_humans = near[[index(position) for position in world.positions[~world.kinds]]].any(axis=0)
    shared = (near & used_by_humans[None, :]).sum(axis=1)
    large = np.flatnonzero(world.kinds)
    # an ai agent without source access counts as facing the highest load, so it is among those moved.
    reached = [near[index(world.positions[agent])] for agent in large]
    faced = np.array([load[mask].max() if mask.any() else np.inf for mask in reached])
    moved = np.sort(large[np.argsort(-faced, kind='stable')[:int(np.floor(sited_share*len(large)+.5))]])
    for agent in moved:
        occupancy[index(world.positions[agent])] -= 1
        load = load-spread[index(world.positions[agent])]*demand_ratio
    positions = world.positions.copy()
    for agent in moved:
        worst = np.where(near, load[None, :]+spread*demand_ratio, -np.inf).max(axis=1)
        worst[(reach == 0) | (occupancy >= spec.maximum_occupancy)] = np.inf
        if not np.isfinite(worst).any():
            raise ValueError('no cell with source access is free for siting')
        acceptable = worst <= max(limit, worst.min())
        fewest = shared[acceptable].min()
        chosen = int(np.argmin(np.where(acceptable & (shared == fewest), worst, np.inf)))
        positions[agent] = cells[chosen]
        occupancy[chosen] += 1
        load = load+spread[chosen]*demand_ratio
    return replace(world, positions=positions)


def cell_context(config, cell, index):
    """World, scenario, fixed requests (or None for scaled requests) and trajectory seed of one cell."""
    spec = cell_spec(config, cell)
    scenario = replace(Scenario(**config['scenario']), **spec.get('scenario', {}))
    world = make_world(config, index, **spec.get('world_spec', {}))
    if spec.get('siting', 'random') != 'random':
        # requests keep the size they have under random placement, so both placements face the same demand in resource units;
        # the reachable renewal differs between placements, so the same demand is another pressure on it.
        reachable = float(world.renewal[world.access.any(axis=0)].sum())
        human_request = (config['conference_requests']['human'] if spec.get('requests', 'scaled') == 'fixed'
                         else scenario.pressure*reachable/(world.spec.humans+scenario.demand_ratio*world.spec.ai))
        sited = site_ai_agents(world, scenario.demand_ratio, human_request, spec.get('sited_share', 1.))
        supply = [float(w.renewal[w.access.any(axis=0)].sum()) for w in [world, sited]]
        world, scenario = sited, replace(scenario, pressure=scenario.pressure*supply[0]/supply[1])
    requests = fixed_requests(config, world) if spec.get('requests', 'scaled') == 'fixed' else None
    return world, scenario, requests, trajectory_seed(config, index)


def describe(world, scenario, requests, initial_volume):
    acc = world.access.any(axis=0)
    supply = float(world.renewal[acc].sum())
    return dict(world_id=world.id, capacity=float(world.spec.capacity), density=world.spec.density,
                accessible_sources=int(acc.sum()), accessible_baseline_renewal=supply,
                total_ordinary_request=float(requests.sum()), relative_load=float(requests.sum()/supply),
                initial_stored_volume=float(initial_volume), allocation=scenario.allocation, feedback=scenario.feedback,
                theta=scenario.theta, reversal_probability=scenario.epsilon,
                delivery_ceiling_baseline_renewal=delivery_ceiling(world, requests*scenario.defect_ratio),
                delivery_ceiling_largest_renewal=delivery_ceiling(world, requests*scenario.defect_ratio, maximum_multiplier(scenario)),
                placement='conditioned_on_access' if world.spec.require_access else 'unconditioned',
                agents_without_access=int((~world.access.any(axis=1)).sum()),
                human_reachable_sources=float(world.access[~world.kinds].sum(axis=1).mean()),
                ai_reachable_sources=float(world.access[world.kinds].sum(axis=1).mean()))


def initial_state(world, mode, volume=None):
    if mode == 'full':
        water = world.capacity.copy()
    elif mode == 'empty':
        water = np.zeros_like(world.capacity)
    else:
        water = np.minimum(world.capacity, float(volume))
    return State(0, water, world.initial.copy())


def emit(rows, common, window_rows):
    for row in window_rows:
        rows.append(common | row)


# experiment A: baseline continuity and physical drivers ----------------------------------

def continuity_world(config, index, cell='reference'):
    """Original game only. Each condition records what it changes and what it holds fixed."""
    c, base = config['continuity'], Scenario(**config['scenario'])
    reference = Policy(t=config['payoffs']['t'], s=config['payoffs']['s'])
    steps, windows, drivers = c['steps'], c['windows'], trajectory_seed(config, index)
    rows, arrays, series = [], {}, {}

    def run(name, group, changes, world, scenario, absolute, initial='full'):
        requests = fixed_requests(config, world) if absolute else None
        start = initial_state(world, initial, c['equal_initial_volume'])
        final, window_rows, window_arrays, info = run_segment(world, scenario, reference, drivers, 0, steps, state=start,
                                                             cooperative_request=requests, windows=windows)
        common = dict(experiment='continuity', cell=cell, world_index=index, condition=name, group=group, changes=changes,
                      requests='fixed_absolute' if absolute else 'scaled_to_accessible_renewal', initial_stock=initial,
                      maximum_balance_residual=info['maximum_balance_residual'],
                      **describe(world, scenario, info['cooperative_request'], start.water.sum()))
        emit(rows, common, window_rows)
        series[name] = info['series'].astype(np.float32)
        arrays[name] = dict(load=info['load'], classes=info['classes'], late_stock=window_arrays[windows[-1][0]]['source_stock'])

    conference = replace(base, allocation='human_priority', feedback='repository')
    loose = make_world(config, index, require_access=False)
    world = make_world(config, index)
    ladder = [('conference_like', 'fixed requests, unconditioned placement, human-first allocation, repository renewal law', loose, conference, True),
              ('conference_like_published_law', 'as conference_like with the published Algorithm 2 renewal law', loose, replace(conference, feedback='manuscript'), True),
              ('conditioned_placement', 'as conference_like; every agent placed with source access', world, conference, True),
              ('proportional_allocation', 'as conditioned_placement; shortages shared in proportion to requests', world, replace(conference, allocation='proportional'), True),
              ('journal_reference', 'as proportional_allocation; ordinary requests rescaled to the configured load', world, base, False)]
    for name, changes, w, scenario, absolute in ladder:
        run(name, 'ladder', changes, w, scenario, absolute)
    for absolute, label, scenario in [(True, 'fixed_requests', replace(conference, allocation='proportional')), (False, 'journal', base)]:
        for capacity in c['capacities']:
            w = make_world(config, index, capacity=float(capacity))
            for initial in ['full', 'equal_volume', 'empty']:
                run(f'storage_{label}_K{capacity}_{initial}', f'storage_{label}',
                    f'capacity {capacity}; initial stock {initial}; same landscape, renewal and requests', w, scenario, absolute, initial)
        for density in c['densities']:
            w = make_world(config, index, density=float(density))
            note = ('requests fixed, so relative load changes with supply' if absolute
                    else 'relative load fixed, so requests change with supply')
            run(f'supply_{label}_density{density}', f'supply_{label}', f'source density {density}; new landscape; {note}', w, scenario, absolute)
    return rows, arrays, series


# experiment B: payoff changes after development and from initialization -----------------

def interventions_world(config, index, cell='reference'):
    b = config['interventions']
    world, scenario, requests, drivers = cell_context(config, cell, index)
    t0, after = b['develop_steps'], b['after_steps']
    relative = [(f'{a}-{z}', a, z) for a, z in b['windows_after']]
    later = [(name, t0+a, t0+z) for name, a, z in relative]
    # saved final states come from the window that starts last, whatever the order of the configured windows.
    last = max(later, key=lambda window: (window[1], window[2]))[0]
    from_start = cell in b.get('from_initialization_cells', cells_of(config, 'interventions'))
    rows, arrays, series = [], {}, {}

    def segment(name, start, stop, state=None, windows=()):
        made, physical, options = design_setup(config, name, scenario)
        return run_segment(world, physical, made, drivers, start, stop, state=state, cooperative_request=requests,
                           windows=windows, **options)

    state0, developed_rows, developed_arrays, info0 = segment('unchanged', 0, t0, windows=[('developed', t0-b['developed_window'], t0)])
    base = dict(experiment='interventions', cell=cell, world_index=index, requests='fixed_absolute' if requests is not None else 'scaled_to_accessible_renewal',
                **describe(world, scenario, info0['cooperative_request'], world.capacity.sum()))
    emit(rows, base | dict(condition='unchanged', timing='development', **policy_record(config, 'unchanged')), developed_rows)
    # an unchanged restart must reproduce the uninterrupted history exactly.
    whole, _, _, whole_info = segment('unchanged', 0, t0+after)
    final, window_rows, window_arrays, info = segment('unchanged', t0, t0+after, state=state0, windows=later)
    if (not np.array_equal(whole.water, final.water) or not np.array_equal(whole.strategies, final.strategies)
            or not np.array_equal(whole_info['series'][t0:], info['series'], equal_nan=True)):
        raise ArithmeticError('unchanged continuation differs from the uninterrupted simulation')
    emit(rows, base | dict(condition='unchanged', timing='after_development', continuation_check='identical_to_uninterrupted',
                           maximum_balance_residual=max(info0['maximum_balance_residual'], info['maximum_balance_residual']),
                           **policy_record(config, 'unchanged')), window_rows)
    if from_start:
        _, early_rows, _, early_info = segment('unchanged', 0, t0+after, windows=relative)
        emit(rows, base | dict(condition='unchanged', timing='from_initialization', maximum_balance_residual=early_info['maximum_balance_residual'],
                               **policy_record(config, 'unchanged')), early_rows)
    series['unchanged'] = whole_info['series'].astype(np.float32)
    arrays['unchanged'] = dict(load=info['load'], classes=info['classes'], developed_stock=developed_arrays['developed']['source_stock'],
                               final_stock=window_arrays[last]['source_stock'],
                               final_activity=window_arrays[last]['agent_activity'],
                               final_cooperation=window_arrays[last]['source_cooperation'])
    for name in b['designs']:
        _, window_rows, window_arrays, info = segment(name, t0, t0+after, state=state0, windows=later)
        emit(rows, base | dict(condition=name, timing='after_development', maximum_balance_residual=info['maximum_balance_residual'],
                               **policy_record(config, name)), window_rows)
        series[f'{name}|after_development'] = info['series'].astype(np.float32)
        arrays[name] = dict(final_stock=window_arrays[last]['source_stock'], final_activity=window_arrays[last]['agent_activity'],
                            final_cooperation=window_arrays[last]['source_cooperation'])
        if from_start:
            same_time = [(f'same_time:{window}', a, z) for window, a, z in later]
            _, window_rows, _, info = segment(name, 0, t0+after, windows=relative+same_time)
            emit(rows, base | dict(condition=name, timing='from_initialization', maximum_balance_residual=info['maximum_balance_residual'],
                                   **policy_record(config, name)), window_rows)
            series[f'{name}|from_initialization'] = info['series'].astype(np.float32)
    return rows, arrays, series


# experiment C: what behavior-dependent renewal contributes ------------------------------

def feedback_world(config, index, cell='reference'):
    f = config['feedback']
    parts = feedback_parts(config, cell)
    world, scenario, requests, drivers = cell_context(config, cell, index)
    rows, arrays, series = [], {}, {}
    base = dict(experiment='feedback', cell=cell, world_index=index,
                requests='fixed_absolute' if requests is not None else 'scaled_to_accessible_renewal')

    def segment(name, start, stop, **options):
        made, physical, extra = design_setup(config, name, scenario)
        return run_segment(world, physical, made, drivers, start, stop, cooperative_request=requests, **extra, **options)

    if 'start_dependence' in parts:
        # do long-run source states depend on where the stocks started?
        steps, windows = f['hysteresis_steps'], f['hysteresis_windows']

        def hysteresis(name, initial, control, label):
            start = initial_state(world, initial)
            _, window_rows, window_arrays, info = segment(name, 0, steps, state=start, renewal_control=control, windows=windows)
            common = base | dict(part='hysteresis', condition=f'{name}|{label}|{initial}', renewal_rule=label, initial_stock=initial,
                                 maximum_balance_residual=info['maximum_balance_residual'],
                                 **policy_record(config, name), **describe(world, scenario, info['cooperative_request'], start.water.sum()))
            emit(rows, common, window_rows)
            key = f'hysteresis|{name}|{label}|{initial}'
            arrays[key] = {w: window_arrays[w]['source_stock'] for w, _, _ in windows} | dict(
                load=info['load'], classes=info['classes'], agent_activity=window_arrays[windows[-1][0]]['agent_activity'],
                source_cooperation=window_arrays[windows[-1][0]]['source_cooperation'])
            series[key] = info['series'].astype(np.float32)
            return window_arrays

        coupled = hysteresis('unchanged', 'full', None, 'coupled')
        hysteresis('unchanged', 'empty', None, 'coupled')
        # the control reads each source's usual cooperation from the coupled full-start run.
        usual = tuple(float(x) for x in coupled[windows[-1][0]]['source_cooperation'])
        for initial in ['full', 'empty']:
            hysteresis('unchanged', initial, RenewalControl('fixed_cooperation', usual), 'fixed_cooperation')
            hysteresis('unchanged', initial, RenewalControl('baseline'), 'baseline_only')
            for name in hysteresis_designs(config, cell):
                hysteresis(name, initial, None, 'coupled')
    if 'controls' not in parts and 'recovery' not in parts:
        return rows, arrays, series
    # the remaining parts start from one developed unchanged state.
    t0, after = f['develop_steps'], f['after_steps']
    relative = [(f'{a}-{z}', t0+a, t0+z) for a, z in f['windows_after']]
    state0, _, developed, info0 = segment('unchanged', 0, t0, windows=[('developed', t0-f['developed_window'], t0)])
    frozen = RenewalControl('fixed_cooperation', tuple(float(x) for x in developed['developed']['source_cooperation']))
    described = describe(world, scenario, info0['cooperative_request'], world.capacity.sum())

    def continue_from(part, name, label, control=None, disturbance=None):
        events = ()
        if disturbance:
            d = f['disturbances'][disturbance]
            events = (Disturbance(d['kind'], t0, t0+d.get('duration', 1), d['value']),)
        _, window_rows, window_arrays, info = segment(name, t0, t0+after, state=state0, renewal_control=control,
                                                      disturbances=events, windows=relative)
        emit(rows, base | dict(part=part, condition=f'{name}|{label}', renewal_rule=label if part == 'control' else 'coupled',
                               disturbance=disturbance or 'none', maximum_balance_residual=info['maximum_balance_residual'],
                               **policy_record(config, name), **described), window_rows)
        key = f'{part}|{name}|{label}'
        arrays[key] = dict(final_stock=window_arrays[relative[-1][0]]['source_stock'], classes=info['classes'], load=info['load'],
                           developed_stock=developed['developed']['source_stock'])
        series[key] = info['series'].astype(np.float32)

    if 'controls' in parts:
        for name in ['unchanged', *f['control_designs']]:
            continue_from('control', name, 'coupled')
            continue_from('control', name, 'fixed_cooperation', frozen)
            continue_from('control', name, 'baseline_only', RenewalControl('baseline'))
    if 'recovery' in parts:
        for name in ['unchanged', *f['recovery_designs']]:
            continue_from('recovery', name, 'none')
            for disturbance in f['disturbances']:
                continue_from('recovery', name, disturbance, disturbance=disturbance)
    return rows, arrays, series


# experiment D: a map of the payoff family and a test of the designs it selects --------------

SEARCH_BOUNDS = ('always_ordinary', 'always_intensified')


def validate_search(config):
    s = config['search']
    for cell in s['cells']:
        cell_spec(config, cell)
    windows = {f'{a}-{z}' for a, z in s['windows_after']}
    if any(item['window'] not in windows for item in [*s['objectives'], s['requirement']]):
        raise ValueError('search objectives and the requirement must use declared windows')
    if (len(s['objectives']) < 2 or s['points_per_axis'] < 2 or s['selected_per_cell'] < 1 or s['test_worlds'] < 2
            or not s['test_split'].startswith('paper') or s['test_split'] == config['split']):
        raise ValueError('invalid search settings')


def payoff_grid(config):
    """Relative changes on the grid of the staged study: equally spaced points per axis, with zero added."""
    low, high = config['payoffs']['relative_bounds']
    axis = np.unique(np.r_[np.linspace(low, high, config['search']['points_per_axis']), 0.])
    return [(float(r_c), float(r_d)) for r_c in axis for r_d in axis]


def grid_name(r_c, r_d):
    return f'r_c={r_c:+.4f},r_d={r_d:+.4f}'


def grid_record(config, r_c, r_d):
    p = config['payoffs']
    made = Policy.from_relative(r_c, r_d, p['t'], p['s'])
    return made, dict(design=grid_name(r_c, r_d), forced=None, r_c=r_c, r_d=r_d, q_c=float(made.q_c), q_d=float(made.q_d),
                      cost_l2=made.cost('L2'), policy_id=made.id, consequence_weight=0., reversal_override=None, admissible=True)


def bound_record(config, name):
    p = config['payoffs']
    made = Policy(t=p['t'], s=p['s'], forced='C' if name == 'always_ordinary' else 'D')
    return made, dict(design=name, forced=made.forced, r_c=0., r_d=0., q_c=float(made.q_c), q_d=float(made.q_d), cost_l2=0.,
                      policy_id=made.id, consequence_weight=0., reversal_override=None, admissible=False)


def _search_runs(config, index, cell, experiment, designs):
    """Continue one developed state of the original game under each design, with matched inputs."""
    s = config['search']
    world, scenario, requests, drivers = cell_context(config, cell, index)
    t0, after = s['develop_steps'], s['after_steps']
    windows = [(f'{a}-{z}', t0+a, t0+z) for a, z in s['windows_after']]
    reference, _ = grid_record(config, 0., 0.)
    state0, _, _, info0 = run_segment(world, scenario, reference, drivers, 0, t0, cooperative_request=requests)
    base = dict(experiment=experiment, cell=cell, world_index=index, timing='after_development',
                requests='fixed_absolute' if requests is not None else 'scaled_to_accessible_renewal',
                **describe(world, scenario, info0['cooperative_request'], world.capacity.sum()))
    rows = []
    for made, record in designs:
        _, window_rows, _, info = run_segment(world, scenario, made, drivers, t0, t0+after, state=state0,
                                              cooperative_request=requests, windows=windows)
        emit(rows, base | dict(condition=record['design'], maximum_balance_residual=info['maximum_balance_residual'], **record), window_rows)
    return rows, {}, {}


def search_world(config, index, cell='reference'):
    """Every design of the payoff grid, and the two forced rules that bound what any design can do."""
    designs = [grid_record(config, r_c, r_d) for r_c, r_d in payoff_grid(config)]
    return _search_runs(config, index, cell, 'search', designs+[bound_record(config, name) for name in SEARCH_BOUNDS])


def search_test_world(config, index, cell='reference'):
    """The original game and the designs frozen from the search, on worlds the search never used."""
    chosen = [(0., 0.), *[(d['r_c'], d['r_d']) for d in config['selected_designs'][cell]]]
    test = dict(config, split=config['search']['test_split'])
    return _search_runs(test, index, cell, 'search_test', [grid_record(config, r_c, r_d) for r_c, r_d in chosen])


def search_table(rows, config, cell):
    """Mean outcomes of every grid design over the search worlds of one cell, and what the selection reads.

    All objectives are maximized. A design meets the requirement when the mean of the
    requirement outcome does not exceed the original game's mean by more than the allowance.
    """
    s = config['search']
    wanted = [(item['outcome'], item['window']) for item in [*s['objectives'], s['requirement']]]
    sums, counts, records = {}, {}, {}
    for row in rows:
        if row['experiment'] != 'search' or row['cell'] != cell or not row['admissible']:
            continue
        for outcome, window in wanted:
            if row['window'] == window:
                key = (row['condition'], outcome, window)
                sums[key] = sums.get(key, 0.)+row[outcome]
                counts[key] = counts.get(key, 0)+1
        records.setdefault(row['condition'], {key: row[key] for key in ['r_c', 'r_d', 'q_c', 'q_d', 'cost_l2']})
    means = {key: total/counts[key] for key, total in sums.items()}
    original = grid_name(0., 0.)
    limit_outcome, limit_window = s['requirement']['outcome'], s['requirement']['window']
    limit = means[(original, limit_outcome, limit_window)]+s['requirement']['allowance']
    table = []
    for name, record in sorted(records.items(), key=lambda item: (item[1]['cost_l2'], item[0])):
        values = [means[(name, item['outcome'], item['window'])] for item in s['objectives']]
        table.append(dict(record, design=name, objectives=values, requirement_value=means[(name, limit_outcome, limit_window)],
                          meets_requirement=bool(means[(name, limit_outcome, limit_window)] <= limit+1e-12)))
    return table


def nondominated(table, budget):
    """Designs within the budget that meet the requirement and are not beaten on every objective by another."""
    feasible = [row for row in table if row['meets_requirement'] and row['cost_l2'] <= budget+1e-12]
    values = np.array([row['objectives'] for row in feasible])
    return [row for i, row in enumerate(feasible)
            if not np.any(np.all(values >= values[i], axis=1) & np.any(values > values[i], axis=1))]


def select_designs(table, budget, limit):
    """At most `limit` non-dominated designs other than the original: each objective's best, then the widest spread."""
    front = [row for row in nondominated(table, budget) if row['design'] != grid_name(0., 0.)]
    if len(front) <= limit:
        return front
    values = np.array([row['objectives'] for row in front])
    spread = np.ptp(values, axis=0)
    scaled = (values-values.min(axis=0))/np.where(spread > 0, spread, 1.)
    chosen = list(dict.fromkeys(int(np.argmax(scaled[:, k])) for k in range(scaled.shape[1])))[:limit]
    while len(chosen) < limit:
        distance = ((scaled[:, None, :]-scaled[chosen][None, :, :])**2).sum(axis=2).min(axis=1)
        distance[chosen] = -1
        chosen.append(int(np.argmax(distance)))
    return [front[i] for i in chosen]


def frozen_selection(config, output, rows, done):
    """Designs chosen from the completed search; written once and never changed by later results."""
    path = Path(output)/'selected_designs.json'
    if path.exists():
        return json.loads(path.read_text())
    s = config['search']
    missing = [(cell, index) for cell in s['cells'] for index in range(config['worlds']) if ('search', cell, index) not in done]
    if missing:
        raise ValueError('the test of selected designs requires the completed search')
    selected = {cell: [{key: row[key] for key in ['design', 'r_c', 'r_d', 'cost_l2']}
                       for row in select_designs(search_table(rows, config, cell), s['selection_budget'], s['selected_per_cell'])]
                for cell in s['cells']}
    write_json(path, selected)
    return selected


# execution -------------------------------------------------------------------------------

WORLD_FUNCTIONS = dict(continuity=continuity_world, interventions=interventions_world, feedback=feedback_world,
                       search=search_world, search_test=search_test_world)


def planned_steps(config, experiments):
    """Exact model steps for the requested experiments; nothing is simulated."""
    c, b, f = config['continuity'], config['interventions'], config['feedback']
    t0, after, designs = b['develop_steps'], b['after_steps'], len(b['designs'])
    from_start = set(b.get('from_initialization_cells', cells_of(config, 'interventions')))
    interventions = sum(t0+(t0+after)+after+designs*after+((1+designs)*(t0+after) if cell in from_start else 0)
                        for cell in cells_of(config, 'interventions'))
    feedback = 0
    for cell in cells_of(config, 'feedback'):
        parts = feedback_parts(config, cell)
        if 'start_dependence' in parts:
            feedback += f['hysteresis_steps']*(2+2*(2+len(hysteresis_designs(config, cell))))
        if 'controls' in parts or 'recovery' in parts:
            feedback += f['develop_steps']
        if 'controls' in parts:
            feedback += f['after_steps']*3*(1+len(f['control_designs']))
        if 'recovery' in parts:
            feedback += f['after_steps']*(1+len(f['recovery_designs']))*(1+len(f['disturbances']))
    per_world = dict(continuity=c['steps']*(5+2*(3*len(c['capacities'])+len(c['densities']))),
                     interventions=interventions, feedback=feedback)
    if set(experiments) & set(SEARCH_EXPERIMENTS):
        s = config['search']
        per_cell = s['develop_steps']+s['after_steps']
        per_world['search'] = len(s['cells'])*(s['develop_steps']+s['after_steps']*(len(payoff_grid(config))+len(SEARCH_BOUNDS)))
        # the test of the selected designs is planned at its largest possible size.
        per_world['search_test'] = len(s['cells'])*(per_cell+s['after_steps']*s['selected_per_cell'])
    return {name: per_world[name]*worlds_of(config, name) for name in experiments}


def _worker_start():
    # the parent saves results and handles interruption; workers only compute.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _work(job):
    config, name, cell, index = job
    rows, arrays, series = WORLD_FUNCTIONS[name](config, index, cell)
    return name, cell, index, rows, arrays, series


def run_protocol(config, output, experiments, maximum_steps, workers=1):
    validate_protocol(config)
    output = Path(output)
    experiments = experiments_of(config) if experiments == 'all' else [experiments]
    plan = planned_steps(config, experiments)
    if sum(plan.values()) > maximum_steps:
        raise ValueError(f'planned {sum(plan.values()):,} model steps exceed the limit {maximum_steps:,}; no simulations started')
    config_id, sources = identifier(config), source_hashes()
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['configuration_id'] != config_id or manifest['source_sha256'] != sources:
            raise ValueError('configuration or source changed; start a new output folder')
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('output folder must be empty or contain a matching run')
        manifest = dict(protocol=config['protocol'], role=config['role'], run_id=identifier([config_id, sources])[:24],
                        created_utc=utc_now(), configuration=config, configuration_id=config_id, source_sha256=sources,
                        environment=environment(), completed=[])
        write_json(manifest_path, manifest)
    done = {tuple(item) for item in manifest['completed']}
    # a run stopped while saving one job may have left that job's rows behind; keep only rows of completed jobs.
    saved = read_jsonl(output/'rows.jsonl')
    kept = [row for row in saved if (row['experiment'], row['cell'], row['world_index']) in done]
    if len(kept) != len(saved):
        (output/'rows.jsonl').write_text(''.join(canonical(row)+'\n' for row in kept), encoding='utf-8')
    jobs = []
    for name in experiments:
        # the designs to test are frozen from the completed search before any test world is simulated.
        job_config = dict(config, selected_designs=frozen_selection(config, output, kept, done)) if name == 'search_test' else config
        jobs += [(job_config, name, cell, index) for cell in cells_of(config, name)
                 for index in range(worlds_of(config, name)) if (name, cell, index) not in done]
    pool = None
    if workers > 1 and len(jobs) > 1:
        # one numerical thread per worker: the jobs are the parallel units.
        for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
            os.environ.setdefault(variable, '1')
        pool = get_context('spawn').Pool(min(workers, len(jobs)), initializer=_worker_start)
    # results arrive in job order, so saved files do not depend on the worker count.
    results = pool.imap(_work, jobs, chunksize=1) if pool else map(_work, jobs)
    try:
        for name, cell, index, rows, arrays, series in results:
            for row in rows:
                append_json(output/'rows.jsonl', row)
            (output/'arrays').mkdir(exist_ok=True)
            np.savez_compressed(output/'arrays'/f'{name}-{cell}-{index:04d}.npz',
                                **{f'{condition}::{key}': value for condition, values in arrays.items() for key, value in values.items()})
            np.savez_compressed(output/'arrays'/f'{name}-{cell}-{index:04d}-series.npz', columns=np.array(SERIES), **series)
            manifest['completed'].append([name, cell, index])
            manifest['updated_utc'] = utc_now()
            write_json(manifest_path, manifest)
            saved, total = len(manifest['completed']), len(done)+len(jobs)
            print(f'{name} [{cell}]: world {index+1} of {worlds_of(config, name)} saved ({saved} of {total} jobs)', flush=True)
    except BaseException:
        # stop the workers at once; saved jobs stay saved and the same command resumes.
        if pool:
            pool.terminate()
        raise
    finally:
        if pool:
            pool.close()
            pool.join()
    return dict(run_id=manifest['run_id'], completed=len(manifest['completed']), planned_model_steps=plan, rows=len(read_jsonl(output/'rows.jsonl')))


def run_all(configs, results, workers=None):
    """Run each (name, configuration) into results/<name> with its planned size, then write its report.

    Completed worlds are reused, so repeating the command after an interruption continues it.
    """
    workers = workers or max(1, min(16, (os.cpu_count() or 1)-2))
    summary = []
    for name, config in configs:
        output = Path(results)/name
        needed = experiments_of(config)
        limit = sum(planned_steps(config, needed).values())
        label = config['protocol']
        print(f'{label}: {limit:,} model steps on {workers} workers, saving to {output}', flush=True)
        made = {}
        for experiment in needed:
            made = run_protocol(config, output, experiment, limit, workers)
        from .paper_tables import report
        try:
            tables = report(output)['tables']
        except Exception as error:
            # the simulations are saved; a report can be rebuilt without running anything again.
            tables = f'report not written ({error}); the saved simulations are complete'
        summary.append(dict(protocol=config['protocol'], output=str(output), jobs=made.get('completed'), tables=tables))
    return summary
