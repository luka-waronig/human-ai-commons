"""Frozen-observation spatial dynamics with independent indexed random inputs."""
from dataclasses import dataclass
import numpy as np
from .metrics import Depletion, LocalStress, MetricSpec, Service, request_fulfillment
from .provenance import rng
from .world import torus_distances


@dataclass(frozen=True)
class Scenario:
    pressure: float = 1
    demand_ratio: float = 20
    defect_ratio: float = 1.5
    theta: float = 3
    epsilon: float = .1
    feedback: str = 'repository'
    schedule: str = 'simultaneous'
    allocation: str = 'human_priority'

    def __post_init__(self):
        values = [self.pressure, self.demand_ratio, self.defect_ratio, self.theta, self.epsilon]
        if (not all(np.isfinite(v) for v in values) or min(self.pressure, self.demand_ratio) <= 0
                or self.defect_ratio < 1 or self.theta < 0 or not 0 <= self.epsilon < .5):
            raise ValueError('invalid physical scenario')
        if self.feedback not in ['repository', 'manuscript', 'pre_floor_change']:
            raise ValueError('unknown feedback law')
        if self.schedule not in ['simultaneous', 'sequential_planning']:
            raise ValueError('unknown planning schedule')
        if self.allocation not in ['human_priority', 'proportional']:
            raise ValueError('unknown withdrawal allocation')


def multiplier(n, cooperation, theta, law):
    if law == 'manuscript':
        expression = theta*cooperation-(1-cooperation)
    elif law in ['repository', 'pre_floor_change']:
        expression = theta*cooperation-1
    else:
        raise ValueError('unknown feedback law')
    return np.clip(1+.5*n*(1-n)*expression, 1 if law == 'repository' else .1, 3)


class Drivers:
    """Each (stream, step) has its own PCG64 generator; missing targets skip no draws."""
    def __init__(self, driver_seed, population):
        self.seed, self.population = driver_seed, population

    def step(self, t):
        return (rng(self.seed, 'target-rank', t).random(self.population),
                rng(self.seed, 'action-reversal', t).random(self.population),
                rng(self.seed, 'planning-permutation', t).permutation(self.population))

    def archive(self, path, steps):
        rows = [self.step(t) for t in range(steps)]
        np.savez_compressed(path, target=np.array([r[0] for r in rows]),
                            reversal=np.array([r[1] for r in rows]), order=np.array([r[2] for r in rows]))


class ArchivedDrivers:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as arrays:
            self.arrays = {name: arrays[name] for name in ['target', 'reversal', 'order']}

    def step(self, t):
        return tuple(self.arrays[name][t] for name in ['target', 'reversal', 'order'])


class Simulator:
    def __init__(self, world, scenario, policy):
        self.world, self.scenario, self.policy = world, scenario, policy
        self.water = world.capacity.copy()
        self.strategies = world.initial.copy()
        if policy.forced:
            self.strategies[:] = policy.forced == 'C'
        self.access = world.access
        self.accessible = self.access.any(axis=0)
        supply = world.renewal[self.accessible].sum()
        if supply <= 0:
            raise ValueError('accessible baseline supply must be positive')
        human_unit = scenario.pressure*supply/(world.spec.humans+scenario.demand_ratio*world.spec.ai)
        self.cooperative_request = np.where(world.kinds, scenario.demand_ratio*human_unit, human_unit)
        distances = torus_distances(world.resources, world.positions, world.spec.width, world.spec.height)
        self.ecological = (distances <= world.spec.feedback_radius).astype(float)
        self.interaction = distances <= world.spec.game_radius
        self.ecological_count = self.ecological.sum(axis=1)
        self.interaction_count = self.interaction.sum(axis=1)
        self.withdrawal_order = np.argsort(world.kinds, kind='stable')

    def step(self, target_rank, reversal, order):
        n_agents = len(self.strategies)
        for values in [target_rank, reversal]:
            if (np.shape(values) != (n_agents,) or not np.all(np.isfinite(values))
                    or np.any(np.asarray(values) < 0) or np.any(np.asarray(values) >= 1)):
                raise ValueError('invalid random driver')
        if not np.array_equal(np.sort(order), np.arange(n_agents)):
            raise ValueError('planning order must be a permutation')
        before = self.water.sum()
        old = self.strategies.copy()
        cooperative = self.ecological @ old.astype(float)
        local = np.divide(cooperative, self.ecological_count, out=np.zeros_like(cooperative), where=self.ecological_count > 0)
        factors = multiplier(self.water/self.world.capacity, local, self.scenario.theta, self.scenario.feedback)
        if self.scenario.feedback != 'manuscript':
            factors[self.ecological_count == 0] = 1
        replenished = np.minimum(self.world.capacity, self.water+self.world.renewal*factors)
        resource_renewal = replenished-self.water
        actual_renewal = replenished.sum()-before
        planned = old.copy()
        target = np.full(n_agents, -1, dtype=int)
        planning_order = np.arange(n_agents) if self.scenario.schedule == 'simultaneous' else order
        for i in planning_order:
            available = np.flatnonzero(self.access[i] & (replenished > 0))
            if not len(available):
                continue
            j = available[min(int(target_rank[i]*len(available)), len(available)-1)]
            target[i] = j
            if self.policy.forced:
                planned[i] = self.policy.forced == 'C'
                continue
            observed = old if self.scenario.schedule == 'simultaneous' else planned
            if self.policy.observation == 'fixed':
                x = .5
            elif self.policy.observation == 'global':
                x = (observed.sum()-observed[i])/(n_agents-1) if n_agents > 1 else .5
            else:
                neighbors = self.interaction[j].copy()
                neighbors[i] = False
                x = float(observed[neighbors].mean()) if neighbors.any() else .5
            best = self.policy.gap(replenished[j]/self.world.capacity[j], x) <= 0
            planned[i] = bool(best) != bool(reversal[i] < self.scenario.epsilon)
        requested = self.cooperative_request*np.where(planned, 1, self.scenario.defect_ratio)
        delivered = np.zeros(n_agents)
        water = replenished.copy()
        if self.scenario.allocation == 'proportional':
            selected = target >= 0
            total_request = np.bincount(target[selected], weights=requested[selected], minlength=len(water))
            fraction = np.minimum(1, np.divide(water, total_request, out=np.ones_like(water), where=total_request > 0))
            delivered[selected] = requested[selected]*fraction[target[selected]]
            water = np.maximum(0, water-np.bincount(target[selected], weights=delivered[selected], minlength=len(water)))
        else:
            for i in self.withdrawal_order:
                if target[i] >= 0:
                    j = target[i]
                    delivered[i] = min(requested[i], water[j])
                    water[j] -= delivered[i]
        residual = float(water.sum()-(before+actual_renewal-delivered.sum()))
        if abs(residual) > 1e-9*max(1, before):
            raise ArithmeticError('water balance failed')
        self.strategies, self.water = planned, water
        return dict(requested=requested, delivered=delivered, actual_renewal=float(actual_renewal),
                    balance_residual=residual, targets=target, resource_renewal=resource_renewal,
                    source_withdrawal=np.bincount(target[target >= 0], weights=delivered[target >= 0], minlength=len(water)))


def simulate(world, scenario, policy, driver_seed, steps, burn, metric=MetricSpec(), *, record=False, drivers=None,
             welfare=None, stress=None):
    if isinstance(steps, bool) or not isinstance(steps, int) or not isinstance(burn, int) or not 0 <= burn < steps:
        raise ValueError('require integer steps > burn >= 0')
    simulator = Simulator(world, scenario, policy)
    drivers = drivers or Drivers(driver_seed, len(world.positions))
    accumulator = Depletion(int(simulator.accessible.sum()), metric)
    service = Service(simulator.cooperative_request*scenario.defect_ratio, world.kinds, welfare,
                      baseline=simulator.cooperative_request) if welfare else None
    local_stress = LocalStress(int(simulator.accessible.sum()) if stress.basis == 'resource_stock' else len(world.positions),
                              steps-burn, stress) if stress else None
    observations, stocks, strategies = [], [], []
    agent_requests, agent_deliveries = [], []
    targets, renewals, withdrawals = [], [], []
    requested_totals = np.zeros(len(world.positions))
    delivered_totals = np.zeros(len(world.positions))
    unserved_steps = np.zeros(len(world.positions), dtype=int)
    totals = np.zeros(10)
    largest_residual = 0
    for t in range(steps):
        event = simulator.step(*drivers.step(t))
        normalized = simulator.water/world.capacity
        row = [float(normalized[simulator.accessible].mean()), float(normalized.mean()),
               float(simulator.strategies.mean()), float(event['requested'].sum()),
               float(event['delivered'].sum()), event['actual_renewal']]
        for kind in [False, True]:
            group = world.kinds == kind
            row.extend([float(event['requested'][group].sum()), float(event['delivered'][group].sum())])
        if t >= burn:
            accumulator.update(normalized[simulator.accessible])
            if service:
                service.update(event['delivered'])
            if local_stress:
                local_stress.update(normalized[simulator.accessible] if stress.basis == 'resource_stock'
                                    else request_fulfillment(event['delivered'], event['requested']))
            totals += row
            requested_totals += event['requested']
            delivered_totals += event['delivered']
            unserved_steps += event['delivered'] == 0
        if record:
            stocks.append(simulator.water.copy())
            strategies.append(simulator.strategies.copy())
            observations.append(row)
            agent_requests.append(event['requested'].copy())
            agent_deliveries.append(event['delivered'].copy())
            targets.append(event['targets'].copy())
            renewals.append(event['resource_renewal'].copy())
            withdrawals.append(event['source_withdrawal'].copy())
        largest_residual = max(largest_residual, abs(event['balance_residual']))
    names = ['accessible_resource', 'all_resource', 'cooperation', 'requested', 'delivered',
             'actual_renewal', 'human_requested', 'human_delivered', 'ai_requested', 'ai_delivered']
    summary = accumulator.finish() | {f'mean_{key}': float(value/(steps-burn)) for key, value in zip(names, totals)}
    summary.update(mean_unmet=summary['mean_requested']-summary['mean_delivered'],
                   maximum_balance_residual=largest_residual, policy_id=policy.id, world_id=world.id)
    summary.update(agent_reference_request=simulator.cooperative_request.tolist(),
                   agent_mean_requested=(requested_totals/(steps-burn)).tolist(),
                   agent_mean_delivered=(delivered_totals/(steps-burn)).tolist(),
                   agent_unserved_fraction=(unserved_steps/(steps-burn)).tolist())
    if service:
        summary.update(service.finish())
    if local_stress:
        summary.update(local_stress.finish())
    data = dict(stocks=np.asarray(stocks), strategies=np.asarray(strategies), observations=np.asarray(observations),
                observation_columns=np.array(names), accessible=simulator.accessible,
                agent_requested=np.asarray(agent_requests), agent_delivered=np.asarray(agent_deliveries),
                agent_targets=np.asarray(targets), resource_renewal=np.asarray(renewals), source_withdrawal=np.asarray(withdrawals),
                agent_reference_request=simulator.cooperative_request.copy(), agent_is_ai=world.kinds.copy()) if record else None
    return summary, data
