"""Conservative finite-window strict-safety obstruction, not optimizer infeasibility."""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from .dynamics import Simulator
from .payoffs import Policy


def routing_bound(access, demands, capacities):
    access = np.asarray(access, dtype=bool)
    demands, capacities = np.asarray(demands, dtype=float), np.asarray(capacities, dtype=float)
    if (access.shape != (len(demands), len(capacities)) or not np.all(np.isfinite(demands))
            or not np.all(np.isfinite(capacities)) or np.any(demands < 0) or np.any(capacities < 0)):
        raise ValueError('invalid routing relaxation')
    agents, resources = np.where(access)
    n, m, edges = *access.shape, len(agents)
    if edges:
        indices = np.arange(edges)
        incidence = coo_matrix((np.ones(2*edges), (np.r_[agents, n+resources], np.r_[indices, indices])), shape=(n+m, edges)).tocsr()
        solution = linprog(-np.ones(edges), A_ub=incidence, b_ub=np.r_[demands, capacities], bounds=(0, None), method='highs')
        if not solution.success:
            raise ArithmeticError(solution.message)
        flow = np.zeros(access.shape)
        flow[agents, resources] = solution.x
    else:
        flow = np.zeros(access.shape)
    tolerance = 1e-8*max(1, demands.sum(), capacities.sum())
    reachable_agents = demands-flow.sum(axis=1) > tolerance
    reachable_resources = np.zeros(m, dtype=bool)
    for _ in range(n+m+1):
        new_resources = reachable_resources | access[reachable_agents].any(axis=0)
        new_agents = reachable_agents | (flow[:, new_resources] > tolerance).any(axis=1)
        if np.array_equal(new_resources, reachable_resources) and np.array_equal(new_agents, reachable_agents):
            break
        reachable_resources, reachable_agents = new_resources, new_agents
    cut_demand = float(demands[reachable_agents].sum())
    cut_capacity = float(capacities[reachable_resources].sum())
    excluded = cut_demand-cut_capacity > tolerance
    return dict(required=float(demands.sum()), relaxed_deliverable=float(flow.sum()),
                total_capacity=float(capacities.sum()), relaxed_routing_infeasible=bool(excluded),
                cut_agents=np.flatnonzero(reachable_agents).tolist(),
                cut_resources=np.flatnonzero(reachable_resources).tolist(),
                cut_demand=cut_demand, cut_capacity=cut_capacity, numerical_tolerance=tolerance,
                interpretation='Static routing relaxation; infeasibility is a capacity obstruction for the supplied demands.')


def strict_safety_bound(world, scenario, window, floor):
    if window <= 0 or not 0 < floor <= 1:
        raise ValueError('invalid strict-safety window or floor')
    simulator = Simulator(world, scenario, Policy())
    amplitude = scenario.theta if scenario.feedback == 'manuscript' else max(scenario.theta-1, 0)
    maximum_multiplier = min(3, 1+amplitude/8)
    capacities = (1-floor)*world.capacity+window*world.renewal*maximum_multiplier
    # Isolated agents cannot withdraw water and cannot force depletion. The main
    # ensemble has none; the unconditioned robustness ensemble can contain them.
    active = simulator.access.any(axis=1)
    result = routing_bound(simulator.access[active], window*simulator.cooperative_request[active], capacities)
    result['cut_agents'] = np.flatnonzero(active)[result['cut_agents']].tolist()
    return result | dict(strict_safety_ruled_out=result['relaxed_routing_infeasible'],
        excluded_isolated_agents=int((~active).sum()),
        window=window, resource_floor=floor, maximum_renewal_multiplier=maximum_multiplier,
        scope='strict_zero_depletion_over_this_finite_window')
