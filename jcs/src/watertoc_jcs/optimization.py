import numpy as np
from scipy.optimize import NonlinearConstraint, differential_evolution
from scipy.stats import qmc
from .payoffs import Policy, resource_only_encoding, effective_change_interval
from .provenance import rng, seed


def one_dimensional(evaluate, budget, t, s, norm, settings, bounds, family, feasibility=None):
    low, high = bounds
    if family == 'equal_relative':
        low, high = max(low, -budget), min(high, budget)
        encode = lambda k: Policy.from_relative(k, k, t, s)
    elif family == 'resource_only':
        low, high = effective_change_interval(budget, t, s, norm, bounds)
        encode = lambda k: resource_only_encoding(float(k), t, s, bounds, norm)
    elif family == 'redistribution':
        ratio = (t-1)/(-s)
        limit = budget*(np.sqrt(2/(1+ratio**2)) if norm == 'L2' else 2/(1+ratio))
        low, high = max(low, -bounds[1]/ratio, -limit), min(high, -bounds[0]/ratio, limit)
        encode = lambda k: Policy(float((t-1)*k), float(-(t-1)*k), t, s)
    else:
        raise ValueError('unknown one-dimensional comparator')
    values = {}
    def sample(points):
        for point in points:
            if float(point) not in values:
                policy = encode(point)
                if policy.cost(norm) > budget+1e-12:
                    raise ArithmeticError('infeasible comparator encoding')
                value = evaluate(policy)
                score = (max(0., feasibility(policy)), value) if feasibility else value
                values[float(point)] = (score, policy)
    sample(np.unique(np.r_[np.linspace(low, high, settings['initial_grid_points']), 0.]))
    for _ in range(settings['refinement_rounds']):
        points = sorted(values)
        intervals = list(zip(points[:-1], points[1:]))
        intervals.sort(key=lambda pair: (min(values[q][0] for q in pair), max(values[q][0] for q in pair), pair))
        for left, right in intervals[:settings['refined_intervals_per_round']]:
            sample(np.linspace(left, right, settings['refinement_grid_points']))
    return [pair[1] for _, pair in sorted(values.items(),
            key=lambda item: (item[1][0], item[1][1].cost(norm), item[0]))]


def differential_search(evaluate, budget, t, s, norm, bounds, settings, optimizer_seed, equal_relative, feasibility=None):
    observation = settings.get('observation', 'local')
    if budget == 0:
        evaluate(Policy(t=t, s=s, observation=observation))
        return
    population = settings['explicit_population_size']
    samples = qmc.Sobol(2, scramble=True, seed=seed(optimizer_seed, 'sobol') % 2**32).random_base2(int(np.ceil(np.log2(population))))
    members = []
    for radial, angular in samples:
        direction = np.array([np.cos(2*np.pi*angular), np.sin(2*np.pi*angular)])
        cost = np.linalg.norm(direction)/np.sqrt(2) if norm == 'L2' else np.abs(direction).mean()
        delta = direction*(budget*np.sqrt(radial)/cost)
        ratio = np.min(np.divide(np.where(delta >= 0, bounds[1], -bounds[0]), np.abs(delta),
                                 out=np.full(2, np.inf), where=np.abs(delta) > 0))
        members.append(delta*min(1, ratio))
    members = np.asarray(members[:population])
    members[0], members[1] = [0., 0.], equal_relative.relative
    make = lambda r: Policy.from_relative(float(r[0]), float(r[1]), t, s, observation)
    constraints = [NonlinearConstraint(lambda r: make(r).cost(norm), 0, budget+1e-14)]
    if feasibility:
        def outcome_constraint(r):
            policy = make(r)
            return 1. if policy.cost(norm) > budget+1e-14 else feasibility(policy)
        constraints.append(NonlinearConstraint(outcome_constraint, -np.inf, 0))
    differential_evolution(lambda r: evaluate(make(r)), [bounds, bounds], strategy=settings['strategy'],
                           init=members, maxiter=settings['maximum_generations'], tol=0, atol=0,
                           mutation=tuple(settings['mutation']), recombination=settings['recombination'],
                           rng=rng(optimizer_seed, 'differential-evolution'), constraints=tuple(constraints),
                           updating='deferred', polish=False, workers=1)
