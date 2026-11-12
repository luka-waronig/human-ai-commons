"""Constrained Pareto search and independent assessment of payoff designs."""
from dataclasses import replace
from itertools import product
import json
import numpy as np
from .config import metric_spec, problems, robustness_variants, stress_spec, scenario
from .capacity import strict_safety_bound
from .metrics import paired_interval, probability_upper_bound
from .payoffs import Policy, effective_change_interval, resource_only_encoding
from .provenance import identifier, rng, write_json, append_json, read_jsonl


OBJECTIVES = ('mean_human_service', 'mean_ai_service', 'mean_loss')


def enabled(config):
    return config['objective']['name'] == 'pareto_human_ai_depletion'


def values(row):
    """All coordinates are minimized; human and AI support change sign."""
    result = np.array([-row[OBJECTIVES[0]], -row[OBJECTIVES[1]], row[OBJECTIVES[2]]])
    if not np.isfinite(result).all():
        raise ValueError('finite Pareto objectives required')
    return result


def nondominated(rows):
    """Exact dominance of sample means, without an assumed welfare weight."""
    ordered = sorted(rows, key=lambda r: (Policy(**r['policy']).cost(), r['policy_id']))
    unique = {}
    for row in ordered:
        unique.setdefault(tuple(values(row)), row)
    rows = list(unique.values())
    if not rows:
        return []
    matrix = np.array([values(r) for r in rows])
    return [row for i, row in enumerate(rows)
            if not np.any(np.all(matrix <= matrix[i], axis=1) & np.any(matrix < matrix[i], axis=1))]


def spread(rows, limit):
    """Selection of objective extremes and gaps in normalized outcome space."""
    rows = sorted({r['policy_id']: r for r in rows}.values(), key=lambda r: r['policy_id'])
    if len(rows) <= limit:
        return rows
    matrix = np.array([values(r) for r in rows])
    chosen = list(dict.fromkeys(int(np.argmin(matrix[:, k])) for k in range(3)))
    if limit < len(chosen):
        raise ValueError('selection limit cannot preserve all objective ends')
    while len(chosen) < limit:
        distances = ((matrix[:, None, :]-matrix[chosen][None, :, :])**2).sum(axis=2).min(axis=1)
        distances[chosen] = -1
        chosen.append(int(np.argmax(distances)))
    return [rows[i] for i in chosen]


def select(rows, limit):
    feasible = [r for r in rows if r['search_feasible']]
    if not feasible and rows:
        least = min(r['stress_probability_violation'] for r in rows)
        feasible = [r for r in rows if r['stress_probability_violation'] == least]
    return spread(nondominated(feasible), limit)


def grid_search(evaluate, axes, encode, admissible, rounds, anchors):
    """A fixed grid followed by shrinking neighborhoods of diverse Pareto points.

    Every admissible initial grid node is evaluated. Refinement is an empirical
    search over continuous parameters, not a certificate of global optimality.
    """
    evaluated = {}
    def sample(points):
        for point in points:
            point = tuple(float(x) for x in point)
            if point not in evaluated and all(a[0] <= x <= a[-1] for x, a in zip(point, axes)):
                policy = encode(point)
                if admissible(policy):
                    evaluated[point] = evaluate(policy)
    sample(product(*axes))
    sample([tuple(0. for _ in axes)])
    spacing = [float(np.max(np.diff(a))) if len(a) > 1 else 0. for a in axes]
    for iteration in range(rounds):
        candidates = select(list(evaluated.values()), anchors)
        centers = {r['policy_id'] for r in candidates}
        points = [point for point, row in evaluated.items() if row['policy_id'] in centers]
        for center in points:
            sample(product(*[[x+d*step/2**(iteration+1) for d in [-1, 0, 1]]
                             for x, step in zip(center, spacing)]))
    return list({r['policy_id']: r for r in evaluated.values()}.values())


def train(ctx):
    config, search = ctx.config, ctx.config['search']
    p, settings = config['payoffs'], search['pareto']
    t, s, bounds = p['t'], p['s'], p['relative_bounds']
    shortlist = config['validation']['maximum_shortlisted_candidates_per_family_and_problem']
    design, archive = [], {}
    for problem in problems(config):
        pressure, budget, norm = problem['pressure'], problem['budget'], problem['norm']
        def evaluate(policy):
            ctx.objective(pressure, policy)
            return ctx.candidates[(pressure, policy.observation, policy.id)]
        def admissible(policy):
            return policy.cost(norm) <= budget+1e-12
        n = next((m['points_per_axis'] for m in search['payoff_maps'] if m['pressure'] == pressure),
                 settings['initial_points_per_axis'])
        axis = np.unique(np.r_[np.linspace(*bounds, n), 0.])
        families = {}
        line_points = search['one_dimensional_search']['initial_grid_points']
        for family in ['equal_relative', 'resource_only', 'redistribution']:
            if family == 'resource_only':
                low, high = effective_change_interval(budget, t, s, norm, bounds)
                encode = lambda q: resource_only_encoding(q[0], t, s, bounds, norm)
            elif family == 'equal_relative':
                low, high = max(bounds[0], -budget), min(bounds[1], budget)
                encode = lambda q: Policy.from_relative(q[0], q[0], t, s)
            else:
                ratio = (t-1)/(-s)
                limit = budget*(np.sqrt(2/(1+ratio**2)) if norm == 'L2' else 2/(1+ratio))
                low, high = max(bounds[0], -bounds[1]/ratio, -limit), min(bounds[1], -bounds[0]/ratio, limit)
                encode = lambda q: Policy((t-1)*q[0], -(t-1)*q[0], t, s)
            families[family] = grid_search(evaluate, [np.unique(np.r_[np.linspace(low, high, line_points), 0.])],
                                          encode, admissible, settings['refinement_rounds'], settings['refinement_anchors'])
        families['local'] = grid_search(evaluate, [axis, axis],
            lambda q: Policy.from_relative(*q, t, s), admissible,
            settings['refinement_rounds'], settings['refinement_anchors'])
        # The local family contains both of these one-dimensional families.
        families['local'] += families['equal_relative']+families['redistribution']
        global_problem = search.get('global_composition_reoptimization')
        if norm == 'L2' and global_problem and all(problem[k] == global_problem[k] for k in ['pressure', 'budget']):
            families['global'] = grid_search(evaluate, [axis, axis],
                lambda q: Policy.from_relative(*q, t, s, observation='global'), admissible,
                settings['refinement_rounds'], settings['refinement_anchors'])
        for family, rows in families.items():
            archive.setdefault((pressure, family), {}).update({r['policy_id']: r for r in rows})
        design.append(dict(problem=problem, problem_id=identifier(problem)[:16], families=list(families)))
    # Evaluation sharing before shortlisting gives larger admissible sets
    # access to all smaller-budget designs in the same family.
    for row in design:
        problem = row['problem']
        families = {family: [r for r in archive[(problem['pressure'], family)].values()
                    if Policy(**r['policy']).cost(problem['norm']) <= problem['budget']+1e-12]
                    for family in row['families']}
        row['families'] = {family: select(rows, shortlist) for family, rows in families.items()}
        row['evaluated_counts'] = {family: len(rows) for family, rows in families.items()}
    write_json(ctx.output/'shortlists.json', design)


def validate(ctx):
    design = json.loads((ctx.output/'shortlists.json').read_text())
    frozen = []
    for row in design:
        assessed = {family: [ctx.aggregate(row['problem']['pressure'], Policy(**r['policy']), 'validation',
                    ctx.config['validation']['validation_realizations']) for r in candidates]
                    for family, candidates in row['families'].items()}
        # Independently assessed nested candidates also belong to the local family.
        assessed['local'] += assessed['equal_relative']+assessed['redistribution']
        families = {family: select(candidates, ctx.config['validation']['maximum_tested_candidates_per_family_and_problem'])
                    for family, candidates in assessed.items()}
        for candidates in families.values():
            for candidate in candidates:
                candidate['selection_status'] = ('empirically_feasible' if candidate['search_feasible']
                                                  else 'diagnostic_no_empirically_feasible_candidate')
        frozen.append(dict(problem=row['problem'], problem_id=row['problem_id'], families=families,
                           validation_candidates=assessed))
    write_json(ctx.output/'frozen_policies.json', frozen)


def _primary(rows, config):
    target = config['search'].get('global_composition_reoptimization') or config['search']['main_problems'][0]
    return next(row for row in rows if row['problem']['norm'] == 'L2'
                and all(row['problem'][key] == target[key] for key in ['pressure', 'budget']))


def assess_constraints(rows, confidence, maximum):
    unique = {(r['problem']['pressure'], c['policy_id']) for r in rows
              for candidates in r['families'].values() for c in candidates}
    adjusted = 1-(1-confidence)/max(1, len(unique))
    for row in rows:
        for candidates in row['families'].values():
            for candidate in candidates:
                upper = probability_upper_bound(np.array(candidate['stress_events'], dtype=bool), adjusted)
                candidate.update(simultaneous_stress_probability_upper=upper,
                                 simultaneous_protection_supported=bool(upper <= maximum),
                                 protected_family_size=len(unique), family_confidence=confidence)
    return len(unique)


def test(ctx):
    frozen = json.loads((ctx.output/'frozen_policies.json').read_text())
    config, count = ctx.config, ctx.config['validation']['test_realizations']
    results = []
    for row in frozen:
        families = {family: [ctx.aggregate(row['problem']['pressure'], Policy(**c['policy']), 'test', count, record=True)
                             for c in candidates] for family, candidates in row['families'].items()}
        reference = ctx.aggregate(row['problem']['pressure'], Policy(t=config['payoffs']['t'], s=config['payoffs']['s']),
                                  'test', count, record=True)
        families['reference'] = [reference]
        results.append(dict(problem=row['problem'], problem_id=row['problem_id'], families=families))
    confidence = config['validation'].get('confidence_level', .95)
    claims = assess_constraints(results, confidence, stress_spec(config).maximum_probability)
    comparisons = []
    for row in results:
        baseline = row['families']['reference'][0]
        for family, candidates in row['families'].items():
            for candidate in candidates:
                if candidate['world_ids'] != baseline['world_ids']:
                    raise ArithmeticError('frontier comparison requires paired worlds')
                for key, field in [('human', 'human_services'), ('ai', 'ai_services'), ('depletion', 'losses')]:
                    differences = np.array(candidate[field])-np.array(baseline[field])
                    interval = paired_interval(differences, rng('pareto-comparison', row['problem_id'], candidate['policy_id'], key),
                                               confidence=confidence)
                    # Differences lie in [-1,1]. A union bound covers all tested
                    # coordinates without relying on a normal approximation.
                    radius = np.sqrt(2*np.log(6*max(1, claims)/(1-confidence))/count)
                    comparisons.append(dict(problem_id=row['problem_id'], **row['problem'], family=family,
                        policy_id=candidate['policy_id'], outcome=key, difference=float(differences.mean()),
                        lower=interval['lower'], upper=interval['upper'],
                        simultaneous_lower=max(-1., float(differences.mean()-radius)),
                        simultaneous_upper=min(1., float(differences.mean()+radius)),
                        interval_scope='bootstrap descriptive; simultaneous bounds use bounded independent differences',
                        constraint_supported=candidate['simultaneous_protection_supported']))
    anchor = _primary(frozen, config)
    existing = {(r['pressure'], r['world_id']) for r in read_jsonl(ctx.output/'capacity_bounds.jsonl')}
    for pressure in sorted({r['problem']['pressure'] for r in results}):
        for index in range(count):
            world = ctx.world('test', index)
            if (pressure, world.id) not in existing:
                bound = strict_safety_bound(world, scenario(config, pressure),
                    config['objective']['total_steps']-config['objective']['excluded_initial_steps'], metric_spec(config).floor)
                append_json(ctx.output/'capacity_bounds.jsonl', dict(pressure=pressure, index=index, world_id=world.id, **bound))
    controls = []
    for c in anchor['families']['local']:
        controls.append(dict(policy_id=c['policy_id'], observations={observation: ctx.aggregate(anchor['problem']['pressure'],
            replace(Policy(**c['policy']), observation=observation), 'test', count, record=True)
            for observation in ['local', 'global', 'fixed']}))
    write_json(ctx.output/'test_comparisons.json', dict(mode='pareto', problems=results,
        paired_outcome_comparisons=comparisons, same_matrix_observation_controls=controls,
        primary_problem=anchor['problem'], interpretation='Estimated finite-sample frontiers; no global optimality certificate.'))


def robustness(ctx):
    frozen = json.loads((ctx.output/'frozen_policies.json').read_text())
    anchor = _primary(frozen, ctx.config)
    count = ctx.config['validation'].get('robustness_realizations', ctx.config['validation']['validation_realizations'])
    rows = []
    for variant in robustness_variants(ctx.config):
        families = {family: [ctx.aggregate(anchor['problem']['pressure'], Policy(**c['policy']), 'robustness', count,
                             variant=variant, record=False) for c in candidates]
                    for family, candidates in anchor['families'].items()}
        families['reference'] = [ctx.aggregate(anchor['problem']['pressure'],
            Policy(t=ctx.config['payoffs']['t'], s=ctx.config['payoffs']['s']), 'robustness', count, variant=variant)]
        assessment = dict(problem=anchor['problem'], families=families)
        assess_constraints([assessment], ctx.config['validation'].get('confidence_level', .95), stress_spec(ctx.config).maximum_probability)
        rows.append(dict(variant=variant, families=families, interpretation='Transfer of a frozen set; no reoptimization.'))
    write_json(ctx.output/'robustness.json', rows)
