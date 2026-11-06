from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from .dynamics import Scenario
from .metrics import MetricSpec, StressSpec, WelfareSpec
from .world import WorldSpec
from .payoffs import Policy


def merge(base, changes):
    result = deepcopy(base)
    for key, value in changes.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def load_config(path, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError('cyclic configuration inheritance')
    seen.add(path)
    config = json.loads(path.read_text())
    parent = config.pop('extends', None)
    if parent:
        parent_path = (path.parent/parent).resolve()
        if not parent_path.is_relative_to(path.parent):
            raise ValueError('configuration parent must be inside the configuration folder')
        config = merge(load_config(parent_path, seen), config)
    validate_config(config)
    return config


def world_spec(config):
    w = config['world']
    if w.get('condition') not in ['unconditioned', 'placements_conditioned_on_access_given_landscape']:
        raise ValueError('unknown placement condition')
    return WorldSpec(width=w['width'], height=w['height'], humans=w['humans'], ai=w['ai'],
                     density=w['water_cell_density'], capacity=w['water_capacity'],
                     renewal_low=w['baseline_renewal_uniform'][0], renewal_high=w['baseline_renewal_uniform'][1],
                     maximum_occupancy=w['maximum_agents_per_cell'], access_radius=w['access_radius'],
                     feedback_radius=w['feedback_radius'], game_radius=w['payoff_composition_radius'],
                     initial_cooperation=w['initial_cooperation_fraction'],
                     require_access=w.get('condition', '') != 'unconditioned')


def scenario(config, pressure):
    w = config['world']
    return Scenario(pressure=pressure, demand_ratio=w['ai_to_human_cooperative_demand_ratio'],
                    defect_ratio=w['defect_to_cooperate_request_ratio'], theta=w['theta'],
                    epsilon=w['action_reversal_probability'], feedback=config['reference_feedback_rule'],
                    allocation=w.get('withdrawal_allocation', 'human_priority'))


def metric_spec(config):
    values = config.get('depletion_diagnostic', config['objective'])
    return MetricSpec(values['resource_floor'], values['spatial_upper_tail_fraction'])


def welfare_spec(config):
    if config['objective']['name'] == 'expected_spatial_upper_tail_cumulative_depletion':
        return None
    if config['objective']['name'] != 'joint_service_welfare_with_persistent_local_stress_constraint':
        raise ValueError('unknown optimization objective')
    w = config['objective']['welfare']
    if w['service_reference'] != 'fixed_defection_request':
        raise ValueError('service target must be fixed independently of the chosen strategy')
    return WelfareSpec(w['human_weight'], w['utility'], w.get('additional_use_value', 1))


def stress_spec(config):
    values = config['objective'].get('persistent_local_stress')
    return StressSpec(**values) if values else None


def validate_config(config):
    world_spec(config)
    scenario(config, 1)
    metric_spec(config)
    welfare_spec(config)
    stress_spec(config)
    w, p, o, s, v = (config[name] for name in ['world', 'payoffs', 'objective', 'search', 'validation'])
    if o['name'] not in ['expected_spatial_upper_tail_cumulative_depletion', 'joint_service_welfare_with_persistent_local_stress_constraint']:
        raise ValueError('unknown optimization objective')
    if welfare_spec(config) is not None and (w['humans'] < 1 or w['ai'] < 1 or stress_spec(config) is None):
        raise ValueError('joint welfare requires both agent types and an explicit stress constraint')
    if not 0 < v.get('confidence_level', .95) < 1:
        raise ValueError('confidence level must be in (0,1)')
    if welfare_spec(config):
        checks = config['secondary_checks']
        for weight in checks['welfare_human_weight']:
            WelfareSpec(human_weight=weight)
        for utility in checks['delivery_utility']:
            WelfareSpec(utility=utility)
        for extra in checks['additional_use_value']:
            WelfareSpec(additional_use_value=extra)
        for reference in checks['stress_reference']:
            StressSpec(reference=reference)
        for duration in checks['stress_duration_steps']:
            StressSpec(duration_steps=duration)
    if not w['toroidal'] or w['initial_water'] != 'capacity' or w['payoff_composition_center'] != 'selected_resource_cell':
        raise ValueError('unsupported world or observation convention')
    if not w['exclude_focal_agent_from_payoff_composition'] or w['empty_composition_value'] != .5:
        raise ValueError('unsupported focal-agent or empty-neighborhood convention')
    if not w['freeze_strategies_before_replenishment_and_planning']:
        raise ValueError('reference schedule must freeze observations; use an explicit scheduling sensitivity')
    if o['observation'] != 'post_withdrawal' or o['resource_cells'] != 'fixed_geometrically_accessible_water_cells':
        raise ValueError('unsupported objective convention')
    if not isinstance(o['total_steps'], int) or not isinstance(o['excluded_initial_steps'], int) or not 0 <= o['excluded_initial_steps'] < o['total_steps']:
        raise ValueError('invalid observation horizon')
    if stress_spec(config) and stress_spec(config).duration_steps > o['total_steps']-o['excluded_initial_steps']:
        raise ValueError('stress duration cannot exceed the observation horizon')
    if p.get('optimized_parameters') != ['u', 'v']:
        raise ValueError('the journal protocol requires the u,v endpoint matrix')
    Policy(t=p['t'], s=p['s'])
    if (p['optimized_parameters'] != ['u', 'v'] or p['tie_action'] != 'C'
            or len(p['relative_bounds']) != 2 or not all(math.isfinite(x) for x in p['relative_bounds'])
            or not -1 < p['relative_bounds'][0] < 0 < p['relative_bounds'][1]):
        raise ValueError('unsupported payoff convention')
    for field in ['characterization_realizations', 'training_realizations', 'validation_realizations', 'test_realizations', 'maximum_shortlisted_candidates_per_family_and_problem']:
        minimum = 3 if field == 'maximum_shortlisted_candidates_per_family_and_problem' else (2 if field == 'test_realizations' else 1)
        if isinstance(v[field], bool) or not isinstance(v[field], int) or v[field] < minimum:
            raise ValueError(f'invalid {field}')
    if s['explicit_population_size'] < 5 or s['maximum_generations'] < 0 or s['optimizer_restarts'] < 1:
        raise ValueError('invalid DE budget')
    for key in ['explicit_population_size', 'maximum_generations', 'optimizer_restarts']:
        if isinstance(s[key], bool) or not isinstance(s[key], int):
            raise ValueError(f'integer {key} required')
    if s['strategy'] != 'best1bin' or s['updating'] != 'deferred' or s['polish']:
        raise ValueError('unsupported optimizer convention')
    if s['tol'] != 0 or s['atol'] != 0:
        raise ValueError('fixed-budget search requires zero tolerance stopping settings')
    if (len(s['mutation']) != 2 or not all(math.isfinite(x) for x in s['mutation'])
            or not 0 <= s['mutation'][0] <= s['mutation'][1] < 2
            or not math.isfinite(s['recombination']) or not 0 <= s['recombination'] <= 1):
        raise ValueError('invalid DE mutation or crossover settings')
    for key in ['initial_grid_points', 'refined_intervals_per_round', 'refinement_grid_points', 'refinement_rounds']:
        value = s['one_dimensional_search'][key]
        minimum = 0 if key == 'refinement_rounds' else (2 if 'grid_points' in key else 1)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'invalid comparator setting: {key}')
    for mapping in s['payoff_maps']:
        scenario(config, mapping['pressure'])
        count = mapping['points_per_axis']
        if isinstance(count, bool) or not isinstance(count, int) or count < 2:
            raise ValueError('payoff maps require an integer grid size >= 2')
    for problem in s['main_problems']:
        scenario(config, problem['pressure'])
        if not math.isfinite(problem['budget']) or problem['budget'] < 0:
            raise ValueError('negative intervention budget')
    for problem in problems(config):
        scenario(config, problem['pressure'])
        if not math.isfinite(problem['budget']) or problem['budget'] < 0:
            raise ValueError('invalid intervention budget')
    target = s.get('global_composition_reoptimization')
    if target and target not in s['main_problems']:
        raise ValueError('global comparison must match a main search problem')
    names = set()
    for variant in robustness_variants(config):
        allowed = {'name', 'world_transform', 'world_spec', 'initial_fraction', 'scenario', 'policy', 'steps', 'burn'}
        if set(variant)-allowed or variant.get('world_transform') not in [None, 'cluster_ai']:
            raise ValueError('unknown robustness option')
        if not isinstance(variant.get('name'), str) or variant['name'] in names:
            raise ValueError('robustness variants require unique names')
        names.add(variant['name'])
        replace(world_spec(config), **variant.get('world_spec', {}))
        replace(scenario(config, 1), **variant.get('scenario', {}))
        replace(Policy(t=p['t'], s=p['s']), **variant.get('policy', {}))
        fraction = variant.get('initial_fraction', w['initial_cooperation_fraction'])
        steps, burn = variant.get('steps', o['total_steps']), variant.get('burn', o['excluded_initial_steps'])
        if not 0 <= fraction <= 1 or any(isinstance(x, bool) or not isinstance(x, int) for x in [steps, burn]) or not 0 <= burn < steps:
            raise ValueError('invalid robustness fraction or horizon')
        if stress_spec(config) and stress_spec(config).duration_steps > steps-burn:
            raise ValueError('stress duration cannot exceed a robustness observation horizon')


def problems(config):
    rows = [dict(p, norm='L2') for p in config['search']['main_problems']]
    alternative = config['secondary_checks'].get('budget_norm')
    if alternative and config.get('execution', {}).get('include_l1_search', True):
        rows.append(dict(pressure=alternative['reoptimize_pressure'], budget=alternative['reoptimize_budget'], norm='L1'))
    return rows


def budget_plan(config):
    """Conservative upper bounds; no worlds or simulations are constructed."""
    search, validation, objective = (config[name] for name in ['search', 'validation', 'objective'])
    rows = problems(config)
    de = search['explicit_population_size']*(1+search['maximum_generations'])*search['optimizer_restarts']
    comparison = search['one_dimensional_search']
    one_dimensional = comparison['initial_grid_points']+1+comparison['refined_intervals_per_round']*comparison['refinement_grid_points']*comparison['refinement_rounds']
    maps = sum(row['points_per_axis']**2 for row in search['payoff_maps'])
    global_control = int(bool(search.get('global_composition_reoptimization')))
    candidates = len(rows)*(de+3*one_dimensional)+maps+global_control*de
    slots = validation['maximum_shortlisted_candidates_per_family_and_problem']
    validation_cap = len(rows)*4*slots+global_control*slots
    pressures = sorted(set(p['pressure'] for p in rows))
    characterization = 3*len(pressures)*validation['characterization_realizations']
    # Five extra policies cover same-matrix observation controls and forced diagnostics.
    test_cap = len(rows)*4+global_control+len(pressures)*3+5
    steps = objective['total_steps']
    variants = robustness_variants(config)
    robust_steps = sum((variant.get('steps', steps))*(5+global_control)*validation.get('robustness_realizations', validation['validation_realizations']) for variant in variants)
    caps = dict(characterize=characterization*steps, train=candidates*validation['training_realizations']*steps,
                validate=validation_cap*validation['validation_realizations']*steps,
                test=test_cap*validation['test_realizations']*steps, robustness=robust_steps)
    return dict(candidate_caps=dict(characterize=characterization, train=candidates, validate=validation_cap, test=test_cap),
                model_step_caps=caps, all_model_steps=sum(caps.values()),
                world=asdict(world_spec(config)), simulations_started=False,
                interpretation='Upper bounds before cache reuse. Metrics and analysis do not run simulations.')


def robustness_variants(config):
    if 'robustness_variants' in config.get('execution', {}):
        return config['execution']['robustness_variants']
    s, v = config['secondary_checks'], config['validation']
    variants = [dict(name='clustered_ai', world_transform='cluster_ai'),
                dict(name='human_priority_allocation', scenario={'allocation': 'human_priority'}),
                dict(name='sequential_planning', scenario={'schedule': 'sequential_planning'}),
                dict(name='manuscript_feedback', scenario={'feedback': s['feedback_law']}),
                dict(name='long_horizon', steps=v['long_horizon_total_steps'], burn=v['long_horizon_excluded_initial_steps']),
                dict(name='larger_grid', world_spec=s['grid']),
                dict(name='unconditioned_placement', world_spec={'require_access': False})]
    for key, runtime_key in [('theta', 'theta'), ('action_reversal_probability', 'epsilon'),
                             ('ai_to_human_demand_ratio_at_fixed_total_load', 'demand_ratio')]:
        variants += [dict(name=f'{runtime_key}_{value}', scenario={runtime_key: value}) for value in s[key]]
    variants += [dict(name=f'initial_{value}', initial_fraction=value) for value in s['initial_cooperation']]
    return variants
