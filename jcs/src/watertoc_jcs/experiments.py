"""Staged searches, policy freezing, held-out comparisons, and replay bundles."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import numpy as np
from .capacity import strict_safety_bound
from .config import budget_plan, metric_spec, problems, robustness_variants, scenario, stress_spec, welfare_spec, world_spec
from .dynamics import ArchivedDrivers, Drivers, Scenario, simulate
from .metrics import MetricSpec, StressSpec, WelfareSpec, paired_interval, probability_upper_bound
from .optimization import differential_search, one_dimensional
from .payoffs import Policy
from .provenance import append_json, environment, identifier, read_jsonl, rng, seed, sha256, source_files, source_hashes, utc_now, write_json
from .world import World, generate_world, transform_world

STAGES = ['characterize', 'train', 'validate', 'test', 'robustness']


def bundle_hashes(output):
    return {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*'))
            if p.is_file() and p.relative_to(output) != Path('checksums.json')
            and p.relative_to(output).parts[0] != 'analysis'}


def candidate_key(row, norm):
    return (max(0., row.get('stress_probability_violation', 0.)),
            row.get('objective_loss', row['mean_loss']), Policy(**row['policy']).cost(norm), row['policy_id'])


def problem_id(problem):
    return identifier(problem)[:16]


def nested_shortlist(ranked, nested, slots):
    selected = {}
    nested = [nested] if isinstance(nested, dict) else nested
    for row in [*nested, *ranked]:
        selected.setdefault(row['policy_id'], row)
        if len(selected) == slots:
            break
    return list(selected.values())


class Evaluation:
    def __init__(self, config, output, maximum_steps):
        self.config, self.output = config, Path(output)
        self.maximum_steps, self.used = maximum_steps, 0
        self.world_cache, self.candidates = {}, {}
        self.encountered = set()
        self.cache = {r['evaluation_id']: r for r in read_jsonl(self.output/'realizations.jsonl')}
        for row in read_jsonl(self.output/'training_candidates.jsonl'):
            self.candidates[(row['pressure'], row['policy']['observation'], row['policy_id'])] = row

    def world(self, split, index, variant=None):
        variant = variant or {}
        spec = replace(world_spec(self.config), **variant.get('world_spec', {}))
        key = identifier([spec, split, index, variant.get('initial_fraction'), variant.get('world_transform')])
        if key not in self.world_cache:
            world_seed = seed(self.config['validation']['base_seed'], split, index, 'world')
            world = generate_world(spec, world_seed)
            if variant.get('world_transform') == 'cluster_ai' or 'initial_fraction' in variant:
                world = transform_world(world, cluster_ai=variant.get('world_transform') == 'cluster_ai',
                                        initial_fraction=variant.get('initial_fraction'), transform_seed=world_seed)
            path = self.output/'worlds'/f'{world.id}.npz'
            if not path.exists():
                world.save(path)
            self.world_cache[key] = world
        return self.world_cache[key]

    def realization(self, pressure, policy, split, index, *, variant=None, record=False):
        variant = variant or {}
        world = self.world(split, index, variant)
        physical = replace(scenario(self.config, pressure), **variant.get('scenario', {}))
        policy = replace(policy, **variant.get('policy', {}))
        objective = self.config['objective']
        steps = variant.get('steps', objective['total_steps'])
        burn = variant.get('burn', objective['excluded_initial_steps'])
        metric = metric_spec(self.config)
        driver_seed = seed(self.config['validation']['base_seed'], split, index, 'trajectory')
        welfare, stress = welfare_spec(self.config), stress_spec(self.config)
        inputs = dict(world_id=world.id, scenario=asdict(physical), policy=asdict(policy), driver_seed=driver_seed,
                      steps=steps, burn=burn, metric=asdict(metric),
                      welfare_spec=asdict(welfare) if welfare else None, stress_spec=asdict(stress) if stress else None)
        evaluation_id = identifier(inputs)[:32]
        trajectory = self.output/'trajectories'/f'{evaluation_id}.npz'
        if evaluation_id in self.cache and (not record or trajectory.exists()):
            return self.cache[evaluation_id]
        if self.used+steps > self.maximum_steps:
            raise RuntimeError(f'model-step budget exhausted: {self.used}+{steps}>{self.maximum_steps}')
        self.used += steps
        append_json(self.output/'budget_ledger.jsonl', dict(evaluation_id=evaluation_id, steps=steps,
                    split=split, status='reserved', invocation_steps_used=self.used))
        try:
            summary, data = simulate(world, physical, policy, driver_seed, steps, burn, metric, record=record,
                                     welfare=welfare, stress=stress)
        except Exception:
            append_json(self.output/'budget_ledger.jsonl', dict(evaluation_id=evaluation_id, steps=0, status='failed'))
            raise
        row = inputs | summary | dict(evaluation_id=evaluation_id, split=split, index=index,
                                      variant=variant.get('name', 'reference'), world_path=f'worlds/{world.id}.npz')
        if record:
            trajectory.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(trajectory, **data)
            driver_id = identifier([driver_seed, steps, len(world.positions)])[:24]
            driver_path = self.output/'drivers'/f'{driver_id}.npz'
            driver_path.parent.mkdir(parents=True, exist_ok=True)
            if not driver_path.exists():
                Drivers(driver_seed, len(world.positions)).archive(driver_path, steps)
            row.update(trajectory_path=str(trajectory.relative_to(self.output)),
                       drivers_path=str(driver_path.relative_to(self.output)))
        append_json(self.output/'realizations.jsonl', row)
        self.cache[evaluation_id] = row
        append_json(self.output/'budget_ledger.jsonl', dict(evaluation_id=evaluation_id, steps=0,
                                                          completed_steps=steps, status='complete'))
        return row

    def aggregate(self, pressure, policy, split, count, *, variant=None, record=False):
        records = [self.realization(pressure, policy, split, i, variant=variant, record=record) for i in range(count)]
        losses = [r['loss'] for r in records]
        result = dict(mean_loss=float(np.mean(losses)), variance=float(np.var(losses, ddof=1)) if count > 1 else None,
                    losses=losses, realization_ids=[r['evaluation_id'] for r in records],
                    world_ids=[r['world_id'] for r in records], count=count,
                    policy=records[0]['policy'], policy_id=records[0]['policy_id'])
        if 'welfare' in records[0]:
            values = [r['welfare'] for r in records]
            events = np.array([r['persistent_local_stress'] for r in records], dtype=bool)
            probability = float(events.mean())
            maximum = stress_spec(self.config).maximum_probability
            confidence = self.config['validation'].get('confidence_level', .95)
            upper = probability_upper_bound(events, confidence)
            result.update(welfares=values, mean_welfare=float(np.mean(values)), objective_loss=1-float(np.mean(values)),
                          human_services=[r['human_service'] for r in records],
                          ai_services=[r['ai_service'] for r in records],
                          mean_human_service=float(np.mean([r['human_service'] for r in records])),
                          mean_ai_service=float(np.mean([r['ai_service'] for r in records])),
                          stress_events=events.tolist(), stress_probability=probability,
                          persistence_basis=stress_spec(self.config).basis,
                          stress_probability_violation=probability-maximum,
                          search_feasible=bool(probability <= maximum),
                          stress_probability_upper=upper, stress_bound_confidence=confidence,
                          protection_supported=bool(upper <= maximum))
        else:
            result['objective_loss'] = result['mean_loss']
        return result

    def objective(self, pressure, policy):
        key = (pressure, policy.observation, policy.id)
        self.encountered.add(key)
        if key not in self.candidates:
            row = self.aggregate(pressure, policy, 'train', self.config['validation']['training_realizations'])
            row.update(pressure=pressure, regime=policy.regime)
            self.candidates[key] = row
            append_json(self.output/'training_candidates.jsonl', row)
        return self.candidates[key]['objective_loss']

    def stress_constraint(self, pressure, policy):
        self.objective(pressure, policy)
        return self.candidates[(pressure, policy.observation, policy.id)].get('stress_probability_violation', 0.)

    def shortlist(self, problem, observation, count):
        rows = [r for key, r in self.candidates.items() if key in self.encountered
                and key[0] == problem['pressure'] and key[1] == observation
                and Policy(**r['policy']).cost(problem['norm']) <= problem['budget']+1e-12]
        return sorted(rows, key=lambda r: candidate_key(r, problem['norm']))[:count]


def characterize(ctx):
    config = ctx.config
    pressures = sorted(set(p['pressure'] for p in problems(config)))
    count = config['validation']['characterization_realizations']
    base = Policy(t=config['payoffs']['t'], s=config['payoffs']['s'])
    results = []
    for pressure in pressures:
        policies = dict(reference=base, always_C=replace(base, forced='C'), always_D=replace(base, forced='D'))
        results.append(dict(pressure=pressure, policies={name: ctx.aggregate(pressure, policy, 'characterization', count, record=True)
                                                       for name, policy in policies.items()}))
    write_json(ctx.output/'reference_characterization.json', results)


def train(ctx):
    config = ctx.config
    s, p = config['search'], config['payoffs']
    t, bounds = p['t'], tuple(p['relative_bounds'])
    slots = config['validation']['maximum_shortlisted_candidates_per_family_and_problem']
    for mapping in s['payoff_maps']:
        for rc in np.linspace(*bounds, mapping['points_per_axis']):
            for rd in np.linspace(*bounds, mapping['points_per_axis']):
                ctx.objective(mapping['pressure'], Policy.from_relative(float(rc), float(rd), t, p['s']))
    design = []
    for problem in problems(config):
        pressure, budget, norm = problem['pressure'], problem['budget'], problem['norm']
        evaluate = lambda policy: ctx.objective(pressure, policy)
        feasible = (lambda policy: ctx.stress_constraint(pressure, policy)) if welfare_spec(config) else None
        equal_relative = one_dimensional(evaluate, budget, t, p['s'], norm, s['one_dimensional_search'], bounds, 'equal_relative', feasible)
        resource_only = one_dimensional(evaluate, budget, t, p['s'], norm, s['one_dimensional_search'], bounds, 'resource_only', feasible)
        for restart in range(s['optimizer_restarts']):
            differential_search(evaluate, budget, t, p['s'], norm, bounds, s,
                                seed(config['validation']['base_seed'], 'optimizer', problem, restart), equal_relative[0], feasible)
        equal_relative_row = ctx.candidates[(pressure, 'local', equal_relative[0].id)]
        redistribution = one_dimensional(evaluate, budget, t, p['s'], norm, s['one_dimensional_search'], bounds, 'redistribution', feasible)
        local = ctx.shortlist(problem, 'local', slots)
        redistribution_row = ctx.candidates[(pressure, 'local', redistribution[0].id)]
        families = dict(local=nested_shortlist(local, [equal_relative_row, redistribution_row], slots),
                        redistribution=[ctx.candidates[(pressure, 'local', q.id)] for q in redistribution[:slots]], equal_relative=[ctx.candidates[(pressure, 'local', q.id)] for q in equal_relative[:slots]],
                        resource_only=[ctx.candidates[(pressure, 'fixed', q.id)] for q in resource_only[:slots]])
        global_problem = s.get('global_composition_reoptimization')
        if norm == 'L2' and global_problem and pressure == global_problem['pressure'] and budget == global_problem['budget']:
            settings = s | dict(observation='global')
            for restart in range(s['optimizer_restarts']):
                differential_search(evaluate, budget, t, p['s'], norm, bounds, settings,
                                    seed(config['validation']['base_seed'], 'global-optimizer', restart), equal_relative[0], feasible)
            families['global'] = ctx.shortlist(problem, 'global', slots)
        design.append(dict(problem=problem, problem_id=problem_id(problem), families=families))
    write_json(ctx.output/'shortlists.json', design)


def validate(ctx):
    design = json.loads((ctx.output/'shortlists.json').read_text())
    frozen = []
    for row in design:
        winners = {}
        for family, candidates in row['families'].items():
            if not candidates:
                raise RuntimeError(f'empty {family} shortlist')
            assessed = [ctx.aggregate(row['problem']['pressure'], Policy(**r['policy']), 'validation',
                                      ctx.config['validation']['validation_realizations']) for r in candidates]
            winners[family] = min(assessed, key=lambda r: candidate_key(r, row['problem']['norm']))
            if 'search_feasible' in winners[family]:
                winners[family]['selection_status'] = ('empirically_feasible' if winners[family]['search_feasible']
                                                       else 'no_empirically_feasible_candidate')
        frozen.append(dict(problem=row['problem'], problem_id=row['problem_id'], winners=winners))
    write_json(ctx.output/'frozen_policies.json', frozen)


def primary(frozen, config):
    target = config['search'].get('global_composition_reoptimization') or config['search']['main_problems'][0]
    return next(row for row in frozen if row['problem']['norm'] == 'L2'
                and row['problem']['pressure'] == target['pressure'] and row['problem']['budget'] == target['budget'])


def test(ctx):
    frozen = json.loads((ctx.output/'frozen_policies.json').read_text())
    count = ctx.config['validation']['test_realizations']
    results = []
    for row in frozen:
        comparisons = {family: ctx.aggregate(row['problem']['pressure'], Policy(**candidate['policy']), 'test', count, record=True)
                       for family, candidate in row['winners'].items()}
        first, second = comparisons['local'], comparisons['resource_only']
        if first['world_ids'] != second['world_ids']:
            raise ArithmeticError('paired policies do not share worlds')
        interval = paired_interval(np.array(second['losses'])-np.array(first['losses']),
                                   rng(ctx.config['validation']['base_seed'], 'analysis', row['problem_id']),
                                   confidence=ctx.config['validation'].get('confidence_level', .95))
        interval['contrast_direction'] = 'resource_only_minus_local_positive_is_improvement'
        result = dict(problem=row['problem'], policies=comparisons, local_vs_resource_only=interval)
        if 'welfares' in first:
            result['welfare_local_vs_resource_only'] = paired_interval(
                np.array(first['welfares'])-np.array(second['welfares']),
                rng(ctx.config['validation']['base_seed'], 'welfare-analysis', row['problem_id']),
                confidence=ctx.config['validation'].get('confidence_level', .95))
        results.append(result)
    claimed = {(row['problem']['pressure'], value['policy_id']): value
               for row in results for value in row['policies'].values() if 'stress_events' in value}
    if claimed:
        confidence = ctx.config['validation'].get('confidence_level', .95)
        adjusted_confidence = 1-(1-confidence)/len(claimed)
        maximum = stress_spec(ctx.config).maximum_probability
        for row in results:
            for value in row['policies'].values():
                upper = probability_upper_bound(np.array(value['stress_events'], dtype=bool), adjusted_confidence)
                value.update(simultaneous_stress_probability_upper=upper,
                             simultaneous_protection_supported=bool(upper <= maximum),
                             protected_family_size=len(claimed), family_confidence=confidence)
    anchor = primary(frozen, ctx.config)
    pressure = anchor['problem']['pressure']
    t, s = ctx.config['payoffs']['t'], ctx.config['payoffs']['s']
    winner = Policy(**anchor['winners']['local']['policy'])
    observation_controls = {obs: ctx.aggregate(pressure, replace(winner, observation=obs), 'test', count, record=True)
                            for obs in ['local', 'global', 'fixed']}
    diagnostics = {}
    existing_bounds = {(r['pressure'], r['world_id']) for r in read_jsonl(ctx.output/'capacity_bounds.jsonl')}
    for load in sorted(set(row['problem']['pressure'] for row in frozen)):
        diagnostics[str(load)] = {name: ctx.aggregate(load, policy, 'test', count, record=True) for name, policy in
                                  [('reference', Policy(t=t, s=s)), ('always_C', Policy(t=t, s=s, forced='C')),
                                   ('always_D', Policy(t=t, s=s, forced='D'))]}
        for i in range(count):
            world = ctx.world('test', i)
            if (load, world.id) in existing_bounds:
                continue
            bound = strict_safety_bound(world, scenario(ctx.config, load),
                                        ctx.config['objective']['total_steps']-ctx.config['objective']['excluded_initial_steps'],
                                        metric_spec(ctx.config).floor)
            append_json(ctx.output/'capacity_bounds.jsonl', dict(pressure=load, index=i, world_id=world.id, **bound))
    original_comparison = None
    if welfare_spec(ctx.config):
        local = next(row['policies']['local'] for row in results if row['problem'] == anchor['problem'])
        reference = diagnostics[str(pressure)]['reference']
        if local['world_ids'] != reference['world_ids']:
            raise ArithmeticError('original-game comparison requires paired worlds')
        original_comparison = {}
        confidence = 1-(1-ctx.config['validation'].get('confidence_level', .95))/2
        for outcome in ['human', 'ai']:
            original_comparison[outcome] = paired_interval(
                np.array(local[f'{outcome}_services'])-np.array(reference[f'{outcome}_services']),
                rng(ctx.config['validation']['base_seed'], 'original-comparison', outcome), confidence=confidence)
        original_comparison['both_improve_supported'] = bool(
            all(original_comparison[name]['lower'] > 0 for name in ['human', 'ai'])
            and local['simultaneous_protection_supported'])
    write_json(ctx.output/'test_comparisons.json', dict(problems=results, same_matrix_observation_controls=observation_controls,
                                                      diagnostics=diagnostics, primary_problem=anchor['problem'],
                                                      original_game_outcome_comparison=original_comparison))


def robustness(ctx):
    frozen = json.loads((ctx.output/'frozen_policies.json').read_text())
    anchor = primary(frozen, ctx.config)
    policies = {family: Policy(**candidate['policy']) for family, candidate in anchor['winners'].items()}
    policies['reference'] = Policy(t=ctx.config['payoffs']['t'], s=ctx.config['payoffs']['s'])
    count = ctx.config['validation'].get('robustness_realizations', ctx.config['validation']['validation_realizations'])
    rows = []
    for variant in robustness_variants(ctx.config):
        comparisons = {name: ctx.aggregate(anchor['problem']['pressure'], policy, 'robustness', count, variant=variant, record=True)
                       for name, policy in policies.items()}
        rows.append(dict(variant=variant, policies=comparisons,
                         interpretation='Transfer of frozen policies; changed-game results are not reoptimized frontiers.'))
    write_json(ctx.output/'robustness.json', rows)


def run_stages(config, output, stages, maximum_steps):
    output = Path(output)
    expanded = STAGES if stages == 'all' else (['characterize', 'train'] if stages == 'train' else [stages])
    plan = budget_plan(config)
    manifest_path = output/'manifest.json'
    sources, config_id = source_hashes(), identifier(config)
    if manifest_path.exists():
        from .analysis import verify_bundle
        if (output/'checksums.json').exists():
            verify_bundle(output)
        manifest = json.loads(manifest_path.read_text())
        if manifest['configuration_id'] != config_id or manifest['source_sha256'] != sources:
            raise ValueError('configuration or source changed; start a new output bundle')
        if manifest['environment']['dependencies'] != environment()['dependencies']:
            raise ValueError('dependency versions changed; start a new output bundle')
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('output folder must be empty or contain a matching bundle')
        manifest = dict(run_id=identifier([config_id, sources])[:24], created_utc=utc_now(),
                        configuration=config, configuration_id=config_id, source_sha256=sources,
                        environment=environment(), stages={})
    pending = [s for s in expanded if manifest['stages'].get(s) != 'complete']
    for stage in pending:
        index = STAGES.index(stage)
        required = STAGES[index-1] if stage in ['validate', 'test'] else ('validate' if stage == 'robustness' else None)
        if required and required not in pending and manifest['stages'].get(required) != 'complete':
            raise ValueError(f'{stage} requires completed {required}')
    upper_bound = sum(plan['model_step_caps'][stage] for stage in pending)
    if upper_bound > maximum_steps:
        raise ValueError(f'planned upper bound {upper_bound:,} model steps exceeds limit {maximum_steps:,}; no simulations started')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'resolved_config.json', config)
    write_json(output/'budget_plan.json', plan)
    if not (output/'source_snapshot').exists():
        for relative, source_path in source_files().items():
            target = output/'source_snapshot'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target)
    write_json(manifest_path, manifest)
    ctx = Evaluation(config, output, maximum_steps)
    try:
        for stage in pending:
            manifest['stages'][stage] = 'running'
            write_json(manifest_path, manifest)
            globals()[stage](ctx)
            manifest['stages'][stage] = 'complete'
            manifest['model_steps_logged'] = sum(r['steps'] for r in read_jsonl(output/'budget_ledger.jsonl'))
            write_json(manifest_path, manifest)
    except Exception as error:
        manifest['last_error'] = str(error)
        manifest['model_steps_logged'] = sum(r['steps'] for r in read_jsonl(output/'budget_ledger.jsonl'))
        write_json(manifest_path, manifest)
        write_json(output/'checksums.json', bundle_hashes(output))
        raise
    manifest.pop('last_error', None)
    manifest['updated_utc'] = utc_now()
    write_json(manifest_path, manifest)
    write_json(output/'checksums.json', bundle_hashes(output))
    return manifest


def replay(output, evaluation_id):
    output = Path(output)
    rows = {r['evaluation_id']: r for r in read_jsonl(output/'realizations.jsonl')}
    row = rows[evaluation_id]
    manifest = json.loads((output/'manifest.json').read_text())
    if source_hashes() != manifest['source_sha256']:
        raise ValueError('replay requires the recorded source snapshot and dependencies')
    if environment()['dependencies'] != manifest['environment']['dependencies']:
        raise ValueError('replay requires the recorded dependency versions')
    world = World.load(output/row['world_path'])
    driver = ArchivedDrivers(output/row['drivers_path']) if 'drivers_path' in row else None
    summary, data = simulate(world, Scenario(**row['scenario']), Policy(**row['policy']), row['driver_seed'],
                             row['steps'], row['burn'], MetricSpec(**row['metric']), record=True, drivers=driver,
                             welfare=WelfareSpec(**row['welfare_spec']) if row.get('welfare_spec') else None,
                             stress=StressSpec(**row['stress_spec']) if row.get('stress_spec') else None)
    if 'trajectory_path' in row:
        with np.load(output/row['trajectory_path'], allow_pickle=False) as archived:
            for key in ['stocks', 'strategies', 'observations', 'agent_requested', 'agent_delivered',
                        'agent_reference_request', 'agent_is_ai', 'agent_targets', 'resource_renewal', 'source_withdrawal']:
                np.testing.assert_array_equal(data[key], archived[key])
    for key in ['loss', 'mean_requested', 'mean_delivered', 'longest_continuous_depletion_steps',
                'cell_longest_continuous_depletion_steps', 'cell_depleted_fraction', 'minimum_accessible_resource',
                'agent_reference_request', 'agent_mean_requested', 'agent_mean_delivered', 'agent_unserved_fraction']:
        if summary[key] != row[key]:
            raise ArithmeticError(f'replay mismatch: {key}')
    for key in ['welfare', 'human_service', 'ai_service', 'agent_mean_service', 'agent_service_target',
                'persistent_local_stress', 'persistent_cell_fraction', 'persistent_site_fraction', 'stress_basis',
                'stress_duration_steps', 'site_longest_continuous_stress_steps', 'cell_longest_continuous_stress_steps']:
        if key in row and summary[key] != row[key]:
            raise ArithmeticError(f'replay mismatch: {key}')
    return dict(evaluation_id=evaluation_id, matched=True, archived_trajectory='trajectory_path' in row,
                simulated_steps=row['steps'])
