import copy
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from human_ai_commons import paper_experiments
from human_ai_commons.dynamics import Drivers, RenewalControl, Scenario, Simulator, multiplier, simulate
from human_ai_commons.paper_experiments import Disturbance, State, local_load, load_classes, run_segment
from human_ai_commons.payoffs import Policy
from human_ai_commons.world import WorldSpec, generate_world

SPEC = WorldSpec(width=8, height=8, humans=8, ai=8, density=.3, capacity=6.)
SCENARIO = Scenario(pressure=1, allocation='proportional')
WINDOWS = [('early', 0, 20), ('late', 40, 60)]
CONFIGS = Path(__file__).parents[1]/'configs'


def tiny_config():
    config = copy.deepcopy(paper_experiments.load_protocol(CONFIGS/'pilot.json'))
    config['worlds'] = 2
    config['world'].update(width=8, height=8, humans=8, ai=8, capacity=6)
    config['continuity'].update(steps=60, windows=[['conference_final', 10, 20], ['late', 40, 60]], capacities=[3, 6], densities=[.5])
    config['interventions'].update(develop_steps=30, developed_window=10, after_steps=30, windows_after=[[0, 10], [10, 30]],
                                   designs=['threshold_up', 'offset_defectors', 'always_ordinary'])
    config['feedback'].update(hysteresis_steps=60, hysteresis_windows=[['a', 20, 40], ['b', 40, 60]], hysteresis_designs=['threshold_up'],
                              develop_steps=30, developed_window=10, after_steps=30, windows_after=[[0, 10], [10, 30]],
                              control_designs=['threshold_down'], recovery_designs=['always_ordinary'])
    config['feedback']['disturbances']['drought']['duration'] = 10
    config['cells'] = {'strong': {'scenario': {'theta': 10}}, 'conference': {'requests': 'fixed'}}
    config['policies'].update(unchanged_no_reversals={'r_c': 0, 'r_d': 0, 'epsilon': 0},
                              very_high_threshold={'r_c': 18, 'r_d': 18, 'diagnostic': True},
                              consequence_scale={'r_c': 0, 'r_d': 0, 'consequence_weight': 1})
    config['interventions'].update(cells=['reference', 'conference'], from_initialization_cells=['reference'],
                                   designs=['threshold_up', 'offset_defectors', 'always_ordinary', 'unchanged_no_reversals',
                                            'very_high_threshold', 'consequence_scale'])
    config['feedback'].update(cells=['reference', 'strong'], hysteresis_designs=['threshold_up', 'consequence_scale'])
    config['feedback']['part_cells'] = {'controls': ['strong'], 'recovery': ['reference']}
    config['confirmatory'] = {'alpha': .05, 'smallest_difference_of_interest': .01, 'contrasts': [
        {'name': 'capacity', 'experiment': 'continuity', 'outcome': 'cooperation',
         'select': {'condition': 'storage_journal_K6_full', 'window': 'conference_final'},
         'reference': {'condition': 'storage_journal_K3_full', 'window': 'conference_final'}},
        {'name': 'scale', 'experiment': 'interventions', 'cell': 'conference', 'outcome': 'ai_activity',
         'select': {'condition': 'consequence_scale', 'timing': 'after_development', 'window': '10-30'},
         'reference': {'condition': 'unchanged', 'timing': 'after_development', 'window': '10-30'}},
        {'name': 'start', 'experiment': 'feedback', 'cell': 'strong', 'outcome': 'contested_stock',
         'select': {'condition': 'unchanged|coupled|empty', 'window': 'b'},
         'reference': {'condition': 'unchanged|coupled|full', 'window': 'b'}}]}
    return config


def test_default_path_is_unchanged_and_identical_requests_make_no_contrast():
    world = generate_world(SPEC, 3)
    usual = Simulator(world, SCENARIO, Policy())
    same = Simulator(world, SCENARIO, Policy(), cooperative_request=usual.cooperative_request.copy(), renewal_control=RenewalControl())
    drivers = Drivers(11, len(world.positions))
    for t in range(40):
        first, second = usual.step(*drivers.step(t)), same.step(*drivers.step(t))
        assert np.array_equal(first['delivered'], second['delivered'])
    assert np.array_equal(usual.water, same.water) and np.array_equal(usual.strategies, same.strategies)
    # the segment runner reproduces the staged simulator's recorded history.
    _, data = simulate(world, SCENARIO, Policy(), 11, 40, 0, record=True)
    final, _, _, info = run_segment(world, SCENARIO, Policy(), 11, 0, 40)
    assert np.array_equal(data['stocks'][-1], final.water) and np.array_equal(data['strategies'][-1], final.strategies)
    assert np.allclose(info['series'][:, 2], data['observations'][:, 2])


@pytest.mark.parametrize('design', [Policy(), Policy.from_relative(1, 0), Policy(forced='C')])
def test_unchanged_continuation_reproduces_uninterrupted_history(design):
    world = generate_world(SPEC, 5)
    whole, _, _, whole_info = run_segment(world, SCENARIO, design, 7, 0, 60)
    middle, _, _, _ = run_segment(world, SCENARIO, design, 7, 0, 25)
    assert middle.step == 25
    resumed, rows, _, info = run_segment(world, SCENARIO, design, 7, 25, 60, state=middle, windows=[('late', 40, 60)])
    assert np.array_equal(whole.water, resumed.water) and np.array_equal(whole.strategies, resumed.strategies)
    assert np.array_equal(whole_info['series'][25:], info['series'], equal_nan=True)
    # the window covers only the continued segment, measured at absolute steps.
    assert rows[0]['window_start'] == 40 and np.isclose(rows[0]['stock'], whole_info['series'][40:60, 3].mean())


def test_continuation_uses_the_absolute_random_driver_step():
    world = generate_world(SPEC, 5)
    middle, _, _, _ = run_segment(world, SCENARIO, Policy(), 7, 0, 25)
    changed = Policy.from_relative(1, 1)
    resumed, _, _, _ = run_segment(world, SCENARIO, changed, 7, 25, 45, state=middle)
    manual = Simulator(world, SCENARIO, changed)
    manual.restore(middle.water, middle.strategies)
    drivers = Drivers(7, len(world.positions))
    for t in range(25, 45):
        manual.step(*drivers.step(t))
    assert np.array_equal(manual.water, resumed.water) and np.array_equal(manual.strategies, resumed.strategies)
    shifted = Simulator(world, SCENARIO, changed)
    shifted.restore(middle.water, middle.strategies)
    for t in range(20):
        shifted.step(*drivers.step(t))
    assert not np.array_equal(shifted.strategies, resumed.strategies) or not np.array_equal(shifted.water, resumed.water)


def test_continuation_rejects_missing_or_mismatched_state():
    world = generate_world(SPEC, 5)
    state, _, _, _ = run_segment(world, SCENARIO, Policy(), 7, 0, 10)
    with pytest.raises(ValueError):
        run_segment(world, SCENARIO, Policy(), 7, 10, 20)
    with pytest.raises(ValueError):
        run_segment(world, SCENARIO, Policy(), 7, 12, 20, state=state)
    with pytest.raises(ValueError):
        run_segment(world, SCENARIO, Policy(), 7, 10, 20, state=State(10, state.water*10, state.strategies))
    with pytest.raises(ValueError):
        run_segment(world, SCENARIO, Policy(), 7, 0, 20, windows=[('outside', 10, 30)])


def test_water_balance_and_bounds_under_controls_and_disturbances():
    world = generate_world(SPEC, 9)
    events = (Disturbance('renewal_scale', 10, 30, .25), Disturbance('stock_scale', 35, 36, 0.))
    for control in [None, RenewalControl('baseline'), RenewalControl('fixed_cooperation', .4)]:
        final, rows, arrays, info = run_segment(world, SCENARIO, Policy(), 13, 0, 60, renewal_control=control,
                                                disturbances=events, windows=WINDOWS)
        assert info['maximum_balance_residual'] < 1e-9
        assert np.all(final.water >= 0) and np.all(final.water <= world.capacity)
        assert np.all(info['series'][:, 7] >= -1e-12)
    # the stock removal empties every source immediately before step 35.
    before, _, _, _ = run_segment(world, SCENARIO, Policy(), 13, 0, 35)
    simulator = Simulator(world, SCENARIO, Policy())
    simulator.restore(before.water*0, before.strategies)
    event = simulator.step(*Drivers(13, len(world.positions)).step(35))
    assert event['delivered'].sum() <= event['resource_renewal'].sum()+1e-12


def test_renewal_controls_remove_exactly_the_stated_dependence():
    world = generate_world(SPEC, 9)
    half = world.capacity/2
    drivers = Drivers(1, len(world.positions)).step(0)
    factors = {}
    for control in [None, RenewalControl('fixed_cooperation', .4), RenewalControl('baseline')]:
        for action in ['C', 'D']:
            simulator = Simulator(world, SCENARIO, Policy(forced=action), renewal_control=control)
            simulator.restore(half, np.full(len(world.kinds), action == 'C'))
            factors[(control.mode if control else 'coupled', action)] = simulator.step(*drivers)['renewal_multiplier']
    assert not np.array_equal(factors[('coupled', 'C')], factors[('coupled', 'D')])
    assert np.array_equal(factors[('fixed_cooperation', 'C')], factors[('fixed_cooperation', 'D')])
    assert np.allclose(factors[('fixed_cooperation', 'C')], multiplier(.5, .4, SCENARIO.theta, SCENARIO.feedback))
    assert np.all(factors[('baseline', 'C')] == 1) and np.all(factors[('baseline', 'D')] == 1)
    with pytest.raises(ValueError):
        RenewalControl('fixed_cooperation')
    with pytest.raises(ValueError):
        RenewalControl('baseline', .5)


def test_fixed_requests_and_local_load():
    world = generate_world(SPEC, 4)
    requests = np.where(world.kinds, 2., .1)
    simulator = Simulator(world, replace(SCENARIO, pressure=7.), Policy(forced='D'), cooperative_request=requests)
    event = simulator.step(*Drivers(1, len(world.positions)).step(0))
    assert np.allclose(event['requested'], 1.5*requests)
    load, classes = load_classes(world, requests, SCENARIO)
    reaching = world.access.any(axis=1)
    assert np.isclose((load*world.renewal).sum(), requests[reaching].sum())
    assert np.array_equal(load, local_load(world, requests))
    assert np.all(classes[load*1.5 <= 1] == 0) and np.all(classes[load > 1.25] == 2)
    assert paper_experiments.maximum_multiplier(SCENARIO) == 1.25
    with pytest.raises(ValueError):
        Simulator(world, SCENARIO, Policy(), cooperative_request=requests[:-1])


def test_capacity_changes_keep_the_landscape_and_inputs():
    config = tiny_config()
    small, large = paper_experiments.make_world(config, 0, capacity=3.), paper_experiments.make_world(config, 0, capacity=12.)
    for name in ['resources', 'renewal', 'positions', 'kinds', 'initial']:
        assert np.array_equal(getattr(small, name), getattr(large, name))
    assert np.all(large.capacity == 12) and paper_experiments.initial_state(large, 'equal_volume', 3).water.sum() == small.capacity.sum()
    assert paper_experiments.trajectory_seed(config, 0) != paper_experiments.trajectory_seed(config, 1)


def test_search_audit_reads_a_saved_bundle_without_simulating(tmp_path, monkeypatch):
    from human_ai_commons.config import load_config
    from human_ai_commons.experiments import run_stages
    from human_ai_commons.paper_tables import audit_search
    config = load_config(Path(__file__).parents[1]/'configs/smoke.json')
    run_stages(config, tmp_path/'smoke', 'all', 10000)
    monkeypatch.setattr(Simulator, 'step', lambda *a, **k: (_ for _ in ()).throw(AssertionError('the audit must not simulate')))
    made = audit_search(tmp_path/'smoke', tmp_path/'audit', worlds=2)
    assert {'search_score_ranges.csv', 'forced_action_scores.csv', 'source_states_by_local_load.csv', 'stock_event_geometry.csv',
            'search_audit.json', 'source_stock_by_local_load.pdf'} <= set(made['files'])
    for row in made['stock_event_geometry']:
        assert row['worlds'] == 2 and 0 <= row['overloaded_sources'] <= row['accessible_sources']
        assert row['held_and_overloaded']+row['held_and_contested'] <= row['sources_held_below_reference']+1e-12
    audit = json.loads((tmp_path/'audit/search_audit.json').read_text())
    assert audit['bundle_run_id'] == json.loads((tmp_path/'smoke/manifest.json').read_text())['run_id']


def test_protocol_runs_plans_exactly_and_refuses_changes(tmp_path, monkeypatch):
    config = tiny_config()
    calls = dict(count=0)
    original = Simulator.step

    def counted(self, *args):
        calls['count'] += 1
        return original(self, *args)

    monkeypatch.setattr(Simulator, 'step', counted)
    plan = paper_experiments.planned_steps(config, list(paper_experiments.EXPERIMENTS))
    with pytest.raises(ValueError):
        paper_experiments.run_protocol(config, tmp_path/'run', 'all', sum(plan.values())-1)
    result = paper_experiments.run_protocol(config, tmp_path/'run', 'all', sum(plan.values()))
    assert calls['count'] == sum(plan.values()) and result['completed'] == 10
    rows = [json.loads(line) for line in (tmp_path/'run/rows.jsonl').read_text().splitlines()]
    checked = [r for r in rows if r.get('continuation_check')]
    assert checked and all(r['continuation_check'] == 'identical_to_uninterrupted' for r in checked)
    assert max(r['maximum_balance_residual'] for r in rows if r['timing' if r['experiment'] == 'interventions' else 'experiment'] != 'development') < 1e-9
    rules = {r['renewal_rule'] for r in rows if r['experiment'] == 'feedback'}
    assert (tmp_path/'run/arrays/feedback-strong-0000.npz').exists()
    assert rules == {'coupled', 'fixed_cooperation', 'baseline_only'} and all(isinstance(r['renewal'], float) for r in rows)
    fixed = [r for r in rows if r['condition'] == 'proportional_allocation']
    scaled = [r for r in rows if r['condition'] == 'journal_reference']
    assert fixed[0]['total_ordinary_request'] == 8*.1+8*2. and np.isclose(scaled[0]['relative_load'], 1)
    assert {r['world_index'] for r in rows} == {0, 1}
    from human_ai_commons.paper_tables import report
    made = report(tmp_path/'run')
    assert {'continuity_conditions.csv', 'intervention_effects.csv', 'intervention_distribution.csv', 'feedback_start_dependence.csv',
            'feedback_recovery.csv', 'confirmatory_contrasts.csv'} <= set(made['tables'])
    # the report writes tables only; figures are drawn by the scripts under figures/.
    assert not list((tmp_path/'run/report').glob('*.pdf')) and not list((tmp_path/'run/report').glob('*.png'))
    import csv
    contrasts = list(csv.DictReader(open(tmp_path/'run/report/confirmatory_contrasts.csv')))
    assert [c['name'] for c in contrasts] == ['capacity', 'scale', 'start'] and all(c['declared_contrasts'] == '3' for c in contrasts)
    assert all(float(c['lower']) <= float(c['difference']) <= float(c['upper']) for c in contrasts)
    assert all(float(c['bounded_lower']) <= float(c['difference']) <= float(c['bounded_upper']) for c in contrasts)
    effects = list(csv.DictReader(open(tmp_path/'run/report/intervention_effects.csv')))
    assert {e['cell'] for e in effects} == {'reference', 'conference'}
    assert {e['timing'] for e in effects if e['cell'] == 'conference'} == {'after_development'}
    flags = {e['design']: e['admissible'] for e in effects}
    assert flags['threshold_up'] == 'True' and flags['very_high_threshold'] == 'False' and flags['unchanged_no_reversals'] == 'False'
    cells = {(r['experiment'], r['cell']): r for r in rows}
    assert cells[('feedback', 'strong')]['theta'] == 10 and cells[('interventions', 'conference')]['total_ordinary_request'] == 8*.1+8*2.
    # completed work is reused and a changed configuration is refused.
    again = paper_experiments.run_protocol(config, tmp_path/'run', 'all', sum(plan.values()))
    assert calls['count'] == sum(plan.values()) and again['rows'] == result['rows']
    config['worlds'] = 3
    with pytest.raises(ValueError):
        paper_experiments.run_protocol(config, tmp_path/'run', 'all', 10**9)


def test_consequence_weight_changes_only_which_stock_the_payoff_reads():
    world = generate_world(SPEC, 9)
    drivers = Drivers(3, len(world.positions))
    usual, zero, scaled = (Simulator(world, SCENARIO, Policy(), consequence_weight=w) for w in [0., 0., 1.])
    water = np.where(np.arange(len(world.capacity)) % 2 == 0, world.capacity, 0.)
    for simulator in [usual, zero, scaled]:
        simulator.restore(water, world.initial)
    first, same, other = (simulator.step(*drivers.step(0)) for simulator in [usual, zero, scaled])
    assert np.array_equal(first['delivered'], same['delivered']) and np.array_equal(usual.strategies, zero.strategies)
    # renewal, requests per action, access and source choice are untouched; only the chosen actions can differ.
    assert np.array_equal(first['resource_renewal'], other['resource_renewal']) and np.array_equal(first['targets'], other['targets'])
    assert not np.array_equal(usual.strategies, scaled.strategies)
    assert abs(other['balance_residual']) < 1e-9
    with pytest.raises(ValueError):
        Simulator(world, SCENARIO, Policy(), consequence_weight=1.5)


def test_source_selection_prefers_fuller_sources_and_defaults_to_uniform():
    world = generate_world(SPEC, 9)
    drivers = Drivers(3, len(world.positions))
    water = np.where(np.arange(len(world.capacity)) % 2 == 0, world.capacity, .05*world.capacity)
    made = {name: Simulator(world, replace(SCENARIO, selection=name), Policy()) for name in ['uniform', 'stock_weighted', 'fullest']}
    usual = Simulator(world, SCENARIO, Policy())
    for simulator in [usual, *made.values()]:
        simulator.restore(water, world.initial)
    reference = usual.step(*drivers.step(0))
    events = {name: simulator.step(*drivers.step(0)) for name, simulator in made.items()}
    assert np.array_equal(reference['targets'], events['uniform']['targets']) and np.array_equal(reference['delivered'], events['uniform']['delivered'])
    level = (water+reference['resource_renewal'])/world.capacity
    chosen = reference['targets'] >= 0
    for i in np.flatnonzero(chosen):
        assert level[events['fullest']['targets'][i]] == level[world.access[i]].max()
    # the same random draw moves toward fuller sources, never away from them.
    assert level[events['stock_weighted']['targets'][chosen]].mean() > level[reference['targets'][chosen]].mean()
    assert all(abs(event['balance_residual']) < 1e-9 for event in events.values())
    with pytest.raises(ValueError):
        replace(SCENARIO, selection='nearest')


def test_graded_response_follows_larger_payoff_differences_more_reliably():
    world = generate_world(SPEC, 9)
    drivers = Drivers(3, len(world.positions))
    cooperation = {}
    for temperature in [.02, 5.]:
        simulator = Simulator(world, replace(SCENARIO, response='logit', temperature=temperature), Policy())
        # nearly empty sources make the ordinary request clearly preferable.
        simulator.restore(.02*world.capacity, world.initial)
        simulator.step(*drivers.step(0))
        cooperation[temperature] = simulator.strategies[world.access.any(axis=1)].mean()
    assert cooperation[.02] == 1 and .3 < cooperation[5.] < 1
    with pytest.raises(ValueError):
        replace(SCENARIO, response='logit', temperature=0)


def test_consequence_scope_reads_reachable_sources_only():
    world = generate_world(SPEC, 9)
    drivers = Drivers(3, len(world.positions))
    reach = world.access.any(axis=0)
    # reachable sources are empty and unreachable ones full, so only the wider scope sees abundance.
    water = np.where(reach, 0., world.capacity)
    wide, near = (Simulator(world, replace(SCENARIO, epsilon=0), Policy(), consequence_weight=1., consequence_scope=scope) for scope in ['feedback', 'access'])
    for simulator in [wide, near]:
        simulator.restore(water, world.initial)
        simulator.step(*drivers.step(0))
    assert near.strategies[world.access.any(axis=1)].all() and near.strategies.sum() >= wide.strategies.sum()
    with pytest.raises(ValueError):
        Simulator(world, SCENARIO, Policy(), consequence_scope='global')


def test_window_measurements_agree_with_the_staged_summary():
    from human_ai_commons.metrics import WelfareSpec
    world = generate_world(SPEC, 3)
    summary, _ = simulate(world, SCENARIO, Policy(), 11, 60, 20, welfare=WelfareSpec())
    _, rows, _, _ = run_segment(world, SCENARIO, Policy(), 11, 0, 60, windows=[('scored', 20, 60)])
    assert np.isclose(rows[0]['human_activity'], summary['human_service']) and np.isclose(rows[0]['ai_activity'], summary['ai_service'])
    assert np.isclose(rows[0]['stock'], summary['mean_accessible_resource']) and np.isclose(rows[0]['cooperation'], summary['mean_cooperation'])
    assert np.isclose(rows[0]['stock_all_sources'], summary['mean_all_resource'])


def test_worker_count_does_not_change_saved_results(tmp_path):
    config = tiny_config()
    config['interventions'].update(cells=['reference'], designs=['threshold_up'])
    config['worlds'] = 3
    limit = sum(paper_experiments.planned_steps(config, ['interventions']).values())
    paper_experiments.run_protocol(config, tmp_path/'one', 'interventions', limit, workers=1)
    paper_experiments.run_protocol(config, tmp_path/'three', 'interventions', limit, workers=3)
    assert (tmp_path/'one/rows.jsonl').read_text() == (tmp_path/'three/rows.jsonl').read_text()


def test_delivery_ceiling_bounds_every_sustained_delivery():
    world = generate_world(SPEC, 4)
    simulator = Simulator(world, SCENARIO, Policy())
    target = simulator.cooperative_request*SCENARIO.defect_ratio
    reach = world.access.any(axis=1)
    ceiling = paper_experiments.delivery_ceiling(world, target)
    assert 0 < ceiling <= min(target[reach].sum(), world.renewal[world.access.any(axis=0)].sum())+1e-9
    assert paper_experiments.delivery_ceiling(world, target, 1.25) >= ceiling
    assert np.isclose(paper_experiments.delivery_ceiling(world, target*1e-6), target[reach].sum()*1e-6)
    # with baseline renewal only, no rule can deliver more than the ceiling once initial stocks are gone.
    _, rows, _, _ = run_segment(world, SCENARIO, Policy(forced='D'), 5, 0, 1200, state=paper_experiments.initial_state(world, 'empty'),
                                renewal_control=RenewalControl('baseline'), windows=[('late', 200, 1200)])
    assert rows[0]['human_delivered']+rows[0]['ai_delivered'] <= ceiling+1e-9


def test_resume_discards_rows_of_a_job_that_was_not_completed(tmp_path):
    config = tiny_config()
    config['interventions'].update(cells=['reference'], designs=['threshold_up'])
    limit = sum(paper_experiments.planned_steps(config, ['interventions']).values())
    paper_experiments.run_protocol(config, tmp_path/'run', 'interventions', limit)
    complete = (tmp_path/'run/rows.jsonl').read_text()
    manifest = json.loads((tmp_path/'run/manifest.json').read_text())
    # as if the run had stopped after writing the second world's rows but before recording it as done.
    manifest['completed'] = [job for job in manifest['completed'] if job[2] == 0]
    (tmp_path/'run/manifest.json').write_text(json.dumps(manifest))
    paper_experiments.run_protocol(config, tmp_path/'run', 'interventions', limit)
    assert (tmp_path/'run/rows.jsonl').read_text() == complete


def test_running_everything_writes_reports_and_resumes(tmp_path):
    first, second = tiny_config(), tiny_config()
    first['interventions'].update(cells=['reference'], designs=['threshold_up'])
    first['feedback'].update(cells=['reference'], part_cells={})
    first['confirmatory']['contrasts'] = first['confirmatory']['contrasts'][:1]
    second['protocol'] += '-long'
    second['experiments'] = ['feedback']
    second['feedback'].update(cells=['strong'], parts=['start_dependence'], hysteresis_designs=[], part_cells={})
    second['confirmatory']['contrasts'] = second['confirmatory']['contrasts'][2:]
    made = paper_experiments.run_all([('main', first), ('long', second)], tmp_path/'results', workers=2)
    assert [Path(m['output']).name for m in made] == ['main', 'long']
    assert {json.loads(l)['experiment'] for l in (tmp_path/'results/long/rows.jsonl').read_text().splitlines()} == {'feedback'}
    assert all('confirmatory_contrasts.csv' in m['tables'] for m in made)
    rows = (tmp_path/'results/main/rows.jsonl').read_text()
    again = paper_experiments.run_all([('main', first), ('long', second)], tmp_path/'results', workers=2)
    assert (tmp_path/'results/main/rows.jsonl').read_text() == rows and [m['jobs'] for m in again] == [m['jobs'] for m in made]


def test_paper_configurations_extend_the_study_and_share_its_world():
    from dataclasses import asdict
    from human_ai_commons.config import load_config, scenario, world_spec
    study = load_config(CONFIGS/'research_protocol.json')
    names = sorted(path.stem for path in CONFIGS.glob('*.json') if 'paper' in json.loads(path.read_text()))
    assert set(paper_experiments.PAPER_RUNS) <= set(names)
    for name in names:
        written = json.loads((CONFIGS/f'{name}.json').read_text())
        assert set(written) == {'extends', 'paper'}
        config = paper_experiments.load_protocol(CONFIGS/f'{name}.json')
        assert config['world'] == asdict(world_spec(study)) and config['base_seed'] == study['validation']['base_seed']
        expected = asdict(scenario(study, config['reference_load'])) | config.get('reference_scenario', {})
        assert config['scenario'] == expected and config['split'].startswith('paper')
    splits = {name: paper_experiments.load_protocol(CONFIGS/f'{name}.json') for name in paper_experiments.PAPER_RUNS}
    search = splits['payoff_grid']
    assert len({search['split'], search['search']['test_split'], splits['original_game']['split']}) == 3


def test_measures_under_changed_assumptions_pair_each_placement_on_the_worlds_of_the_local_measures():
    config = paper_experiments.load_protocol(CONFIGS/'local_measures_assumptions.json')
    measures = paper_experiments.load_protocol(CONFIGS/'local_measures.json')
    assert config['split'] == measures['split'] and config['worlds'] == measures['worlds']
    cells = config['interventions']['cells']
    random = [cell for cell in cells if config['cells'][cell].get('siting', 'random') == 'random']
    # every changed assumption is run with random and with sited placement, which differ in nothing else.
    for cell in random:
        sited = config['cells']['sited_'+cell]
        assert sited == config['cells'][cell] | {'siting': 'sited'} and 'sited_'+cell in cells
    assert len(cells) == 2*len(random)
    declared = {(c['cell'], c['select']['condition']) for c in config['confirmatory']['contrasts']}
    assert declared == {(cell, design) for cell in random for design in config['interventions']['designs']} | {('sited_'+cell, 'unchanged') for cell in random}


def test_selection_keeps_feasible_designs_that_no_other_design_beats():
    def row(name, values, meets=True, cost=.5):
        return dict(design=name, objectives=values, meets_requirement=meets, cost_l2=cost, r_c=0., r_d=0.)
    original = row(paper_experiments.grid_name(0., 0.), [1., 1., 0.], cost=0.)
    table = [original, row('more_stock', [.9, .9, 1.]), row('beaten', [.8, .8, .5]), row('fails_requirement', [2., 2., 2.], meets=False),
             row('too_costly', [2., 2., 2.], cost=2.), row('middle', [.95, .95, .6])]
    front = {r['design'] for r in paper_experiments.nondominated(table, 1.)}
    assert front == {original['design'], 'more_stock', 'middle'}
    assert [r['design'] for r in paper_experiments.select_designs(table, 1., 8)] == ['more_stock', 'middle']
    # with one place, the best design on the first objective is taken.
    assert [r['design'] for r in paper_experiments.select_designs(table, 1., 1)] == ['middle']
    assert {r['design'] for r in paper_experiments.nondominated(table, 3.)} == {'too_costly'}


def test_search_maps_the_grid_then_freezes_and_tests_its_selection(tmp_path, monkeypatch):
    config = tiny_config()
    config.update(experiments=['search', 'search_test'], split='paper-check')
    config['confirmatory']['contrasts'] = []
    config['search'] = dict(cells=['reference'], develop_steps=30, after_steps=30, windows_after=[[0, 30], [0, 10], [10, 30]], points_per_axis=2,
                            objectives=[dict(outcome='human_activity', window='0-30'), dict(outcome='ai_activity', window='0-30'),
                                        dict(outcome='contested_stock', window='10-30')],
                            requirement=dict(outcome='contested_empty_share', window='10-30', allowance=0.), budgets=[1., 3.],
                            selection_budget=3., selected_per_cell=3, test_split='paper-check-test', test_worlds=3)
    calls = dict(count=0)
    original = Simulator.step

    def counted(self, *args):
        calls['count'] += 1
        return original(self, *args)

    monkeypatch.setattr(Simulator, 'step', counted)
    plan = paper_experiments.planned_steps(config, ['search', 'search_test'])
    assert len(paper_experiments.payoff_grid(config)) == 9 and plan['search'] == 2*(30+30*11)
    with pytest.raises(ValueError):
        paper_experiments.run_protocol(config, tmp_path/'early', 'search_test', 10**9)
    made = paper_experiments.run_all([('search', config)], tmp_path, workers=1)
    assert plan['search'] < calls['count'] <= sum(plan.values())
    frozen = (tmp_path/'search/selected_designs.json').read_text()
    selected = json.loads(frozen)['reference']
    assert 1 <= len(selected) <= 3 and all(d['design'] != paper_experiments.grid_name(0., 0.) and d['cost_l2'] <= 3 for d in selected)
    rows = [json.loads(line) for line in (tmp_path/'search/rows.jsonl').read_text().splitlines()]
    searched = [r for r in rows if r['experiment'] == 'search']
    tested = [r for r in rows if r['experiment'] == 'search_test']
    assert len(searched) == 2*11*3 and len(tested) == 3*(1+len(selected))*3
    assert not {r['world_id'] for r in searched} & {r['world_id'] for r in tested}
    assert {r['condition'] for r in tested} == {paper_experiments.grid_name(0., 0.), *[d['design'] for d in selected]}
    assert {'search_map.csv', 'search_test_effects.csv'} <= set(made[0]['tables'])
    # a repeated command reuses the saved work and never reselects.
    steps = calls['count']
    paper_experiments.run_all([('search', config)], tmp_path, workers=1)
    assert calls['count'] == steps and (tmp_path/'search/selected_designs.json').read_text() == frozen


def test_scarcity_price_moves_draws_to_fuller_sources_and_zero_price_is_random_choice():
    world = generate_world(SPEC, 9)
    drivers = Drivers(3, len(world.positions))
    water = np.where(np.arange(len(world.capacity)) % 2 == 0, world.capacity, .05*world.capacity)
    made = {price: Simulator(world, replace(SCENARIO, selection='priced', price=price), Policy()) for price in [0., 2., 8., 200.]}
    usual, fullest = Simulator(world, SCENARIO, Policy()), Simulator(world, replace(SCENARIO, selection='fullest'), Policy())
    for simulator in [usual, fullest, *made.values()]:
        simulator.restore(water, world.initial)
    reference, top = usual.step(*drivers.step(0)), fullest.step(*drivers.step(0))
    events = {price: simulator.step(*drivers.step(0)) for price, simulator in made.items()}
    assert np.array_equal(reference['targets'], events[0.]['targets']) and np.array_equal(reference['delivered'], events[0.]['delivered'])
    chosen = reference['targets'] >= 0
    level = [reference['source_level'][event['targets'][chosen]].mean() for event in [reference, events[2.], events[8.]]]
    assert level[0] < level[1] < level[2]
    # a very large price leaves only the fullest reachable sources.
    assert np.allclose(reference['source_level'][events[200.]['targets'][chosen]], reference['source_level'][top['targets'][chosen]])
    assert all(abs(event['balance_residual']) < 1e-9 for event in events.values())
    for wrong in [dict(price=1.), dict(selection='priced', price=-1.)]:
        with pytest.raises(ValueError):
            replace(SCENARIO, **wrong)


@pytest.mark.parametrize('allocation', ['proportional', 'human_priority'])
def test_local_cap_keeps_sources_above_the_floor(allocation):
    world = generate_world(SPEC, 9)
    drivers = Drivers(3, len(world.positions))
    capped, free = (Simulator(world, replace(SCENARIO, pressure=3, allocation=allocation, floor=floor), Policy()) for floor in [.25, 0.])
    lowest = {capped: 1., free: 1.}
    for t in range(40):
        for simulator in [capped, free]:
            event = simulator.step(*drivers.step(t))
            assert abs(event['balance_residual']) < 1e-9
            lowest[simulator] = min(lowest[simulator], float((simulator.water/world.capacity)[simulator.accessible].min()))
    assert lowest[capped] >= .25-1e-12 and lowest[free] < .1
    with pytest.raises(ValueError):
        replace(SCENARIO, floor=1.)


def test_siting_lowers_the_highest_load_and_keeps_the_same_demand():
    config = tiny_config()
    config['cells']['sited'] = {'siting': 'sited'}
    for index in range(2):
        world, scenario, _, _ = paper_experiments.cell_context(config, 'reference', index)
        sited, changed, requests, _ = paper_experiments.cell_context(config, 'sited', index)
        assert requests is None and np.array_equal(world.resources, sited.resources) and np.array_equal(world.initial, sited.initial)
        assert np.array_equal(world.positions[~world.kinds], sited.positions[~world.kinds]) and sited.access[sited.kinds].any(axis=1).all()
        before, after = Simulator(world, scenario, Policy()), Simulator(sited, changed, Policy())
        # the same requests in resource units, although more sources may now be reachable.
        assert np.allclose(before.cooperative_request, after.cooperative_request)
        assert local_load(sited, after.cooperative_request).max() <= local_load(world, before.cooperative_request).max()+1e-12


def test_scarcity_of_the_chosen_source_is_recorded_for_every_window():
    world = generate_world(SPEC, 9)
    _, rows, _, _ = run_segment(world, replace(SCENARIO, selection='priced', price=4.), Policy(), 3, 0, 60, windows=WINDOWS)
    assert all(0 <= row[key] <= 1 for row in rows for key in ['human_scarcity', 'ai_scarcity'])


def test_contrast_between_two_settings_pairs_the_same_worlds():
    from human_ai_commons.paper_tables import _confirmatory
    rows = [dict(experiment='interventions', cell=cell, world_index=w, condition='unchanged', ai_activity=base+.01*w)
            for cell, base in [('sited', .7), ('reference', .6)] for w in range(4)]
    spec = {'contrasts': [{'name': 'siting', 'experiment': 'interventions', 'cell': 'sited', 'reference_cell': 'reference', 'outcome': 'ai_activity',
                           'select': {'condition': 'unchanged'}, 'reference': {'condition': 'unchanged'}}]}
    made = _confirmatory(rows, spec)[0]
    assert made['worlds'] == 4 and made['difference'] == pytest.approx(.1) and made['reference_cell'] == 'reference' and made['worlds_positive'] == 4


def test_contrast_leaves_out_worlds_where_the_outcome_is_undefined():
    from human_ai_commons.paper_tables import _confirmatory
    rows = [dict(experiment='feedback', cell='reference', world_index=w, start=start, contested_stock=None if w == 2 else base+.01*w, stock=base)
            for start, base in [('empty', .4), ('full', .5)] for w in range(5)]
    spec = {'contrasts': [{'name': name, 'experiment': 'feedback', 'outcome': outcome, 'select': {'start': 'empty'}, 'reference': {'start': 'full'}}
                          for name, outcome in [('contested', 'contested_stock'), ('all', 'stock')]]}
    contested, everything = _confirmatory(rows, spec)
    assert contested['worlds'] == 4 and everything['worlds'] == 5
    assert contested['difference'] == pytest.approx(-.1) and contested['bounded_lower'] <= contested['difference'] <= contested['bounded_upper']


def test_price_and_cap_can_be_limited_to_ai_agents():
    world = generate_world(SPEC, 9)
    drivers = Drivers(3, len(world.positions))
    water = np.where(np.arange(len(world.capacity)) % 2 == 0, world.capacity, .3*world.capacity)
    made = {name: Simulator(world, replace(SCENARIO, pressure=3, **changes), Policy()) for name, changes in [
        ('usual', {}), ('price', dict(selection='priced', price=8., regulated='ai')), ('cap', dict(floor=.25, regulated='ai')),
        ('cap_all', dict(floor=.25))]}
    for simulator in made.values():
        simulator.restore(water, world.initial)
    events = {name: simulator.step(*drivers.step(0)) for name, simulator in made.items()}
    human, chosen = ~world.kinds, events['usual']['targets'] >= 0
    level = events['usual']['source_level']
    # human agents keep their random choice under a price on ai agents, and ai agents move toward fuller sources.
    assert np.array_equal(events['price']['targets'][human], events['usual']['targets'][human])
    assert level[events['price']['targets'][chosen & world.kinds]].mean() > level[events['usual']['targets'][chosen & world.kinds]].mean()
    # a cap on ai agents never lowers a human delivery, and ai agents alone leave every source at or above the floor.
    assert np.all(events['cap']['delivered'][human] >= events['usual']['delivered'][human]-1e-12)
    assert events['cap']['delivered'][human].sum() > events['cap_all']['delivered'][human].sum()
    taken = np.bincount(events['cap']['targets'][chosen & world.kinds], weights=events['cap']['delivered'][chosen & world.kinds], minlength=len(water))
    assert np.all(taken <= np.maximum(0, level-.25)*world.capacity+1e-9)
    assert all(abs(event['balance_residual']) < 1e-9 for event in events.values())
    ordered = Simulator(world, replace(SCENARIO, pressure=3, allocation='human_priority', floor=.25, regulated='ai'), Policy())
    ordered.restore(water, world.initial)
    event = ordered.step(*drivers.step(0))
    taken = np.bincount(event['targets'][chosen & world.kinds], weights=event['delivered'][chosen & world.kinds], minlength=len(water))
    assert abs(event['balance_residual']) < 1e-9 and np.all(taken <= np.maximum(0, level-.25)*world.capacity+1e-9)
    with pytest.raises(ValueError):
        replace(SCENARIO, regulated='human')


def test_partial_siting_moves_the_stated_share_of_ai_agents():
    config = tiny_config()
    config['cells'].update(quarter={'siting': 'sited', 'sited_share': .25}, whole={'siting': 'sited'})
    paper_experiments.validate_protocol(config)
    world, scenario, _, _ = paper_experiments.cell_context(config, 'reference', 0)
    for cell, share in [('quarter', .25), ('whole', 1.)]:
        sited, changed, _, _ = paper_experiments.cell_context(config, cell, 0)
        moved = np.any(sited.positions != world.positions, axis=1)
        assert not moved[~world.kinds].any() and moved.sum() <= round(share*world.kinds.sum())
        assert np.allclose(Simulator(world, scenario, Policy()).cooperative_request, Simulator(sited, changed, Policy()).cooperative_request)
    assert 0 < np.any(paper_experiments.cell_context(config, 'quarter', 0)[0].positions != world.positions, axis=1).sum()
    config['cells']['wrong'] = {'sited_share': .5}
    with pytest.raises(ValueError):
        paper_experiments.validate_protocol(config)


def test_report_names_measures_and_compares_placements_of_the_same_worlds(tmp_path):
    import csv
    from human_ai_commons.paper_tables import report
    config = tiny_config()
    config['policies'].update(price_4={'r_c': 0, 'r_d': 0, 'scenario': {'selection': 'priced', 'price': 4, 'regulated': 'ai'}},
                              cap_25_everyone={'r_c': 0, 'r_d': 0, 'scenario': {'floor': .25}},
                              small_users_first={'r_c': 0, 'r_d': 0, 'scenario': {'allocation': 'human_priority'}})
    config['cells']['sited'] = {'siting': 'sited', 'sited_share': .5}
    config['interventions'].update(cells=['reference', 'sited'], from_initialization_cells=[], windows_after=[[0, 10], [10, 30], [0, 30]],
                                   designs=['price_4', 'cap_25_everyone', 'small_users_first', 'threshold_up'])
    late = {'condition': 'unchanged', 'timing': 'after_development', 'window': '0-30'}
    config['confirmatory']['contrasts'] = [{'name': 'siting', 'experiment': 'interventions', 'cell': 'sited', 'reference_cell': 'reference',
                                            'outcome': 'ai_activity', 'select': late, 'reference': late}]
    paper_experiments.run_protocol(config, tmp_path/'run', 'interventions', 10**9)
    made = report(tmp_path/'run')
    assert {'intervention_effects.csv', 'intervention_placement_effects.csv', 'confirmatory_contrasts.csv'} <= set(made['tables'])
    effects = list(csv.DictReader(open(tmp_path/'run/report/intervention_effects.csv')))
    names = {e['design']: e['measure'] for e in effects}
    assert names == {'price_4': 'price, strength 4', 'cap_25_everyone': 'cap at 25%, all agents', 'small_users_first': 'human agents served first',
                     'threshold_up': 'switching level %.2f' % paper_experiments.policy(config, 'threshold_up').q_c}
    assert all(e['ai_scarcity_difference_mean'] != '' for e in effects)
    placement = list(csv.DictReader(open(tmp_path/'run/report/intervention_placement_effects.csv')))
    assert {p['window'] for p in placement} == {'0-10', '10-30', '0-30'} and all(p['random_placement_cell'] == 'reference' and p['worlds'] == '2' for p in placement)
    contrast = list(csv.DictReader(open(tmp_path/'run/report/confirmatory_contrasts.csv')))[0]
    whole = next(p for p in placement if p['window'] == '0-30')
    assert float(contrast['difference']) == pytest.approx(float(whole['ai_activity_difference_mean'])) and contrast['reference_cell'] == 'reference'
    # the table scripts read this report when the results folder is named in JCS_RESULTS; the figure scripts draw the paper's own settings.
    import os, runpy, sys
    (tmp_path/'run').rename(tmp_path/'local_measures')
    scripts = Path(__file__).parents[1]
    old = os.environ.get('JCS_RESULTS'), list(sys.argv), list(sys.path)
    os.environ['JCS_RESULTS'] = str(tmp_path)
    try:
        for script, arguments in [('analysis/table_local_measures.py', []), ('analysis/table_declared_comparisons.py', [])]:
            sys.argv = [script, *arguments]
            runpy.run_path(str(scripts/script), run_name='__main__')
    finally:
        sys.argv, sys.path = old[1], old[2]
        os.environ.pop('JCS_RESULTS') if old[0] is None else os.environ.update(JCS_RESULTS=old[0])


def test_requests_are_recorded_by_group_and_final_states_ignore_window_order():
    config = tiny_config()
    config['policies']['cap_50'] = {'r_c': 0, 'r_d': 0, 'scenario': {'floor': .5, 'regulated': 'ai'}}
    config['interventions'].update(cells=['reference'], from_initialization_cells=[], designs=['cap_50'])
    saved = {}
    for label, windows in [('whole_last', [[0, 10], [10, 30], [0, 30]]), ('whole_first', [[0, 30], [0, 10], [10, 30]])]:
        config['interventions']['windows_after'] = windows
        saved[label] = paper_experiments.interventions_world(config, 0, 'reference')
    for name in ['unchanged', 'cap_50']:
        assert np.array_equal(saved['whole_last'][1][name]['final_stock'], saved['whole_first'][1][name]['final_stock'])
    rows = [r for r in saved['whole_first'][0] if r['timing'] == 'after_development']
    for row in rows:
        # nothing is delivered beyond what was asked, and what a group asks lies between its lower and its higher requests.
        assert row['human_delivered'] <= row['human_requested']+1e-12 and row['ai_delivered'] <= row['ai_requested']+1e-12
        assert 0 <= row['human_cooperation'] <= 1 and 0 <= row['ai_cooperation'] <= 1
        lower = row['total_ordinary_request']
        assert lower-1e-9 <= row['human_requested']+row['ai_requested'] <= 1.5*lower+1e-9
    assert json.loads(next(r for r in rows if r['condition'] == 'cap_50')['scenario_changes']) == {'floor': .5, 'regulated': 'ai'}
    assert rows[0]['ai_reachable_sources'] > 0 and rows[0]['human_reachable_sources'] > 0


def test_a_restriction_with_nothing_to_restrict_is_refused_and_shares_round_half_up():
    with pytest.raises(ValueError):
        replace(SCENARIO, regulated='ai')
    world = generate_world(WorldSpec(), 5)
    moved = np.any(paper_experiments.site_ai_agents(world, 20, .05, .25).positions != world.positions, axis=1)
    assert world.kinds.sum() == 50 and not moved[~world.kinds].any() and moved.sum() <= 13
    with pytest.raises(ValueError):
        paper_experiments.site_ai_agents(world, 20, .05, 0.)


def test_siting_prefers_cells_away_from_human_sources_only_where_sources_can_carry_the_load():
    world = generate_world(WorldSpec(), 5)
    used = world.access[~world.kinds].any(axis=0)
    shares = lambda w: (w.access[w.kinds] & used[None, :]).any(axis=1).mean()
    # a tiny request leaves every source able to carry its load, so every ai agent can be kept off the sources humans use.
    light = paper_experiments.site_ai_agents(world, 20, 1e-6)
    assert shares(light) == 0 and light.access[light.kinds].any(axis=1).all()
    assert np.array_equal(light.positions[~world.kinds], world.positions[~world.kinds]) and np.array_equal(light.resources, world.resources)
    # a large request leaves no such room: the lowest load decides, and ai agents stay on shared sources.
    heavy = paper_experiments.site_ai_agents(world, 20, 1.)
    assert shares(heavy) > .5
    assert np.unique(heavy.positions, axis=0, return_counts=True)[1].max() <= world.spec.maximum_occupancy
