from copy import deepcopy
from dataclasses import replace
import json
import numpy as np
import pytest
import human_ai_commons.experiments as experiments
from human_ai_commons.config import load_config, validate_config
from human_ai_commons.optimization import one_dimensional, differential_search
from human_ai_commons.payoffs import Policy
from human_ai_commons.provenance import ROOT, read_jsonl
from human_ai_commons.world import World, WorldSpec


def test_threshold_refinement_can_follow_a_right_hand_minimum():
    settings = dict(initial_grid_points=5, refinement_rounds=1,
                    refined_intervals_per_round=1, refinement_grid_points=9)
    points = one_dimensional(lambda p: (p.relative[0]-.07)**2, .5, 2., -.5, 'L2', settings, (-.95, 2.), 'equal_relative')
    assert abs(points[0].relative[0]-.07) < .04


@pytest.mark.parametrize('family', ['equal_relative', 'resource_only'])
def test_flat_training_losses_use_the_same_minimum_cost_tie_break(family):
    settings = dict(initial_grid_points=5, refinement_rounds=0,
                    refined_intervals_per_round=1, refinement_grid_points=9)
    points = one_dimensional(lambda p: 0, .5, 2., -.5, 'L2', settings, (-.95, 2.), family)
    assert points[0].cost() == pytest.approx(0, abs=1e-15)


def test_nested_shortlist_fills_all_unique_slots():
    rows = [dict(policy_id=str(i)) for i in range(5)]
    assert experiments.nested_shortlist(rows, rows[0], 3) == rows[:3]


def test_zero_budget_preserves_global_observation():
    evaluated = []
    differential_search(evaluated.append, 0, 2., -.5, 'L2', (-.95, 2.),
                        {'observation': 'global'}, 1, Policy())
    assert evaluated[0].observation == 'global'


def test_payoff_transfer_summary_records_executed_policy(tmp_path):
    config = load_config(ROOT/'configs/smoke.json')
    ctx = experiments.Evaluation(config, tmp_path, 100)
    summary = ctx.aggregate(1, Policy(), 'robustness', 1, variant={'policy': {'u': 1}})
    assert summary['policy']['u'] == 1
    assert summary['policy_id'] == replace(Policy(), u=1).id


def test_failed_simulation_is_charged_to_budget(tmp_path, monkeypatch):
    config = load_config(ROOT/'configs/smoke.json')
    ctx = experiments.Evaluation(config, tmp_path, 12)
    def fail(*args, **kwargs):
        raise RuntimeError('interrupted simulation')
    monkeypatch.setattr(experiments, 'simulate', fail)
    with pytest.raises(RuntimeError, match='interrupted'):
        ctx.realization(1, Policy(), 'train', 0)
    ledger = read_jsonl(tmp_path/'budget_ledger.jsonl')
    assert sum(row['steps'] for row in ledger) == 12
    assert ledger[-1]['status'] == 'failed'
    with pytest.raises(RuntimeError, match='budget exhausted'):
        ctx.realization(1, Policy(), 'train', 0)


def test_resumed_training_reproduces_uninterrupted_shortlists(tmp_path, monkeypatch):
    config = load_config(ROOT/'configs/smoke.json')
    complete, interrupted = tmp_path/'complete', tmp_path/'interrupted'
    experiments.run_stages(config, complete, 'train', 10000)
    original = experiments.frontiers.write_json
    def interrupt(path, data):
        if path.name == 'shortlists.json':
            raise RuntimeError('interrupted before saving shortlists')
        return original(path, data)
    monkeypatch.setattr(experiments.frontiers, 'write_json', interrupt)
    with pytest.raises(RuntimeError, match='before saving'):
        experiments.run_stages(config, interrupted, 'train', 10000)
    monkeypatch.setattr(experiments.frontiers, 'write_json', original)
    experiments.run_stages(config, interrupted, 'train', 10000)
    assert json.loads((complete/'shortlists.json').read_text()) == json.loads((interrupted/'shortlists.json').read_text())


@pytest.mark.parametrize('field,value', [('resources', [[.9, 0]]), ('positions', [[.1, 0]]),
                                       ('kinds', [2]), ('initial', [np.nan])])
def test_world_rejects_silent_coordinate_and_label_casts(field, value):
    arrays = dict(resources=[[0, 0]], capacity=[1], renewal=[.1], positions=[[0, 0]], kinds=[0], initial=[1])
    arrays[field] = value
    with pytest.raises(ValueError):
        World(WorldSpec(width=2, height=2, humans=1, ai=0), **arrays)


@pytest.mark.parametrize('change', ['unknown_condition', 'nan_budget', 'unmatched_global'])
def test_invalid_experiment_specifications_fail_early(change):
    config = deepcopy(load_config(ROOT/'configs/smoke.json'))
    if change == 'unknown_condition': config['world']['condition'] = 'typo'
    elif change == 'nan_budget': config['search']['main_problems'][0]['budget'] = float('nan')
    else: config['search']['global_composition_reoptimization'] = {'pressure': 2, 'budget': .5}
    with pytest.raises(ValueError): validate_config(config)


def test_capacity_relaxation_matches_exhaustive_small_graph_cuts():
    from human_ai_commons.capacity import routing_bound
    generator = np.random.default_rng(71)
    for _ in range(40):
        access = generator.random((3, 3)) < .5
        demand = generator.integers(1, 6, 3)
        capacity = generator.integers(1, 6, 3)
        shortage = 0
        for bits in range(8):
            subset = np.array([bool(bits & (1 << j)) for j in range(3)])
            confined = ~access[:, ~subset].any(axis=1)
            shortage = max(shortage, int(demand[confined].sum()-capacity[subset].sum()))
        result = routing_bound(access, demand, capacity)
        assert result['required']-result['relaxed_deliverable'] == pytest.approx(shortage)
        assert result['relaxed_routing_infeasible'] == (shortage > 0)


@pytest.mark.parametrize('variant', [dict(name='typo', world_transform='clustr'),
                                     dict(name='bad_window', steps=2, burn=3),
                                     dict(name='extra', unexpected=True)])
def test_invalid_robustness_controls_are_not_silently_ignored(variant):
    config = deepcopy(load_config(ROOT/'configs/smoke.json'))
    config['execution']['robustness_variants'] = [variant]
    with pytest.raises(ValueError): validate_config(config)
