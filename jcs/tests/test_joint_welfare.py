import numpy as np
import pytest

from watertoc_jcs.config import load_config, scenario as configured_scenario, world_spec
from watertoc_jcs.dynamics import Scenario, Simulator, simulate
from watertoc_jcs.experiments import candidate_key
from watertoc_jcs.metrics import LocalStress, Service, StressSpec, WelfareSpec, joint_welfare, probability_upper_bound, supported_activity
from watertoc_jcs.optimization import differential_search, one_dimensional
from watertoc_jcs.payoffs import Policy
from watertoc_jcs.provenance import ROOT
from watertoc_jcs.world import World, WorldSpec


def shared_source_world():
    return World(WorldSpec(width=2, height=1, humans=1, ai=1, density=1, capacity=1),
                 [[0, 0]], [1.], [.1], [[0, 0], [1, 0]], [False, True], [True, True])


def test_extra_delivery_can_support_more_activity_without_changing_its_target():
    service = Service([1.5, 1.5], [False, True], WelfareSpec())
    service.update([1., 1.5])
    result = service.finish()
    assert result['human_service'] == pytest.approx(2/3)
    assert result['ai_service'] == 1
    assert result['welfare'] == pytest.approx(np.sqrt(2/3))
    assert result['agent_service_target'] == [1.5, 1.5]


@pytest.mark.parametrize('extra_value', [0., .5, 1.])
def test_the_value_of_additional_use_is_an_explicit_assumption(extra_value):
    baseline, higher = 1., 1.5
    assert supported_activity(0., baseline, higher, extra_value) == 0
    assert supported_activity(higher, baseline, higher, extra_value) == 1
    assert supported_activity(baseline, baseline, higher, extra_value) == pytest.approx(1/(1+.5*extra_value))


def test_zero_additional_use_value_does_not_reward_delivery_above_the_baseline():
    service = Service([1.5, 1.5], [False, True], WelfareSpec(additional_use_value=0), baseline=[1., 1.])
    service.update([1., 1.5])
    assert service.finish()['welfare'] == 1


def test_delivery_is_capped_before_averaging_over_agents_and_time():
    service = Service([1., 1., 1.], [False, False, True], WelfareSpec())
    service.update([.5, 1.5, 1.])
    service.update([0., 2., 1.])
    result = service.finish()
    assert result['human_service'] == .625
    assert result['ai_service'] == 1
    assert result['human_service'] < (1+1)/2


def test_equal_priority_rewards_joint_support_and_priority_extremes_are_explicit():
    assert joint_welfare(.5, .5) > joint_welfare(.9, .1)
    assert joint_welfare(1., 0.) == 0
    assert joint_welfare(.9, .1, 1) == .9
    assert joint_welfare(.9, .1, 0) == .1


def test_main_selection_can_prefer_welfare_over_larger_unused_resource_stock():
    def candidate(policy, human, ai, resource_loss, violation=0):
        return dict(policy=policy.__dict__, policy_id=policy.id, mean_loss=resource_loss,
                    objective_loss=1-float(joint_welfare(human, ai)), stress_probability_violation=violation)
    unused = candidate(Policy(), .1, .1, 0)
    supported = candidate(Policy(.6, .6), .7, .7, .2)
    unacceptable = candidate(Policy(.7, .7), 1., 1., 0, .1)
    assert min([unused, supported, unacceptable], key=lambda r: candidate_key(r, 'L2')) is supported


def test_continuous_stress_includes_the_published_moderately_stressed_regime():
    tracker = LocalStress(2, 10, StressSpec(reference=.5, duration_steps=3))
    for _ in range(10):
        tracker.update([.38, .8])
    result = tracker.finish()
    assert result['persistent_local_stress']
    assert result['persistent_cell_fraction'] == .5
    assert result['cell_longest_continuous_stress_steps'] == [10, 0]
    assert result['stress_duration_steps'] == 3


def test_short_dips_are_distinct_from_a_continuous_shortage():
    tracker = LocalStress(1, 10, StressSpec(duration_steps=3))
    for value in [.1, .8]*5:
        tracker.update([value])
    assert not tracker.finish()['persistent_local_stress']


def test_zero_observed_failures_does_not_mean_zero_failure_probability():
    upper = probability_upper_bound(np.zeros(8, dtype=bool))
    assert upper == pytest.approx(1-.05**(1/8))
    assert upper > .1
    assert probability_upper_bound(np.zeros(64, dtype=bool)) < .1
    assert probability_upper_bound(np.ones(8, dtype=bool)) == 1


def test_proportional_sharing_does_not_give_an_agent_type_first_access():
    world = shared_source_world()
    common = dict(pressure=20, demand_ratio=1, epsilon=0, theta=0)
    proportional = Simulator(world, Scenario(**common, allocation='proportional'), Policy(forced='C'))
    priority = Simulator(world, Scenario(**common, allocation='human_priority'), Policy(forced='C'))
    drivers = (np.zeros(2), np.full(2, .5), np.arange(2))
    shared, ordered = proportional.step(*drivers), priority.step(*drivers)
    np.testing.assert_allclose(shared['delivered'], [.5, .5])
    np.testing.assert_allclose(ordered['delivered'], [1., 0.])
    np.testing.assert_allclose(proportional.water, priority.water)
    assert shared['balance_residual'] == pytest.approx(0)


def test_welfare_and_stress_can_be_recomputed_from_recorded_individual_outcomes():
    world = shared_source_world()
    scenario = Scenario(pressure=1, demand_ratio=1, allocation='proportional')
    spec, stress = WelfareSpec(), StressSpec(duration_steps=3)
    summary, data = simulate(world, scenario, Policy(), 31, 8, 2, record=True, welfare=spec, stress=stress)
    service = Service(data['agent_reference_request']*scenario.defect_ratio, data['agent_is_ai'], spec)
    tracker = LocalStress(int(data['accessible'].sum()), 6, stress)
    for delivered, water in zip(data['agent_delivered'][2:], data['stocks'][2:]):
        service.update(delivered)
        tracker.update(water[data['accessible']]/world.capacity[data['accessible']])
    for key, value in (service.finish() | tracker.finish()).items():
        assert summary[key] == value


def test_more_supported_activity_can_outweigh_a_small_reduction_in_unused_stock():
    world = shared_source_world()
    scenario = Scenario(pressure=.1, demand_ratio=1, allocation='proportional')
    results = [simulate(world, scenario, Policy(forced=action), 3, 8, 2,
                        welfare=WelfareSpec(), stress=StressSpec(duration_steps=3))[0] for action in ['C', 'D']]
    cooperative, higher_activity = results
    assert cooperative['agent_service_target'] == higher_activity['agent_service_target']
    assert cooperative['mean_accessible_resource'] > higher_activity['mean_accessible_resource']
    assert cooperative['welfare'] == pytest.approx(2/3)
    assert higher_activity['welfare'] == pytest.approx(1)
    assert not higher_activity['persistent_local_stress']


def test_comparator_rejects_a_high_benefit_that_violates_the_constraint():
    settings = dict(initial_grid_points=5, refinement_rounds=0, refined_intervals_per_round=1, refinement_grid_points=3)
    points = one_dimensional(lambda p: p.relative[0], .5, 2., -.5, 'L2', settings, (-.95, 2.), 'equal_relative',
                             lambda p: .2-p.relative[0])
    assert points[0].relative[0] >= .2


def test_evolutionary_search_obeys_an_outcome_constraint():
    settings = dict(explicit_population_size=8, maximum_generations=3, strategy='best1bin', mutation=[.5, 1], recombination=.7)
    evaluated = []
    def objective(policy):
        assert policy.relative[0] >= .2
        evaluated.append(policy)
        return policy.q_c
    differential_search(objective, .5, 2., -.5, 'L2', (-.95, 2.), settings, 31, Policy.from_relative(.3, .3), lambda p: .2-p.relative[0])
    assert evaluated


def test_research_configuration_optimizes_joint_welfare_with_a_stress_constraint():
    config = load_config(ROOT/'configs/research_protocol.json')
    assert config['objective']['name'] == 'joint_service_welfare_with_persistent_local_stress_constraint'
    assert config['world']['withdrawal_allocation'] == 'proportional'
    assert config['payoffs']['optimized_parameters'] == ['u','v']


@pytest.mark.parametrize('name,weight', [('human_only_priority', 1), ('ai_only_priority', 0)])
def test_priority_controls_change_the_social_preference_without_changing_the_physical_system(name, weight):
    reference = load_config(ROOT/'configs/research_protocol.json')
    control = load_config(ROOT/f'configs/{name}.json')
    assert control['objective']['welfare']['human_weight'] == weight
    assert world_spec(control) == world_spec(reference)
    assert configured_scenario(control, 1) == configured_scenario(reference, 1)
    assert control['objective']['persistent_local_stress'] == reference['objective']['persistent_local_stress']
    assert control['validation']['base_seed'] == reference['validation']['base_seed']


def test_a_stress_duration_longer_than_the_observed_trajectory_is_rejected():
    with pytest.raises(ValueError, match='duration'):
        LocalStress(1, 2, StressSpec(duration_steps=3))


def test_delivery_condition_is_a_matched_alternative_not_a_change_to_welfare_or_physics():
    base=load_config(ROOT/'configs/research_protocol.json')
    alternative=load_config(ROOT/'configs/delivery_reliability.json')
    assert alternative['objective']['persistent_local_stress']['basis']=='request_fulfillment'
    assert alternative['objective']['welfare']==base['objective']['welfare']
    assert alternative['world']==base['world']
    assert alternative['payoffs']==base['payoffs']


@pytest.mark.parametrize('spec', [dict(human_weight=np.nan), dict(utility='unknown'), dict(additional_use_value=np.nan)])
def test_invalid_welfare_assumptions_are_rejected(spec):
    with pytest.raises(ValueError):
        WelfareSpec(**spec)
