import numpy as np
import pytest

from watertoc_jcs.dynamics import Scenario, simulate
from watertoc_jcs.metrics import Depletion, MetricSpec
from watertoc_jcs.payoffs import Policy
from watertoc_jcs.world import World, WorldSpec


@pytest.mark.parametrize('action', ['C', 'D'])
def test_individual_deliveries_reconstruct_group_and_agent_summaries(action):
    spec = WorldSpec(width=2, height=1, humans=1, ai=1, density=1, capacity=1)
    world = World(spec, [[0, 0]], [1.], [.1], [[0, 0], [1, 0]],
                  [False, True], [True, True])
    scenario = Scenario(pressure=20, demand_ratio=1)
    policy = Policy(forced=action)
    summary, data = simulate(world, scenario, policy, 31, 8, 2, record=True)
    unrecorded_summary, absent = simulate(world, scenario, policy, 31, 8, 2)
    assert summary == unrecorded_summary
    assert absent is None
    assert data['agent_requested'].shape == data['agent_delivered'].shape == (8, 2)
    previous = np.vstack([world.capacity, data['stocks'][:-1]])
    np.testing.assert_allclose(data['stocks'],previous+data['resource_renewal']-data['source_withdrawal'],atol=1e-14)
    for step, selected in enumerate(data['agent_targets']):
        for agent, source in enumerate(selected):
            assert source < 0 or world.access[agent,source]
        assert data['source_withdrawal'][step].sum() == pytest.approx(data['agent_delivered'][step].sum())
    assert np.all(data['agent_delivered'] <= data['agent_requested'])
    assert np.any(data['agent_delivered'] < data['agent_requested'])
    np.testing.assert_array_equal(data['agent_is_ai'], world.kinds)
    np.testing.assert_array_equal(data['agent_reference_request'], summary['agent_reference_request'])
    for quantity in ['requested', 'delivered']:
        values = data[f'agent_{quantity}']
        np.testing.assert_allclose(values[2:].mean(axis=0), summary[f'agent_mean_{quantity}'])
        for name, mask in [('human', ~world.kinds), ('ai', world.kinds)]:
            expected = values[:, mask].sum(axis=1)
            column = list(data['observation_columns']).index(f'{name}_{quantity}')
            np.testing.assert_allclose(data['observations'][:, column], expected)
            assert summary[f'mean_{name}_{quantity}'] == pytest.approx(expected[2:].mean())
    np.testing.assert_allclose(summary['agent_unserved_fraction'],
                               (data['agent_delivered'][2:] == 0).mean(axis=0))


def test_agent_without_resource_access_is_retained_in_outcomes():
    spec = WorldSpec(width=10, height=1, humans=2, ai=0, require_access=False)
    world = World(spec, [[0, 0]], [1.], [.1], [[0, 0], [5, 0]],
                  [False, False], [True, True])
    summary, data = simulate(world, Scenario(pressure=.1), Policy(forced='C'), 7, 5, 1, record=True)
    assert np.all(data['agent_requested'][:, 1] > 0)
    assert np.all(data['agent_delivered'][:, 1] == 0)
    assert summary['agent_unserved_fraction'][1] == 1
    assert summary['agent_mean_delivered'][1] == 0


def test_equal_total_deficits_can_have_different_continuous_durations():
    intermittent = Depletion(1, MetricSpec())
    continuous = Depletion(1, MetricSpec())
    for value in [0, 1, 0, 1]:
        intermittent.update([value])
    for value in [0, 0, 1, 1]:
        continuous.update([value])
    first, second = intermittent.finish(), continuous.finish()
    assert first['loss'] == second['loss'] == .5
    assert first['cell_depleted_fraction'] == second['cell_depleted_fraction'] == [.5]
    assert first['longest_continuous_depletion_steps'] == 1
    assert second['longest_continuous_depletion_steps'] == 2
    assert first['cell_longest_continuous_depletion_steps'] == [1]
    assert second['cell_longest_continuous_depletion_steps'] == [2]
    assert first['minimum_accessible_resource'] == second['minimum_accessible_resource'] == 0
