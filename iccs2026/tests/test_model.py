import json
from pathlib import Path
import numpy as np
import pytest
from water_toc.model import WaterToC
from water_toc.agents import Strategy
from water_toc.feedback import replenishment_multiplier


def test_default_trajectories_match_historical_reference(tmp_path):
    reference = json.loads((Path(__file__).parents[1] / 'reproducibility/baseline_reference.json').read_text())
    for record in reference['cases']:
        case = dict(record['case'])
        steps = case.pop('steps')
        model = WaterToC(**case, save_snapshots=False, snapshot_dir=tmp_path / 'unused')
        for _ in range(steps):
            model.step()
        actual = model.datacollector.get_model_vars_dataframe()[record['columns']].to_numpy()
        np.testing.assert_allclose(actual, record['trajectory'], rtol=0, atol=1e-12)
        np.testing.assert_allclose(model.water_levels, record['water'], rtol=0, atol=1e-12)
        assert [list(agent) for agent in model._get_agent_pos_strategies_list()] == record['agents']
    assert not (tmp_path / 'unused').exists()


def test_feedback_variants_are_explicit():
    assert replenishment_multiplier(.4, .7, 3, 'repository') == pytest.approx(1.132)
    assert replenishment_multiplier(.4, .7, 3, 'manuscript') == pytest.approx(1.216)
    assert replenishment_multiplier(.5, 0, 3, 'pre_floor_change') == pytest.approx(.875)
    assert replenishment_multiplier(.5, 0, 3, 'repository') == 1
    with pytest.raises(ValueError):
        replenishment_multiplier(.5, .5, 3, 'unidentified')


def test_original_action_is_a_half_capacity_threshold():
    agent = WaterToC(seed=4, save_snapshots=False).agents[0]
    for state in [0, .25, .49, .5, .51, .75, 1]:
        expected = Strategy.COOPERATE if state <= .5 else Strategy.DEFECT
        assert agent.choose_best_action(agent.weitz_matrix_env(state)) == expected


@pytest.mark.parametrize('params', [dict(height=0), dict(width=2.5), dict(initial_ai=-1),
    dict(deviation_rate=1.1), dict(water_cell_density=np.nan), dict(ai_C_allocation=-.1),
    dict(max_water_capacity=0), dict(height=2, width=2, initial_humans=13, initial_ai=0)])
def test_invalid_configuration_is_rejected(params):
    with pytest.raises(ValueError):
        WaterToC(**params, save_snapshots=False)


def test_crowded_grid_respects_occupancy_limit():
    model = WaterToC(height=4, width=4, initial_humans=24, initial_ai=24, seed=2, save_snapshots=False)
    assert len(model.agents) == 48
    assert all(len(model.grid.get_cell_list_contents([(x,y)])) <= 3 for x in range(4) for y in range(4))


def test_consumption_accounting_and_bounds():
    model = WaterToC(seed=11, save_snapshots=False, record_spatial=False)
    model.water_levels[model.has_water] *= .5
    before = model.water_levels.copy()
    model.replenish_water()
    actual_renewal = model.water_levels.sum() - before.sum()
    model.water_levels[:] = before
    model.step()
    frame = model.datacollector.get_model_vars_dataframe()
    row = frame.iloc[-1]
    assert model.water_levels.sum() == pytest.approx(before.sum() + actual_renewal - row.Delivered_Water)
    assert row.Requested_Water == pytest.approx(row.Delivered_Water + row.Unmet_Water)
    for _ in range(10):
        model.step()
        assert np.all(model.water_levels >= 0)
        assert np.all(model.water_levels <= model.water_capacity)
    pos = tuple(np.argwhere(model.has_water)[0])
    for amount in [-1, np.nan, np.inf]:
        with pytest.raises(ValueError):
            model.consume_water_at(pos, amount)


def test_unavailable_resource_counts_as_unmet_request():
    model = WaterToC(water_cell_density=0, seed=7, save_snapshots=False)
    model.step()
    row = model.datacollector.get_model_vars_dataframe().iloc[-1]
    assert row.Delivered_Water == 0
    assert row.Unmet_Water == row.Requested_Water > 0


def test_logging_options_do_not_change_dynamics(tmp_path):
    full = WaterToC(seed=32, save_snapshots=False, record_spatial=True)
    light = WaterToC(seed=32, save_snapshots=False, record_spatial=False)
    for _ in range(8):
        full.step(); light.step()
        np.testing.assert_array_equal(full.water_levels, light.water_levels)
        assert [a.strategy for a in full.agents] == [a.strategy for a in light.agents]
