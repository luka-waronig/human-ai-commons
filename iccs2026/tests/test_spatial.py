import numpy as np
import pytest
from spatial_analysis_sweep import morans_I_masked, pair_stats_for_points, agent_cluster_metrics_agents_only


def test_masked_moran_uses_only_edges_with_both_endpoints_in_mask():
    field = np.zeros((5, 4)); mask = np.zeros_like(field, bool)
    field[0, 0], field[0, 1] = 1, -1
    mask[0, 0] = mask[0, 1] = True
    assert morans_I_masked(field, mask) == pytest.approx(-1)
    mask[0, 1] = False; mask[2, 2] = True
    assert np.isnan(morans_I_masked(field, mask))


def test_pair_correlation_has_correct_unordered_pair_normalization():
    points = np.array([[0,0], [1.1,0]])
    _, _, g = pair_stats_for_points(points, 20, 20, np.array([1.,2.]))
    assert g[0] == pytest.approx(200 / np.pi)
    assert g[1] == 0


def test_continuous_uniform_torus_null_is_near_one():
    rng = np.random.default_rng(2026)
    values = [pair_stats_for_points(rng.uniform(0,20,(200,2)),20,20,np.arange(1.,4.5,.5))[2]
              for _ in range(50)]
    assert abs(np.mean(values) - 1) < .04


def test_clusters_support_rectangular_model_arrays():
    agents = [(1,'Human',5,3,'C'), (2,'AI',0,3,'C')]
    assert agent_cluster_metrics_agents_only(agents, height=4, width=6) == (1,2,2.)


def test_conditional_null_matches_model_field_and_preserves_constant_case():
    from water_toc.model import WaterToC
    from water_toc.agents import Strategy
    from water_toc.spatial_null import conditional_moran_test
    model = WaterToC(seed=17,save_snapshots=False)
    agents = model._get_agent_pos_strategies_list()
    mask = np.zeros((model.width,model.height),dtype=bool)
    for agent in agents:mask[agent[2],agent[3]] = True
    observed = morans_I_masked(model._get_local_cooperation_map(),mask)
    result = conditional_moran_test(agents,model.width,model.height,permutations=49,seed=3)
    assert result['observed'] == pytest.approx(observed)
    assert 0 < result['p_upper'] <= 1
    assert result == conditional_moran_test(agents,model.width,model.height,permutations=49,seed=3)
    for agent in model.agents:agent.strategy = Strategy.COOPERATE
    constant = conditional_moran_test(model._get_agent_pos_strategies_list(),20,20,permutations=9)
    assert np.isnan(constant['observed'])


def test_thin_torus_deduplicates_wrapped_neighbors_and_excludes_self():
    from spatial_analysis_sweep import degree_map_moore
    np.testing.assert_array_equal(degree_map_moore((1,3)),np.full((1,3),2))
    np.testing.assert_array_equal(degree_map_moore((1,1)),np.zeros((1,1)))
    field = np.array([[1.,-1.,0.]])
    mask = np.array([[True,True,False]])
    assert morans_I_masked(field,mask) == pytest.approx(-1)
