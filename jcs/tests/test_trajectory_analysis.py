"""Scientific postprocessing checks using synthetic saved observations, never model runs."""
import numpy as np
import pytest
from human_ai_commons.trajectory_analysis import events
from human_ai_commons.trajectory_analysis.pipeline import analyze_trajectories
from human_ai_commons.trajectory_analysis.static_analysis import tail


def test_reference_cohort_uses_first_crossing_and_retains_empty_accessible_sources(tmp_path, monkeypatch):
    # source 0 crosses once, source 1 begins below the threshold and later empties;
    # source 2 is inaccessible and must never enter the reference event cohort.
    stocks = np.array([[.8,.4,.0],[.4,.0,.0],[.2,.8,.0]])
    np.savez(tmp_path/'history.npz',stocks=stocks,accessible=[True,True,False])
    monkeypatch.setattr(events,'P',tmp_path)
    monkeypatch.setattr(events,'load_world',lambda _:({'capacity':np.ones(3)},{}))
    key, cohort=events.first_crossings(('test',1,'reference',7,'evaluation','history.npz','unused'))
    assert key=='1|7'
    assert cohort['0.5']==[1,0,-1]
    assert cohort['0.25']==[2,1,-1]
    assert cohort['0.75']==[1,0,-1]


def test_worst_tenth_uses_fractional_boundary_weight():
    # a 15-agent group has a 1.5-agent tenth: one full observation plus half the next.
    assert tail(np.arange(15)/14)==pytest.approx((0+.5/14)/1.5)


def test_full_trajectory_analysis_cannot_overwrite_saved_bundle(tmp_path):
    bundle=tmp_path/'bundle';bundle.mkdir()
    with pytest.raises(ValueError,match='separate'):
        analyze_trajectories(bundle,bundle/'analysis')
    assert list(bundle.iterdir())==[]


def test_full_trajectory_analysis_rejects_nonempty_destination(tmp_path):
    output=tmp_path/'output';output.mkdir();(output/'existing').write_text('retained')
    with pytest.raises(ValueError,match='empty'):
        analyze_trajectories(tmp_path/'bundle',output)
    assert (output/'existing').read_text()=='retained'


def test_target_coverage_accounting_separates_lower_requests_from_unfilled_requests():
    from human_ai_commons.trajectory_analysis.additional_evidence import coverage_components
    # humans request the full target and get it; AI always requests two-thirds
    # but receives one-third: its gap has equal request and rationing parts.
    parts=coverage_components(1.,1/3,0.,1.)
    assert parts['lower_request_contribution']==pytest.approx(1/3)
    assert parts['unfilled_request_contribution']==pytest.approx(1/3)
    assert parts['coverage_gap']==pytest.approx(2/3)


def test_resource_budget_reconstructs_clipped_renewal_and_stored_water(tmp_path,monkeypatch):
    from human_ai_commons.trajectory_analysis import resource_accounting as accounting
    world=dict(capacity=np.array([20.]),initial=np.array([False,False]),
               resources=np.array([[0,0]]),positions=np.array([[0,0],[0,0]]),
               renewal=np.array([1.]),kinds=np.array([False,True]))
    renewal=np.ones((2000,1));renewal[0]=0
    np.savez(tmp_path/'history.npz',stocks=np.full((2000,1),19.),strategies=np.zeros((2000,2),bool),
             resource_renewal=renewal,source_withdrawal=np.ones((2000,1)),accessible=np.array([True]),
             agent_targets=np.zeros((2000,2),int))
    monkeypatch.setattr(accounting,'P',tmp_path)
    monkeypatch.setattr(accounting,'load_world',lambda _:(world,dict(width=20,height=20,feedback_radius=3)))
    row,profile=accounting.process(('test',1.,'policy',0,'id','history.npz','unused',None,{'forced':None}))
    assert row['stock_step2000']==pytest.approx(.95)
    assert row['actual_renewal_step2000']==pytest.approx(1999/20)
    assert row['unused_renewal_step2000']==pytest.approx(.05)
    assert np.max(abs(profile['stock']-1-profile['actual_renewal']+profile['withdrawal']))<1e-10


def test_full_sources_erase_initial_composition_at_the_first_preference():
    from human_ai_commons.payoffs import Policy
    # at n=1 the u,v changes vanish and either original action advantage is positive.
    for first in [-.95,0.,.5,2.]:
        for second in [-.95,0.,.5,2.]:
            policy=Policy.from_relative(first,second)
            assert np.all(policy.gap(1.,np.linspace(0,1,101))>0)


def test_joint_bootstrap_sensitivity_preserves_scaling_across_independent_splits():
    from human_ai_commons.trajectory_analysis.additional_evidence import bootstrap_sensitivity
    generator=np.random.default_rng(7)
    blocks=[(generator.normal(0,.01,(32,2)),[{'scope':'first','outcome':k} for k in ['a','b']]),
            (generator.normal(0,.01,(24,1)),[{'scope':'second','outcome':'c'}])]
    a=bootstrap_sensitivity(blocks,samples=300)
    b=bootstrap_sensitivity([(v*.3,m) for v,m in blocks],samples=300)
    assert np.allclose(a.critical_value,b.critical_value)
    assert np.allclose(a.exploratory_joint_lower*.3,b.exploratory_joint_lower)
    assert np.allclose(a.exploratory_joint_upper*.3,b.exploratory_joint_upper)
