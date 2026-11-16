"""Read-only notebook data access and agreement with saved model measurements."""
import importlib.util
import json
from dataclasses import replace
import ast
import numpy as np
import pytest
from human_ai_commons.config import load_config, world_spec, scenario, metric_spec, welfare_spec
from human_ai_commons.provenance import ROOT
from human_ai_commons.world import generate_world
from human_ai_commons.dynamics import simulate
from human_ai_commons.payoffs import Policy

spec = importlib.util.spec_from_file_location('jcs_explore', ROOT/'notebooks/_explore.py')
explore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(explore)


@pytest.fixture
def example(tmp_path):
    config = load_config(ROOT/'configs/smoke.json')
    world = generate_world(world_spec(config), 17)
    world = replace(world, capacity=world.capacity*np.linspace(1, 2, len(world.capacity)))
    policy = Policy()
    summary, data = simulate(world, scenario(config, 1), policy, 23, 12, 2,
                              metric_spec(config), record=True, welfare=welfare_spec(config))
    (tmp_path/'manifest.json').write_text(json.dumps({'configuration':config, 'run_id':'software-example'}))
    (tmp_path/'worlds').mkdir(); (tmp_path/'trajectories').mkdir()
    world.save(tmp_path/'worlds'/f'{world.id}.npz')
    np.savez_compressed(tmp_path/'trajectories/example.npz', **data)
    return explore.Bundle(tmp_path), world, summary, data


def test_notebook_traces_match_model_measurements_and_preserve_inputs(example):
    bundle, world, summary, data = example
    before = {str(p):p.read_bytes() for p in bundle.path.rglob('*') if p.is_file()}
    recovered, saved = bundle.history('example',world.id)
    for metric, field in [('human','human_service'),('ai','ai_service')]:
        assert bundle.trace(saved,metric,recovered)[2:].mean() == pytest.approx(summary[field])
    assert bundle.trace(saved,'stock',recovered)[2:].mean() == pytest.approx(summary['mean_accessible_resource'])
    assert bundle.trace(saved,'cooperation',recovered)[2:].mean() == pytest.approx(summary['mean_cooperation'])
    assert {str(p):p.read_bytes() for p in bundle.path.rglob('*') if p.is_file()} == before
    assert bundle.history.cache_info().maxsize == 4


def test_notebook_loader_refuses_paths_outside_bundle(example):
    bundle, *_ = example
    with pytest.raises(ValueError,match='inside'):bundle.file('../outside.json')
    with pytest.raises(FileNotFoundError,match='Missing'):bundle.file('absent.npz')
    with pytest.raises(ValueError,match='Unknown'):bundle.trace({},'unknown',None)


def test_notebook_pairing_rejects_mismatched_world_order(example):
    bundle, *_ = example
    first={'policy_id':'one','world_ids':['a','b'],'realization_ids':['1','2']}
    second={'policy_id':'two','world_ids':['b','a'],'realization_ids':['3','4']}
    problem={'pressure':1,'budget':.5,'norm':'L2'}
    bundle._tests={'problems':[{'problem':problem,'families':{'local':[first,first,second]}}]}
    assert len(bundle.designs(problem)) == 2
    with pytest.raises(ValueError,match='matching worlds'):bundle.matched(problem,'one','two')


def test_notebook_uses_configured_activity_utility(example):
    bundle, world, _, data = example
    bundle.config['objective']['welfare']['utility']='log1p'
    mask=~data['agent_is_ai'];ratio=bundle.config['world']['defect_to_cooperate_request_ratio']
    raw=np.minimum(data['agent_delivered'][:,mask]/(ratio*data['agent_reference_request'][mask]),1)
    assert np.allclose(bundle.trace(data,'human',world),(np.log1p(raw)/np.log(2)).mean(axis=1))


def test_public_notebooks_have_clean_outputs_and_valid_python():
    for path in (ROOT/'notebooks').glob('*.ipynb'):
        nb=json.loads(path.read_text())
        assert nb['nbformat']==4 and nb['metadata']['kernelspec']['name']=='python3'
        for cell in nb['cells']:
            source=''.join(cell['source'])
            assert '/Users/' not in source and '/private/tmp' not in source
            if cell['cell_type']=='code':
                assert cell['outputs']==[] and cell['execution_count'] is None
                ast.parse('\n'.join(line for line in source.splitlines() if not line.startswith('%')))
