import ast
import json
from pathlib import Path
import os
import shutil
import subprocess
import sys
import zipfile
import runpy
import numpy as np
import pytest
from human_ai_commons.analysis import analyze, metric_sensitivity, verify_bundle
from human_ai_commons.config import budget_plan, load_config
import human_ai_commons.experiments as experiments
from human_ai_commons.experiments import replay, run_stages
from human_ai_commons.provenance import ROOT, read_jsonl


def test_production_budget_guard_precedes_any_output(tmp_path):
    config = load_config(ROOT/'configs/research_protocol.json')
    assert budget_plan(config)['all_model_steps'] > 10000
    with pytest.raises(ValueError, match='no simulations started'):
        run_stages(config, tmp_path/'production', 'train', 10000)
    assert not (tmp_path/'production').exists()


def test_reference_characterization_can_be_inspected_before_any_optimization(tmp_path,monkeypatch):
    config=load_config(ROOT/'configs/smoke.json')
    def unexpected(*args,**kwargs):
        raise AssertionError('characterization must not optimize')
    monkeypatch.setattr(experiments,'differential_search',unexpected)
    output=tmp_path/'reference'
    manifest=run_stages(config,output,'characterize',10000)
    assert manifest['stages']=={'characterize':'complete'}
    assert (output/'reference_characterization.json').exists()
    assert not (output/'training_candidates.jsonl').exists()
    analyze(output)
    assert (output/'analysis/spatial_characterization.csv').exists()


@pytest.mark.parametrize('basis', ['resource_stock', 'request_fulfillment'])
def test_smoke_stages_holdouts_integrity_and_exact_replay(tmp_path, basis):
    config = load_config(ROOT/'configs/smoke.json')
    config['objective']['persistent_local_stress']['basis'] = basis
    if basis == 'request_fulfillment':
        config['objective']['persistent_local_stress']['reference'] = 1
    output = tmp_path/'smoke'
    run_stages(config,output,'characterize',10000)
    analyze(output)
    manifest = run_stages(config, output, 'all', 10000)
    assert set(manifest['stages'].values()) == {'complete'}
    assert manifest['model_steps_logged'] <= budget_plan(config)['all_model_steps']
    assert verify_bundle(output)['verified']
    # Completed stages are reused without charging any additional simulations.
    before = (output/'budget_ledger.jsonl').read_text()
    run_stages(config, output, 'all', 1)
    assert (output/'budget_ledger.jsonl').read_text() == before
    rows = read_jsonl(output/'realizations.jsonl')
    splits = {name: {r['world_id'] for r in rows if r['split'] == name} for name in ['characterization', 'train', 'validation', 'test']}
    assert splits['characterization'].isdisjoint(splits['train'] | splits['validation'] | splits['test'])
    assert splits['train'].isdisjoint(splits['validation'] | splits['test'])
    assert splits['validation'].isdisjoint(splits['test'])
    selected = next(r for r in rows if r['split'] == 'test' and 'trajectory_path' in r)
    assert all('welfare' in row and 'persistent_local_stress' in row for row in rows)
    assert replay(output, selected['evaluation_id'])['matched']
    frozen = json.loads((output/'frozen_policies.json').read_text())
    assert all('winners' not in problem for problem in frozen)
    assert all(candidate['count'] == 2 for problem in frozen for candidates in problem['families'].values() for candidate in candidates)
    analyze(output)
    assert (output/'analysis/pareto_frontier.png').exists()
    assert (output/'analysis/pareto_frontier.pdf').exists()
    # The software check cannot establish the production probability limit.
    # Its supported-frontier table must retain its header without false claims.
    assert len((output/'analysis/pareto_frontier.csv').read_text().splitlines()) == 1
    assert (output/'analysis/spatial_depletion.png').exists()
    assert (output/'analysis/observation_controls.png').exists()
    assert (output/'analysis/robustness.png').exists()
    assert (output/'analysis/spatial_stress.png').exists()
    assert (output/'analysis/welfare_assumptions.csv').exists()
    assert (output/'analysis/stress_definitions.csv').exists()
    comparisons = json.loads((output/'test_comparisons.json').read_text())
    assert comparisons['mode'] == 'pareto'
    assert comparisons['paired_outcome_comparisons']
    assert all('simultaneous_protection_supported' in value for row in comparisons['problems'] for candidates in row['families'].values() for value in candidates)
    assert not any(value['simultaneous_protection_supported'] for row in comparisons['problems'] for candidates in row['families'].values() for value in candidates)
    assert all('reference' in row['families'] for row in comparisons['problems'])
    result = metric_sensitivity(output, .3, .2, output/'analysis/rescored.csv')
    assert result['rescored_realizations'] > 0
    assert not result['simulations_started']
    from human_ai_commons.release import export_release
    exported = export_release(output, tmp_path/'release.zip')
    with zipfile.ZipFile(exported['archive']) as archive:
        names = archive.namelist()
        assert 'jcs/src/human_ai_commons/dynamics.py' in names
        assert 'jcs/results/paper/frozen_policies.json' in names
        assert 'jcs/docs/research-plan.md' in names
        assert 'jcs/scripts/verify_payoff_design.py' in names
        assert 'jcs/RELEASE_SHA256.json' in names
        assert 'jcs/RESEARCH_PLAN.md' not in names
        assert not any('/.venv/' in name or '/iccs2026/' in name for name in names)
    with (output/selected['world_path']).open('ab') as stream:
        stream.write(b'tamper')
    with pytest.raises(ValueError, match='checksum'):
        verify_bundle(output)


def test_package_imports_do_not_reference_legacy_modules():
    prohibited = {'mesa', 'sensitivity_analysis', 'server', 'spatial_analysis_sweep'}
    for path in (ROOT/'src').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                assert all(alias.name.split('.')[0] not in prohibited for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                assert node.module.split('.')[0] not in prohibited


def test_copied_folder_works_outside_repository_in_isolated_python(tmp_path):
    copied = tmp_path/'standalone-jcs'
    shutil.copytree(ROOT, copied, ignore=shutil.ignore_patterns('results', '.venv', '__pycache__', '.pytest_cache', '*.egg-info'))
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, '-I', str(copied/'reproduce.py'), 'plan', '--config', str(copied/'configs/smoke.json')],
                            cwd=tmp_path, env=environment, text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)['simulations_started'] is False


def test_direct_python_experiment_runs_and_resumes_outside_repository(tmp_path):
    copied = tmp_path/'journal'
    shutil.copytree(ROOT, copied, ignore=shutil.ignore_patterns('results', '.venv', '__pycache__', '.pytest_cache', '*.egg-info'))
    command = [sys.executable, '-I', str(copied/'reproduce.py'), 'smoke']
    first = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)
    assert 'Finished:' in first.stdout
    output = copied/'results/smoke'
    assert (output/'analysis/pareto_candidates.csv').exists()
    manifest = json.loads((output/'manifest.json').read_text())
    assert set(manifest['stages'].values()) == {'complete'}
    before = (output/'budget_ledger.jsonl').read_bytes()
    subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)
    assert (output/'budget_ledger.jsonl').read_bytes() == before


def test_direct_all_dispatch_uses_all_configured_stages_and_budgets_without_simulating(monkeypatch):
    import human_ai_commons.config as configuration
    import human_ai_commons.analysis as analysis
    calls = []
    def record(config, output, stage, limit):
        expected = configuration.budget_plan(config)['model_step_caps'][stage]
        assert limit == expected
        calls.append((output.name, stage))
    monkeypatch.setattr(experiments, 'run_stages', record)
    monkeypatch.setattr(analysis, 'verify_bundle', lambda *args: None)
    monkeypatch.setattr(analysis, 'analyze', lambda *args: None)
    monkeypatch.setattr(analysis, 'metric_sensitivity', lambda *args: None)
    monkeypatch.setattr(sys, 'argv', ['reproduce.py', 'all'])
    runpy.run_path(str(ROOT/'reproduce.py'), run_name='__main__')
    expected = []
    for name in ['research_protocol', 'human_only_priority', 'ai_only_priority', 'delivery_reliability']:
        config = configuration.load_config(ROOT/'configs'/f'{name}.json')
        expected.extend((name, stage) for stage, cap in configuration.budget_plan(config)['model_step_caps'].items() if cap)
    assert calls == expected
