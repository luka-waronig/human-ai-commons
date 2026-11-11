import importlib.util
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import zipfile

import pytest

from watertoc_jcs.config import load_config
from watertoc_jcs.experiments import run_stages
from watertoc_jcs.provenance import ROOT
from watertoc_jcs.analysis import verify_bundle


def launcher():
    spec = importlib.util.spec_from_file_location('experiment_launcher', ROOT/'run_experiments.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_status_never_creates_an_environment_or_starts_a_run(tmp_path):
    output = tmp_path/'not-started'
    result = subprocess.run([sys.executable, str(ROOT/'run_experiments.py'),
                             '--status', '--output', str(output)], capture_output=True, text=True)
    assert result.returncode == 0
    assert 'Not started' in result.stdout
    assert not output.exists()


def test_setup_preserves_the_original_study_and_installs_the_lock_once(tmp_path, monkeypatch):
    module = launcher()
    root = tmp_path/'run'
    root.mkdir()
    invoked = []
    def fake_command(args, *a, **kw):
        invoked.append(args)
        if args[1:3] == ['-m', 'venv']:
            python = Path(args[3])/'bin/python'
            python.parent.mkdir(parents=True)
            python.touch()
    monkeypatch.setattr(module, 'command', fake_command)
    monkeypatch.setattr(module, 'python312', lambda: sys.executable)
    module.prepare(root, True)
    assert '--require-hashes' in invoked[-1]
    original = (root/'study/configs/smoke.json').read_bytes()
    module.prepare(root, True)
    assert len(invoked) == 2
    assert (root/'study/configs/smoke.json').read_bytes() == original
    with pytest.raises(RuntimeError, match='different run mode'):
        module.prepare(root, False)


def test_full_schedule_includes_all_four_studies_after_reference_measurements(tmp_path, monkeypatch):
    module = launcher()
    (tmp_path/'.software-checked').touch()
    calls = []
    monkeypatch.setattr(module.shutil, 'which', lambda _: None)
    monkeypatch.setattr(module, 'study', lambda root, python, name, config: calls.append(('study', name, config)))
    monkeypatch.setattr(module, 'plan', lambda *args: {})
    monkeypatch.setattr(module, 'stage', lambda *args: calls.append(('reference',)))
    monkeypatch.setattr(module, 'cli', lambda *args: calls.append(('reference-analysis',)))
    assert module.worker(tmp_path, False, 4) == 0
    assert calls[:3] == [('study', 'software-check', 'smoke'), ('reference',), ('reference-analysis',)]
    assert set(calls[3:]) == {('study', name, config) for name, config in module.STUDIES}


def test_launcher_completes_and_resumes_a_tiny_study_without_new_simulations(tmp_path):
    module = launcher()
    root = tmp_path/'run'
    root.mkdir()
    shutil.copytree(ROOT, root/'study', ignore=shutil.ignore_patterns(
        '.venv', 'results', '__pycache__', '.pytest_cache', '*.egg-info'))
    (root/'.venv').symlink_to(sys.prefix, target_is_directory=True)
    module.write(root/'run.json', dict(mode='software-check'))
    module.write(root/'.environment-ready.json', dict(requirements=module.digest(root/'study/requirements.txt')))
    # The enclosing pytest run checks the software; avoid recursively launching pytest.
    (root/'.software-checked').touch()
    args = [sys.executable, str(ROOT/'run_experiments.py'), '--smoke', '--jobs', '1', '--output', str(root)]
    first = subprocess.run(args, capture_output=True, text=True, timeout=90)
    assert first.returncode == 0, first.stdout+first.stderr
    assert (root/'releases/software-check.zip').exists()
    with zipfile.ZipFile(root/'releases/software-check.zip') as archive:
        assert 'jcs/run-experiments' in archive.namelist()
        assert 'jcs/run_experiments.py' in archive.namelist()
    assert (root/'bundles/software-check/analysis/heldout_frontier.csv').exists()
    ledger = root/'bundles/software-check/budget_ledger.jsonl'
    before = ledger.read_bytes()
    second = subprocess.run(args, capture_output=True, text=True, timeout=90)
    assert second.returncode == 0, second.stdout+second.stderr
    assert ledger.read_bytes() == before
    assert 'already complete' in second.stdout
    assert len(list((root/'releases').glob('*.zip'))) == 1


def test_ctrl_c_saves_a_complete_realization_and_allows_verified_resume(tmp_path, monkeypatch):
    import watertoc_jcs.experiments as experiments
    config = load_config(ROOT/'configs/smoke.json')
    simulate = experiments.simulate
    requested = False
    def interrupted(*args, **kwargs):
        nonlocal requested
        if not requested:
            requested = True
            signal.raise_signal(signal.SIGINT)
        return simulate(*args, **kwargs)
    original_handler = signal.getsignal(signal.SIGINT)
    monkeypatch.setattr(experiments, 'simulate', interrupted)
    with pytest.raises(KeyboardInterrupt, match='saving'):
        run_stages(config, tmp_path/'bundle', 'characterize', 10000)
    assert signal.getsignal(signal.SIGINT) == original_handler
    assert verify_bundle(tmp_path/'bundle')['verified']
    monkeypatch.setattr(experiments, 'simulate', simulate)
    result = run_stages(config, tmp_path/'bundle', 'all', 10000)
    assert set(result['stages'].values()) == {'complete'}
    assert verify_bundle(tmp_path/'bundle')['verified']
