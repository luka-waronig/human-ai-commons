"""Trajectory analysis from an immutable saved study bundle."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

STAGES = ('events', 'complete_analysis', 'static_analysis', 'immediate_effects',
          'aggregate_all', 'resource_accounting', 'additional_evidence', 'causal_bands', 'oscillation_controls',
          'local_recurrence', 'sensitivity_tables', 'reference_checks', 'final_verification')


def analyze_trajectories(bundle, destination, workers=8):
    bundle, destination = Path(bundle).resolve(), Path(destination).resolve()
    if destination.is_relative_to(bundle) or bundle.is_relative_to(destination):
        raise ValueError('Analysis output must be separate from the saved bundle')
    if workers < 1:
        raise ValueError('Analysis workers must be positive')
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Full trajectory analysis requires an empty output directory')
    # phase windows, action probabilities and frozen examples define this reported protocol.
    manifest = json.loads((bundle/'manifest.json').read_text())
    expected = 'b9d187ef487b45bd64d14643861a006d185f96f0831215e105854ed9877c13df'
    if manifest['configuration_id'] != expected:
        raise ValueError('Trajectory calculations require the reported research-protocol configuration')
    destination.mkdir(parents=True, exist_ok=True)
    derived = destination / 'derived'
    derived.mkdir()
    env = dict(os.environ, COMMONS_BUNDLE=str(bundle), COMMONS_DERIVED_OUTPUT=str(derived),
               COMMONS_ANALYSIS_WORKERS=str(workers), OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', MKL_NUM_THREADS='1',
               MPLCONFIGDIR=str(destination/'.matplotlib-cache'))
    # the child imports this installed study package even when reproduce.py is used directly.
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get('PYTHONPATH', '')
    package = Path(__file__).resolve().parents[1]
    source_start = {str(path.relative_to(package)):hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(package.rglob('*.py'))}
    started = time.time()
    for i, stage in enumerate(STAGES, 1):
        next_stage = STAGES[i] if i < len(STAGES) else 'plots'
        print(f'Analysis stage {i}/{len(STAGES)}: {stage}; next: {next_stage}', flush=True)
        subprocess.run([sys.executable, '-m', 'human_ai_commons.trajectory_analysis.' + stage], env=env, check=True)
        print(f'Completed {stage}; elapsed {time.time()-started:.0f} seconds', flush=True)
    from ..analysis import study_plots, verify_bundle
    print('Checking all archived artifact hashes before plotting', flush=True)
    verification = verify_bundle(bundle, progress=lambda done,total: print(f'Archived artifacts verified: {done}/{total}', flush=True))
    (derived/'bundle_verification.json').write_text(json.dumps(verification, indent=2))
    os.environ.setdefault('MPLCONFIGDIR', str(destination/'.matplotlib-cache'))
    result = study_plots(bundle, derived, destination / 'plots')
    sources = {}
    package = Path(__file__).resolve().parents[1]
    for path in sorted(package.rglob('*.py')):
        sources[str(path.relative_to(package))] = hashlib.sha256(path.read_bytes()).hexdigest()
    import importlib.metadata
    dependencies={name:importlib.metadata.version(name) for name in ['numpy','scipy','pandas','matplotlib']}
    if sources != source_start:
        raise RuntimeError('Analysis source changed during reproduction; results require a consistent source version')
    result.update(analysis_source_sha256=sources, python_version=sys.version, dependencies=dependencies, elapsed_seconds=time.time()-started,
                  detailed_trajectories=json.loads((derived/'completed_audit.json').read_text())['detailed_trajectories'],
                  analysis_workers=workers)
    (destination/'reproduction_manifest.json').write_text(json.dumps(result, indent=2))
    return result
