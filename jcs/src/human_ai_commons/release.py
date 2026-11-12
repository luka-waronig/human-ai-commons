"""Export of the standalone study and one verified result bundle, without environments."""
import json
from pathlib import Path
import zipfile
from .analysis import verify_bundle
from .provenance import ROOT, sha256, source_files, source_hashes
from .config import robustness_variants


def export_release(bundle, destination):
    bundle, destination = Path(bundle).resolve(), Path(destination).resolve()
    verify_bundle(bundle)
    manifest = json.loads((bundle/'manifest.json').read_text())
    if manifest['source_sha256'] != source_hashes():
        raise ValueError('export requires the source version recorded in this bundle')
    if not (ROOT/'README.md').is_file() or not (ROOT/'configs').is_dir():
        raise ValueError('run export from the self-contained study folder')
    if destination.exists() or destination.is_relative_to(bundle):
        raise ValueError('use a new archive path outside the result bundle')
    required = ['characterize', 'train', 'validate', 'test']
    if robustness_variants(manifest['configuration']):
        required.append('robustness')
    missing = [stage for stage in required if manifest['stages'].get(stage) != 'complete']
    if missing:
        raise ValueError(f'cannot export an incomplete study: {missing}')
    analysis_path = bundle/'analysis/analysis_manifest.json'
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text())
        for relative, expected in analysis['input_sha256'].items():
            path = (bundle/relative).resolve()
            if not path.is_relative_to(bundle) or not path.is_file() or sha256(path) != expected:
                raise ValueError('analysis inputs changed; run analyze again before export')
    files = {}
    for name in ['README.md', 'CITATION.cff', 'pyproject.toml', 'requirements.txt', '.python-version', 'reproduce.py',
                 'run-experiments', 'run_experiments.py',
                 'configs', 'docs', 'src', 'tests', 'scripts', 'verification']:
        source = ROOT/name
        paths = source.rglob('*') if source.is_dir() else [source]
        for path in paths:
            if not path.exists() or not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
                continue
            if path.is_symlink():
                raise ValueError('release files must not be symbolic links')
            files[str(Path('jcs')/path.relative_to(ROOT))] = path
    # A normal installation may execute a wheel rather than ROOT/src. The
    # exported source matches the executed package rather than another copy.
    for relative, path in source_files().items():
        files[str(Path('jcs')/relative)] = path
    for path in bundle.rglob('*'):
        if path.is_file():
            if path.is_symlink():
                raise ValueError('bundle files must not be symbolic links')
            files[str(Path('jcs/results/paper')/path.relative_to(bundle))] = path
    destination.parent.mkdir(parents=True, exist_ok=True)
    inventory = {relative: sha256(path) for relative, path in sorted(files.items())}
    temporary = destination.with_name(destination.name+'.tmp')
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for relative, path in sorted(files.items()):
                archive.write(path, relative)
            archive.writestr('jcs/RELEASE_SHA256.json', json.dumps(inventory, indent=2)+'\n')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(archive=str(destination), files=len(files), sha256=sha256(destination), run_id=manifest['run_id'],
                label=manifest['configuration'].get('execution', {}).get('label', 'research'),
                interpretation='Software and data export; smoke bundles remain software checks.')
