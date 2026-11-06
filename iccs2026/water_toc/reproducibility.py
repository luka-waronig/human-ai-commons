"""Stable identifiers and explicit provenance for regenerated experiments."""

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from datetime import datetime, timezone
from numbers import Integral, Real
from pathlib import Path


def _normalize(value):
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("identifiers cannot contain non-finite numbers")
        return int(number) if number.is_integer() else number
    return value


def canonical_json(value):
    return json.dumps(_normalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_seed(*parts, base_seed=42):
    """A process-independent seed; equivalent integer/float inputs agree."""
    digest = hashlib.sha256(canonical_json([base_seed, *parts]).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def run_identifier(config):
    return hashlib.sha256(canonical_json(config).encode()).hexdigest()[:24]


def model_fingerprint():
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ['model.py', 'agents.py', 'feedback.py']:
        digest.update(name.encode())
        digest.update((package / name).read_bytes())
    return digest.hexdigest()


def write_manifest(path, config, *, input_path=None):
    """Write parameters, dependency versions, source hashes, and optional input hash.

    The executed source hashes identify the implementation independently of its
    Git revision.
    """
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    versions = {}
    for name in ["Mesa", "numpy", "pandas", "scipy", "SALib", "statsmodels", "PyWavelets"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    sources = sorted(root.glob("*.py")) + sorted((root / "water_toc").glob("*.py")) + sorted((root / "plotting").glob("*.py"))
    sources += [root / name for name in ['requirements.txt', 'pyproject.toml'] if (root / name).exists()]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": revision,
        "python": platform.python_version(),
        "dependencies": versions,
        "configuration": config,
        "source_sha256": {
            str(source.relative_to(root)): hashlib.sha256(source.read_bytes()).hexdigest()
            for source in sources
        },
    }
    if input_path is not None:
        source = Path(input_path)
        manifest["input"] = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest
