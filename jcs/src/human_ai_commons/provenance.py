"""Portable content identifiers and deterministic independent random streams."""
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import numpy as np

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1] if (PACKAGE.parents[1]/'pyproject.toml').exists() else Path.cwd()
RNG_VERSION = 'sha256-indexed-PCG64-v1'


def plain(value):
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def canonical(value):
    return json.dumps(plain(value), sort_keys=True, separators=(',', ':'), allow_nan=False)


def identifier(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def seed(*parts):
    return int.from_bytes(hashlib.sha256(canonical([RNG_VERSION, *parts]).encode()).digest()[:16], 'big')


def rng(*parts):
    return np.random.Generator(np.random.PCG64(seed(*parts)))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(canonical(value)+'\n')
        stream.flush()


def read_jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_files():
    # Source hashes cover the executed package, including a non-editable installation.
    paths = {str(Path('src/human_ai_commons')/p.name): p for p in PACKAGE.glob('*.py')}
    paths.update({name: ROOT/name for name in ['pyproject.toml', 'requirements.txt', 'reproduce.py'] if (ROOT/name).exists()})
    return dict(sorted(paths.items()))


def source_hashes():
    return {relative: sha256(path) for relative, path in source_files().items()}


def environment():
    return dict(python=platform.python_version(), platform=platform.platform(),
                dependencies={name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'matplotlib']},
                rng=RNG_VERSION)


def utc_now():
    return datetime.now(timezone.utc).isoformat()
