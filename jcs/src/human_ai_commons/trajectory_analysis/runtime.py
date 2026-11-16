"""Paths supplied by the public analysis command to isolated worker processes."""
import os
from pathlib import Path
P = Path(os.environ.get('COMMONS_BUNDLE', '.')).resolve()
OUT = O = Path(os.environ.get('COMMONS_DERIVED_OUTPUT', '.')).resolve()
WORKERS = max(1, int(os.environ.get('COMMONS_ANALYSIS_WORKERS', '1')))
