"""Run from this folder, including a copy outside the WaterToC repository."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from watertoc_jcs.cli import main

if __name__ == '__main__':
    main()
