"""Capture fixed-commit reference trajectories for software regression tests.

Run with the pinned environment: python reproducibility/capture_baseline.py
The reference commit and source checksums identify the regression baseline.
"""
import json
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = '3c5fdc918b7f4a22dc6d9c20b460878134bfed2a'
CASES = [
    dict(seed=42, steps=30),
    dict(seed=17, steps=25, height=8, width=12, initial_humans=10, initial_ai=8),
    dict(seed=5, steps=3, height=4, width=5, initial_humans=0, initial_ai=0),
    dict(seed=3, steps=4, height=4, width=5, initial_humans=3, initial_ai=2, water_cell_density=0),
]
PROGRAM = '''
import json, sys
from water_toc.model import WaterToC
cases = json.loads(sys.argv[1])
records = []
for case in cases:
    params = {k:v for k,v in case.items() if k != 'steps'}
    model = WaterToC(**params, save_snapshots=False)
    for _ in range(case['steps']):
        model.step()
    columns = ['Total_Water','Environment_State','Coop_Fraction','Human_Coop_Fraction','AI_Coop_Fraction']
    records.append(dict(case=case, columns=columns,
        trajectory=model.datacollector.get_model_vars_dataframe()[columns].to_numpy().tolist(),
        water=model.water_levels.tolist(), agents=model._get_agent_pos_strategies_list()))
print(json.dumps(records))
'''


def main():
    with tempfile.TemporaryDirectory(prefix='watertoc-baseline-') as temp:
        package = Path(temp) / 'water_toc'
        package.mkdir()
        (package / '__init__.py').write_text('')
        for name in ['model.py', 'agents.py']:
            source = subprocess.check_output(['git', 'show', f'{REFERENCE}:water_toc/{name}'], cwd=ROOT)
            (package / name).write_bytes(source)
        output = subprocess.check_output([sys.executable, '-c', PROGRAM, json.dumps(CASES)],
                                         cwd=temp, env={**os.environ, 'PYTHONPATH': temp}, text=True)
    artifact = {'reference_commit': REFERENCE, 'publication_provenance_verified': False,
                'python': sys.version,
                'dependencies': {name:importlib.metadata.version(name) for name in ['Mesa','numpy','pandas','pydantic']},
                'cases': json.loads(output)}
    target = ROOT / 'reproducibility' / 'baseline_reference.json'
    target.write_text(json.dumps(artifact, indent=2) + '\n')
    print(f'Captured {len(CASES)} historical reference trajectories: {target}')


if __name__ == '__main__':
    main()
