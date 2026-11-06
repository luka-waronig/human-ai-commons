import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from water_toc.reproducibility import stable_seed, run_identifier, write_manifest


def test_stable_identifiers_are_canonical_and_process_independent():
    assert stable_seed(2.,np.int64(3)) == stable_seed(2,3.)
    assert run_identifier({'b':2,'a':1}) == run_identifier({'a':1.,'b':2.})
    command = [sys.executable,'-c','from water_toc.reproducibility import stable_seed; print(stable_seed("test",2.))']
    assert int(subprocess.check_output(command,text=True,cwd=Path(__file__).resolve().parents[1])) == stable_seed('test',2)
    with pytest.raises(ValueError):
        stable_seed(np.nan)


def test_manifest_records_input_and_current_sources(tmp_path):
    source = tmp_path/'input.csv'; source.write_text('x\n1\n')
    result = write_manifest(tmp_path/'manifest.json',{'seed':42},input_path=source)
    assert result['dependencies']['Mesa'] == '1.2.1'
    assert 'water_toc/model.py' in result['source_sha256']
    assert len(result['input']['sha256']) == 64
    assert json.loads((tmp_path/'manifest.json').read_text()) == result
