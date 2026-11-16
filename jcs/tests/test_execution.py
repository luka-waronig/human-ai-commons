from pathlib import Path
import sys,json,shutil,subprocess,signal
import pytest
from human_ai_commons.provenance import ROOT,read_jsonl
from human_ai_commons.config import load_config
from human_ai_commons.analysis import verify_bundle

import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import human_ai_commons.experiments as ex
from human_ai_commons.payoffs import Policy
from human_ai_commons.provenance import read_jsonl


def parallel_launcher_module():
 from human_ai_commons import execution
 return execution

def attach_parallel(ctx,pool,workers):
 ctx.parallel_pool=pool;ctx.parallel_workers=workers;ctx.serial_aggregate=ctx.aggregate

def test_recorded_results_and_variant_are_exact(tmp_path):
 config=load_config(ROOT/'configs/smoke.json')
 serial=ex.Evaluation(config,tmp_path/'serial',10000);parallel=ex.Evaluation(config,tmp_path/'parallel',10000)
 serial.output.mkdir();parallel.output.mkdir()
 for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
 with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),initializer=parallel_launcher_module().simulation_worker_init) as pool:
  attach_parallel(parallel,pool,2)
  for variant in (None,dict(name='changed',scenario=dict(theta=1),policy=dict(observation='fixed'),steps=12,burn=3,initial_fraction=.25)):
   expected=serial.aggregate(1,Policy(), 'test',3,record=True,variant=variant)
   actual=parallel_launcher_module().parallel_aggregate(parallel,1,Policy(),'test',3,record=True,variant=variant)
   assert actual==expected
  assert read_jsonl(serial.output/'realizations.jsonl')==read_jsonl(parallel.output/'realizations.jsonl')
  for folder in ('trajectories','drivers'):
   for path in (serial.output/folder).glob('*.npz'):
    with np.load(path) as a,np.load(parallel.output/folder/path.name) as b:
     assert a.files==b.files
     assert all(np.array_equal(a[k],b[k]) for k in a.files)
  before=(parallel.output/'budget_ledger.jsonl').read_bytes()
  parallel_launcher_module().parallel_aggregate(parallel,1,Policy(),'test',3,record=True)
  assert (parallel.output/'budget_ledger.jsonl').read_bytes()==before

def test_stop_drains_batch_and_resumes_without_repeating(tmp_path,monkeypatch):
 config=load_config(ROOT/'configs/smoke.json');ctx=ex.Evaluation(config,tmp_path,10000)
 from human_ai_commons import provenance
 append=provenance.append_json
 def stopping(path,row):
  append(path,row)
  if Path(path).name=='realizations.jsonl':ctx.stop_requested=True
 monkeypatch.setattr(provenance,'append_json',stopping)
 with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),initializer=parallel_launcher_module().simulation_worker_init) as pool:
  attach_parallel(ctx,pool,2)
  with pytest.raises(KeyboardInterrupt):parallel_launcher_module().parallel_aggregate(ctx,1,Policy(),'test',5,record=True)
  saved=read_jsonl(tmp_path/'realizations.jsonl');assert len(saved)==2
  assert len([r for r in read_jsonl(tmp_path/'budget_ledger.jsonl') if r['status']=='complete'])==2
  ctx.stop_requested=False;monkeypatch.setattr(provenance,'append_json',append)
  parallel_launcher_module().parallel_aggregate(ctx,1,Policy(),'test',5,record=True)
  rows=read_jsonl(tmp_path/'realizations.jsonl');assert len(rows)==5
  assert len({r['evaluation_id'] for r in rows})==5

def test_budget_guard_starts_no_simulations(tmp_path):
 ctx=ex.Evaluation(load_config(ROOT/'configs/smoke.json'),tmp_path,0)
 attach_parallel(ctx,None,2)
 with pytest.raises(RuntimeError,match='budget exhausted'):parallel_launcher_module().parallel_aggregate(ctx,1,Policy(),'test',3)
 assert not (tmp_path/'budget_ledger.jsonl').exists()

def test_parallel_smoke_subprocess_and_serial_resume(tmp_path):
 root=tmp_path/'study'
 shutil.copytree(ROOT,root,ignore=shutil.ignore_patterns('results','.venv','__pycache__','.pytest_cache','*.egg-info'))
 command=[sys.executable,str(root/'reproduce.py'),'smoke','--workers','2']
 result=subprocess.run(command,capture_output=True,text=True,timeout=120)
 assert result.returncode==0,result.stdout+result.stderr
 output=root/'results/smoke';verify_bundle(output)
 assert all(v=='complete' for v in json.loads((output/'manifest.json').read_text())['stages'].values())
 before=(output/'budget_ledger.jsonl').read_bytes()
 again=subprocess.run(command,capture_output=True,text=True,timeout=120)
 assert again.returncode==0,again.stdout+again.stderr
 assert before==(output/'budget_ledger.jsonl').read_bytes()
 resumed=subprocess.run([sys.executable,str(root/'reproduce.py'),'smoke'],capture_output=True,text=True,timeout=120)
 assert resumed.returncode==0,resumed.stdout+resumed.stderr
 assert before==(output/'budget_ledger.jsonl').read_bytes()

def test_live_bundle_is_refused_without_writes(tmp_path,monkeypatch):
 output=tmp_path/'results/smoke';output.mkdir(parents=True)
 (output/'manifest.json').write_text(json.dumps(dict(stages=dict(train='running'))))
 before=(output/'manifest.json').read_bytes();monkeypatch.setattr(parallel_launcher_module(),'ROOT',tmp_path)
 with pytest.raises(RuntimeError,match='running stage'):parallel_launcher_module().run_config(load_config(ROOT/'configs/smoke.json'),output,'all',10000,2)
 assert list(output.iterdir())==[output/'manifest.json']
 assert (output/'manifest.json').read_bytes()==before

def test_serial_checkpoint_continues_in_parallel_with_identical_results(tmp_path):
 roots=[]
 for name in ('serial','continued'):
  root=tmp_path/name
  shutil.copytree(ROOT,root,ignore=shutil.ignore_patterns('results','.venv','__pycache__','.pytest_cache','*.egg-info'))
  roots.append(root)
 serial,continued=roots
 result=subprocess.run([sys.executable,str(serial/'reproduce.py'),'smoke','--workers','1'],capture_output=True,text=True,timeout=120)
 assert result.returncode==0,result.stdout+result.stderr
 result=subprocess.run([sys.executable,str(continued/'reproduce.py'),'run','--config',str(continued/'configs/smoke.json'),'--output',str(continued/'results/smoke'),'--stage','characterize','--max-model-steps','10000'],capture_output=True,text=True,timeout=120)
 assert result.returncode==0,result.stdout+result.stderr
 original=read_jsonl(continued/'results/smoke/realizations.jsonl')
 # a changed entry point exercises archived-source resumption.
 entry=continued/'reproduce.py';entry.write_text(entry.read_text()+'\n# Later interface version.\n')
 result=subprocess.run([sys.executable,str(continued/'reproduce.py'),'smoke','--workers','2'],capture_output=True,text=True,timeout=120)
 assert result.returncode==0,result.stdout+result.stderr
 a=read_jsonl(serial/'results/smoke/realizations.jsonl');b=read_jsonl(continued/'results/smoke/realizations.jsonl')
 assert a==b
 assert 'verified archived model source' in result.stdout
 assert b[:len(original)]==original
 replay=subprocess.run([sys.executable,str(continued/'reproduce.py'),'replay',str(continued/'results/smoke'),b[0]['evaluation_id'],'--max-model-steps','10000'],capture_output=True,text=True,timeout=30)
 assert replay.returncode==0,replay.stdout+replay.stderr
 assert 'verified archived model source' in replay.stderr
 for name in ('shortlists.json','frozen_policies.json','test_comparisons.json','robustness.json'):
  assert json.loads((serial/'results/smoke'/name).read_text())==json.loads((continued/'results/smoke'/name).read_text())

def test_real_sigint_saves_submitted_batch_and_can_resume(tmp_path):
 import time
 root=tmp_path/'study'
 shutil.copytree(ROOT,root,ignore=shutil.ignore_patterns('results','.venv','__pycache__','.pytest_cache','*.egg-info'))
 cfg=root/'configs/smoke.json';config=json.loads(cfg.read_text())
 config['objective']['total_steps']=300;config['objective']['excluded_initial_steps']=50
 cfg.write_text(json.dumps(config))
 cmd=[sys.executable,str(root/'reproduce.py'),'smoke','--workers','2']
 process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 ledger=root/'results/smoke/budget_ledger.jsonl'
 deadline=time.monotonic()+30
 try:
  while not ledger.exists() and process.poll() is None and time.monotonic()<deadline:time.sleep(.03)
  assert ledger.exists()
  process.send_signal(signal.SIGINT)
  out,err=process.communicate(timeout=30)
  manifest=json.loads((root/'results/smoke/manifest.json').read_text())
  assert 'Stopped after saving' in manifest.get('last_error',''),out+err
  verify_bundle(root/'results/smoke')
  rows=read_jsonl(ledger)
  assert {r['evaluation_id'] for r in rows if r['status']=='reserved'}=={r['evaluation_id'] for r in rows if r['status']=='complete'}
 finally:
  if process.poll() is None:process.kill();process.communicate()


def test_existing_nonbundle_files_are_never_adopted(tmp_path):
    from human_ai_commons.execution import run_config
    marker=tmp_path/'existing';marker.write_text('retained')
    with pytest.raises(ValueError,match='empty'):
        run_config(load_config(ROOT/'configs/smoke.json'),tmp_path,'characterize',10000,2)
    assert list(tmp_path.iterdir())==[marker]
    assert marker.read_text()=='retained'


def test_tampered_saved_source_is_refused(tmp_path):
    from human_ai_commons.execution import saved_source
    from human_ai_commons.provenance import sha256
    folder=tmp_path/'source_snapshot';folder.mkdir();entry=folder/'reproduce.py';entry.write_text('original')
    manifest={'source_sha256':{'reproduce.py':sha256(entry)}}
    assert saved_source(tmp_path,manifest)==folder
    entry.write_text('changed')
    with pytest.raises(ValueError,match='changed'):saved_source(tmp_path,manifest)


def test_command_help_exposes_search_and_regular_analysis():
    for command in ['run','analyze']:
        result=subprocess.run([sys.executable,str(ROOT/'reproduce.py'),command,'--help'],capture_output=True,text=True,check=True)
        assert '--workers' in result.stdout
        assert '--publication' not in result.stdout
        if command=='run':
            assert 'search' in result.stdout and 'train' not in result.stdout


def test_archived_resume_interrupt_waits_for_a_saved_checkpoint(tmp_path):
 import time
 root=tmp_path/'study'
 shutil.copytree(ROOT,root,ignore=shutil.ignore_patterns('results','.venv','__pycache__','.pytest_cache','*.egg-info'))
 cfg=root/'configs/smoke.json';config=json.loads(cfg.read_text())
 config['objective']['total_steps']=300;config['objective']['excluded_initial_steps']=50
 cfg.write_text(json.dumps(config));output=root/'results/smoke'
 base=[sys.executable,str(root/'reproduce.py'),'run','--config',str(cfg),'--output',str(output),'--workers','2','--max-model-steps','1000000']
 first=subprocess.run(base+['--stage','characterize'],capture_output=True,text=True,timeout=30)
 assert first.returncode==0,first.stdout+first.stderr
 entry=root/'reproduce.py';entry.write_text(entry.read_text()+'\n# Later interface version.\n')
 ledger=output/'budget_ledger.jsonl';initial=ledger.stat().st_size
 process=subprocess.Popen(base+['--stage','all'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 try:
  deadline=time.monotonic()+30
  while ledger.stat().st_size==initial and process.poll() is None and time.monotonic()<deadline:time.sleep(.03)
  assert ledger.stat().st_size>initial
  process.send_signal(signal.SIGINT)
  out,err=process.communicate(timeout=30)
  assert process.returncode==130,out+err
  manifest=json.loads((output/'manifest.json').read_text())
  assert 'Stopped after saving' in manifest['last_error']
  rows=read_jsonl(ledger)
  assert {r['evaluation_id'] for r in rows if r['status']=='reserved'}=={r['evaluation_id'] for r in rows if r['status']=='complete'}
  verify_bundle(output)
 finally:
  if process.poll() is None:process.kill();process.communicate()
