"""CPU execution and progress for the study; simulation equations remain separate."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .provenance import ROOT

LABELS = {'characterize':'Original-game measurements','characterization':'Original-game measurements',
          'train':'Payoff search','validate':'Design selection','validation':'Design selection',
          'test':'Independent tests','robustness':'Robustness checks'}


def report_progress(ctx, pressure, policy, split, count, variant, finished=False):
    now = time.monotonic()
    if not finished and now-getattr(ctx,'last_progress',0)<2:
        return
    ctx.last_progress=now
    variant_name=(variant or {}).get('name','reference')
    # progress counts saved evaluations rather than the adaptive search workload.
    if not hasattr(ctx,'progress_cases'):
        ctx.progress_cases=Counter((r['split'],r['scenario']['pressure'],r['policy_id'],r['variant']) for r in ctx.cache.values())
        ctx.progress_seen=set(ctx.cache)
    for eid in ctx.cache.keys()-ctx.progress_seen:
        r=ctx.cache[eid];ctx.progress_cases[(r['split'],r['scenario']['pressure'],r['policy_id'],r['variant'])]+=1
        ctx.progress_seen.add(eid)
    from dataclasses import replace
    candidate=replace(policy,**(variant or {}).get('policy',{}))
    key=(split,pressure,candidate.id,variant_name)
    done=ctx.progress_cases[key]
    complete=sum(v>=count for k,v in ctx.progress_cases.items() if k[0]==split)
    print(f'{LABELS.get(split,split)}: current payoff {done}/{count} saved simulations; '
          f'{complete} completed payoff evaluations in this stage; '
          f'{len(ctx.cache)} saved simulations in the bundle',flush=True)

def simulation_worker_init():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def simulation_worker(task):
    from human_ai_commons.dynamics import simulate
    return simulate(*task[0], **task[1])


def parallel_aggregate(ctx, pressure, policy, split, count, *, variant=None, record=False):
    """Independent simulations in bounded batches; all bundle writes stay in the parent."""
    from dataclasses import asdict, replace
    import numpy as np
    from human_ai_commons import experiments as ex
    from human_ai_commons.provenance import append_json, identifier, seed
    variant = variant or {}
    for start in range(0, count, ctx.parallel_workers):
        ctx.check_stop()
        pending = []
        for index in range(start, min(start+ctx.parallel_workers, count)):
            world = ctx.world(split, index, variant)
            physical = replace(ex.scenario(ctx.config, pressure), **variant.get('scenario', {}))
            candidate = replace(policy, **variant.get('policy', {}))
            steps = variant.get('steps', ctx.config['objective']['total_steps'])
            burn = variant.get('burn', ctx.config['objective']['excluded_initial_steps'])
            metric, welfare, stress = ex.metric_spec(ctx.config), ex.welfare_spec(ctx.config), ex.stress_spec(ctx.config)
            driver_seed = seed(ctx.config['validation']['base_seed'], split, index, 'trajectory')
            inputs = dict(world_id=world.id, scenario=asdict(physical), policy=asdict(candidate),
                driver_seed=driver_seed, steps=steps, burn=burn, metric=asdict(metric),
                welfare_spec=asdict(welfare) if welfare else None, stress_spec=asdict(stress) if stress else None)
            evaluation_id = identifier(inputs)[:32]
            trajectory = ctx.output/'trajectories'/f'{evaluation_id}.npz'
            if evaluation_id in ctx.cache and (not record or trajectory.exists()):
                continue
            task = ((world, physical, candidate, driver_seed, steps, burn, metric),
                    dict(record=record, welfare=welfare, stress=stress))
            pending.append((index, world, inputs, evaluation_id, trajectory, task))
        needed = sum(item[2]['steps'] for item in pending)
        if ctx.used+needed > ctx.maximum_steps:
            raise RuntimeError(f'model-step budget exhausted: {ctx.used}+{needed}>{ctx.maximum_steps}')
        ctx.check_stop()
        futures = []
        for item in pending:
            ctx.used += item[2]['steps']
            append_json(ctx.output/'budget_ledger.jsonl', dict(evaluation_id=item[3], steps=item[2]['steps'],
                        split=split, status='reserved', invocation_steps_used=ctx.used))
            futures.append((item, ctx.parallel_pool.submit(simulation_worker, item[5])))
        failure = None
        for (index, world, inputs, evaluation_id, trajectory, _), future in futures:
            try:
                summary, data = future.result()
            except Exception as error:
                append_json(ctx.output/'budget_ledger.jsonl', dict(evaluation_id=evaluation_id, steps=0, status='failed'))
                failure = failure or error
                continue
            row = inputs | summary | dict(evaluation_id=evaluation_id, split=split, index=index,
                variant=variant.get('name', 'reference'), world_path=f'worlds/{world.id}.npz')
            if record:
                trajectory.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(trajectory, **data)
                driver_id = identifier([inputs['driver_seed'], inputs['steps'], len(world.positions)])[:24]
                driver_path = ctx.output/'drivers'/f'{driver_id}.npz'
                driver_path.parent.mkdir(parents=True, exist_ok=True)
                if not driver_path.exists():
                    ex.Drivers(inputs['driver_seed'], len(world.positions)).archive(driver_path, inputs['steps'])
                row.update(trajectory_path=str(trajectory.relative_to(ctx.output)),
                           drivers_path=str(driver_path.relative_to(ctx.output)))
            append_json(ctx.output/'realizations.jsonl', row)
            ctx.cache[evaluation_id] = row
            append_json(ctx.output/'budget_ledger.jsonl', dict(evaluation_id=evaluation_id, steps=0,
                        completed_steps=inputs['steps'], status='complete'))
        if failure is not None:
            raise failure
        report_progress(ctx, pressure, policy, split, count, variant)
        ctx.check_stop()
    report_progress(ctx, pressure, policy, split, count, variant, finished=True)
    return ctx.serial_aggregate(pressure, policy, split, count, variant=variant, record=record)


@contextmanager
def parallel_execution(workers):
    from . import experiments as ex
    original=ex.Evaluation.aggregate
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[key]='1'
    try:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                                 initializer=simulation_worker_init) as pool:
            ex.Evaluation.serial_aggregate=original
            ex.Evaluation.parallel_pool=pool
            ex.Evaluation.parallel_workers=workers
            ex.Evaluation.aggregate=parallel_aggregate
            yield
    finally:
        ex.Evaluation.aggregate=original
        for name in ('serial_aggregate','parallel_pool','parallel_workers'):
            if hasattr(ex.Evaluation,name):delattr(ex.Evaluation,name)


def saved_source(bundle, manifest):
    """Verify archived source checksums for compatible resumption."""
    from .provenance import sha256
    source=bundle/'source_snapshot'
    for name,expected in manifest['source_sha256'].items():
        f=(source/name).resolve()
        if not f.is_relative_to(source.resolve()) or not f.is_file() or sha256(f)!=expected:
            raise ValueError('The archived source is missing or changed: '+name)
    return source


def run_config(config, output, stage='all', maximum_steps=10000, workers=8):
    from . import experiments as ex
    from .config import budget_plan
    from .provenance import source_hashes,environment,write_json,sha256
    from .analysis import verify_bundle
    output=Path(output).resolve()
    if workers<1:raise ValueError('workers must be positive')
    if maximum_steps<1:raise ValueError('positive model-step limit required')
    # this guard precedes any directory or process creation.
    expanded=ex.STAGES if stage=='all' else (['characterize','train'] if stage=='train' else [stage])
    limits=budget_plan(config)['model_step_caps']
    expanded=[s for s in expanded if limits[s]]
    manifest=json.loads((output/'manifest.json').read_text()) if (output/'manifest.json').exists() else None
    if manifest:
        if any(v=='running' for v in manifest['stages'].values()) and not manifest.get('last_error'):
            raise RuntimeError('The bundle has a running stage. Stop its experiment terminal before resuming.')
        from .provenance import identifier
        if manifest['configuration_id']!=identifier(config):raise ValueError('configuration changed; start a new bundle')
        if manifest['environment']['dependencies']!=environment()['dependencies']:
            raise ValueError('dependency versions changed; use the recorded environment')
        if (output/'checksums.json').exists():verify_bundle(output)
        if manifest.get('last_error'):print('Resuming saved interruption; completed simulations will be reused.',flush=True)
        if all(manifest['stages'].get(s)=='complete' for s in expanded):
            print('Requested simulation stages already complete. Saved simulations are ready for analysis.',flush=True)
            return manifest
    limits=budget_plan(config)['model_step_caps']
    pending=[s for s in expanded if not manifest or manifest['stages'].get(s)!='complete']
    cap=sum(limits[s] for s in pending)
    if cap>maximum_steps:
        raise ValueError(f'planned upper bound {cap:,} model steps exceeds limit {maximum_steps:,}; no simulations started')
    if manifest and manifest['source_sha256']!=source_hashes():
        source=saved_source(output,manifest)
        print('Resuming with the verified archived model source and existing random streams.',flush=True)
        env=dict(os.environ,COMMONS_SAVED_SOURCE=str(source))
        command=[sys.executable,str(ROOT/'reproduce.py'),'--resume-saved',str(output),stage,str(maximum_steps),str(workers)]
        process=subprocess.Popen(command,env=env)
        try:
            code=process.wait()
        except KeyboardInterrupt:
            previous=signal.signal(signal.SIGINT,signal.SIG_IGN)
            try:
                if process.poll() is None:process.send_signal(signal.SIGINT)
                code=process.wait()
            finally:
                signal.signal(signal.SIGINT,previous)
        if code==130:raise KeyboardInterrupt('Stopped after saving the submitted batch')
        if code:raise RuntimeError('Saved-source execution failed')
        return json.loads((output/'manifest.json').read_text())
    if manifest is None and output.exists() and any(output.iterdir()):
        raise ValueError('A new output folder must be empty')
    for key in pending:
        index=ex.STAGES.index(key)
        required=ex.STAGES[index-1] if key in ['validate','test'] else ('validate' if key=='robustness' else None)
        if required and required not in pending and (not manifest or manifest['stages'].get(required)!='complete'):
            raise ValueError(f'{LABELS[key]} requires completed {LABELS[required]}')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.execution.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another experiment is using this bundle')
        print(f'CPU workers: {workers}. Ctrl+C saves the submitted batch; repeat the command to resume.',flush=True)
        if manifest is None:
            from .provenance import identifier,utc_now
            sources=source_hashes();config_id=identifier(config)
            write_json(output/'manifest.json',dict(run_id=identifier([config_id,sources])[:24],created_utc=utc_now(),
                       configuration=config,configuration_id=config_id,source_sha256=sources,
                       environment=environment(),stages={}))
        receipt=output/'execution';receipt.mkdir(exist_ok=True)
        from .provenance import identifier
        execution=dict(workers=workers,runner_sha256=sha256(Path(__file__)),start_method='spawn')
        write_json(receipt/f'runner-{identifier(execution)[:20]}.json',execution)
        try:
            with parallel_execution(workers):
                result=None
                for i,key in enumerate(pending):
                    following=LABELS.get(pending[i+1],'Analysis') if i+1<len(pending) else 'Analysis'
                    print(f'Current: {LABELS[key]}. Next: {following}.',flush=True)
                    result=ex.run_stages(config,output,key,limits[key])
                    print(f'Completed: {LABELS[key]}.',flush=True)
                return result or manifest
        except KeyboardInterrupt:
            print('Stopped after saving submitted simulations. Repeat the same command to resume.',flush=True)
            raise


def run_experiment(name, workers=8):
    from .config import load_config,budget_plan
    from .analysis import analyze,metric_sensitivity
    config=load_config(ROOT/'configs'/f'{name}.json');output=ROOT/'results'/name
    print(f'Experiment: {name}\nResults: {output}',flush=True)
    run_config(config,output,'all',sum(budget_plan(config)['model_step_caps'].values()),workers)
    print('Current: outcome analysis. Simulation results are saved.',flush=True)
    analyze(output)
    if name=='research_protocol':
        for floor in (.1,.3):
            for tail in (.05,.2):
                print(f'Current: depletion sensitivity, floor {floor}, tail {tail}.',flush=True)
                metric_sensitivity(output,floor,tail,output/'analysis'/f'depletion-floor-{floor}-tail-{tail}.csv')
    from .provenance import identifier
    if identifier(config)=='b9d187ef487b45bd64d14643861a006d185f96f0831215e105854ed9877c13df':
        from .trajectory_analysis.pipeline import analyze_trajectories
        destination=output.with_name(output.name+'-analysis')
        if analysis_ready(destination):
            print('Saved trajectory analysis and plots are ready: '+str(destination),flush=True)
        else:
            print('Current: spatial and temporal analysis. Next: study plots.',flush=True)
            analyze_trajectories(output,destination,workers)
    print('Finished: '+str(output/'analysis'),flush=True)


def analysis_ready(destination):
    """Reuse complete analyses only when both source and recorded inputs match."""
    from .provenance import PACKAGE,sha256
    path=Path(destination)/'reproduction_manifest.json'
    if not path.exists():return False
    manifest=json.loads(path.read_text())
    sources={str(f.relative_to(PACKAGE)):sha256(f) for f in PACKAGE.rglob('*.py')}
    if sources!=manifest.get('analysis_source_sha256'):return False
    for name,expected in manifest['input_sha256'].items():
        f=Path(name)
        if not f.is_file() or sha256(f)!=expected:return False
    return all((Path(destination)/'plots/figures'/name).is_file() for name in manifest['figures'])


def replay_saved(bundle,evaluation_id,maximum_steps):
    from .provenance import source_hashes
    from .experiments import replay
    manifest=json.loads((bundle/'manifest.json').read_text())
    if source_hashes()==manifest['source_sha256']:return replay(bundle,evaluation_id)
    snapshot=saved_source(bundle,manifest)
    print('Replaying with the verified archived model source.',file=sys.stderr,flush=True)
    env=dict(os.environ);env.pop('COMMONS_SAVED_SOURCE',None)
    result=subprocess.run([sys.executable,str(snapshot/'reproduce.py'),'replay',str(bundle),evaluation_id,
                           '--max-model-steps',str(maximum_steps)],env=env,capture_output=True,text=True)
    if result.returncode:raise RuntimeError('Archived replay failed: '+result.stderr)
    return json.loads(result.stdout)
