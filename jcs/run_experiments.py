"""Journal experiments with an isolated environment and saved source copy."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time


SOURCE = Path(__file__).resolve().parent
FILES = ['README.md', 'CITATION.cff', 'pyproject.toml', 'requirements.txt',
         '.python-version', 'reproduce.py', 'run-experiments', 'run_experiments.py',
         'configs', 'docs', 'src', 'tests', 'scripts', 'verification']
STUDIES = [('joint', 'research_protocol'), ('human', 'human_only_priority'),
           ('ai', 'ai_only_priority'), ('delivery', 'delivery_reliability')]
STAGES = [('characterize', 'original-game measurements'), ('train', 'payoff search'),
          ('validate', 'design selection'), ('test', 'independent tests'),
          ('robustness', 'robustness experiments')]
STOP = threading.Event()
PRINT_LOCK = threading.Lock()


def say(message):
    with PRINT_LOCK:
        print(message, flush=True)


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def command(args, log, *, label, bundle=None):
    if STOP.is_set():
        raise InterruptedError('Run stopped.')
    say(label)
    started = time.monotonic()
    offsets, counts = {}, {}
    with log.open('ab') as output:
        output.write(('\n' + label + '\n').encode())
        output.flush()
        process = subprocess.Popen(args, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        last_update = started
        interrupted = False
        while process.poll() is None:
            if STOP.is_set() and not interrupted:
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
                interrupted = True
            now = time.monotonic()
            if bundle is not None and now - last_update >= 20:
                for name in ['realizations', 'training_candidates']:
                    path = bundle / (name + '.jsonl')
                    if path.exists():
                        with path.open('rb') as stream:
                            stream.seek(offsets.get(name, 0))
                            counts[name] = counts.get(name, 0) + stream.read().count(b'\n')
                            offsets[name] = stream.tell()
                say('  {}: {} saved realizations, {} evaluated designs; {:.0f} min elapsed'.format(
                    bundle.name, counts.get('realizations', 0),
                    counts.get('training_candidates', 0), (now-started)/60))
                last_update = now
            try:
                time.sleep(.2)
            except KeyboardInterrupt:
                STOP.set()
    if interrupted or STOP.is_set():
        raise InterruptedError('Run stopped after saving completed work.')
    if process.returncode:
        tail = '\n'.join(log.read_text(errors='replace').splitlines()[-15:])
        raise RuntimeError('{} failed. Log: {}\n{}'.format(label, log, tail))


def python312():
    choices = [sys.executable, str(SOURCE.parent / '.venv/bin/python'),
               shutil.which('python3.12')]
    for choice in choices:
        if choice and Path(choice).is_file():
            result = subprocess.run([choice, '-c',
                'import sys; print("%d.%d" % sys.version_info[:2])'],
                capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip() == '3.12':
                return choice
    raise RuntimeError('Python 3.12 is required. Install it, then run this command again.')


def prepare(root, smoke):
    mode = 'software-check' if smoke else 'full-study'
    metadata = root / 'run.json'
    if metadata.exists():
        if read(metadata)['mode'] != mode:
            raise RuntimeError('This folder belongs to a different run mode. Use another --output folder.')
        say('Resuming the saved study in ' + str(root))
    else:
        if any(p.name != '.run.lock' for p in root.iterdir()):
            raise RuntimeError('This folder already contains files. Use a new --output folder.')
        temporary = root / 'study.tmp'
        temporary.mkdir()
        for name in FILES:
            source = SOURCE / name
            if source.is_dir():
                shutil.copytree(source, temporary / name, ignore=shutil.ignore_patterns(
                    '__pycache__', '.pytest_cache', '*.pyc', '*.egg-info'))
            elif source.exists():
                shutil.copy2(source, temporary / name)
        temporary.rename(root / 'study')
        write(metadata, dict(mode=mode, source=str(SOURCE), created=time.time()))
        say('Saved the study source in ' + str(root / 'study'))
    for name in ['logs', 'plans', 'bundles', 'releases', 'completed']:
        (root / name).mkdir(exist_ok=True)
    python = root / '.venv/bin/python'
    if not python.exists():
        command([python312(), '-m', 'venv', str(root / '.venv')],
                root / 'logs/setup.log', label='Creating the Python 3.12 environment')
    requirements = root / 'study/requirements.txt'
    marker = root / '.environment-ready.json'
    if not marker.exists():
        command([str(python), '-m', 'pip', 'install', '--require-hashes', '-r', str(requirements)],
                root / 'logs/setup.log', label='Installing the pinned dependencies')
        write(marker, dict(requirements=digest(requirements)))
    elif read(marker)['requirements'] != digest(requirements):
        raise RuntimeError('The saved dependency lock changed. Use a new experiment folder.')
    return python


def cli(root, python, args, label, bundle=None):
    command([str(python), str(root / 'study/reproduce.py')] + args,
            root / 'logs' / ((bundle.name if bundle else 'setup') + '.log'),
            label=label, bundle=bundle)


def plan(root, python, config):
    destination = root / 'plans' / (config + '.json')
    result = subprocess.run([str(python), str(root / 'study/reproduce.py'), 'plan',
        '--config', str(root / 'study/configs' / (config + '.json'))],
        capture_output=True, text=True, check=True)
    value = json.loads(result.stdout)
    write(destination, value)
    return value


def stage(root, python, name, config, limits, key, label):
    bundle = root / 'bundles' / name
    if read(bundle / 'manifest.json', {}).get('stages', {}).get(key) == 'complete':
        say(name + ': ' + label + ' already complete')
        return
    if not limits['model_step_caps'][key]:
        return
    cap = limits['model_step_caps'][key]
    if key == 'train' and read(bundle / 'manifest.json', {}).get('stages', {}).get('characterize') != 'complete':
        cap += limits['model_step_caps']['characterize']
    cli(root, python, ['run', '--config', str(root / 'study/configs' / (config + '.json')),
        '--output', str(bundle), '--stage', key, '--max-model-steps', str(cap)],
        name + ': ' + label, bundle)


def finish(root, python, name):
    bundle = root / 'bundles' / name
    cli(root, python, ['verify', str(bundle)], name + ': checking saved results', bundle)
    receipt = root / 'completed' / (name + '.json')
    previous = read(receipt, {})
    archive = Path(previous.get('archive', str(root / 'releases' / (name + '.zip'))))
    if (previous.get('results') == digest(bundle / 'checksums.json') and archive.exists()
            and previous.get('archive_sha256') == digest(archive)):
        say(name + ': results and release already complete')
        return
    cli(root, python, ['analyze', str(bundle)], name + ': making tables and figures', bundle)
    if name == 'joint':
        for floor in [.1, .3]:
            for tail in [.05, .2]:
                cli(root, python, ['rescore', str(bundle), '--floor', str(floor), '--tail', str(tail),
                    '--output', str(bundle / 'analysis' / ('depletion-floor-{}-tail-{}.csv'.format(floor, tail)))],
                    name + ': checking the depletion measurement', bundle)
    comparisons = read(bundle / 'test_comparisons.json')
    primary = next(row for row in comparisons['problems'] if row['problem'] == comparisons['primary_problem'])
    local = primary['families']['local'][0] if comparisons.get('mode') == 'pareto' else primary['policies']['local']
    evaluation = local['realization_ids'][0]
    steps = read(bundle / 'manifest.json')['configuration']['objective']['total_steps']
    cli(root, python, ['replay', str(bundle), evaluation, '--max-model-steps', str(steps)],
        name + ': replaying one saved realization', bundle)
    archive = root / 'releases' / (name + '.zip')
    number = 2
    while archive.exists():
        archive = root / 'releases' / ('{}-{}.zip'.format(name, number))
        number += 1
    cli(root, python, ['export', str(bundle), '--output', str(archive)],
        name + ': creating the verified release archive', bundle)
    write(receipt, dict(results=digest(bundle / 'checksums.json'),
                        archive=str(archive), archive_sha256=digest(archive)))
    say(name + ': complete')


def study(root, python, name, config):
    limits = plan(root, python, config)
    for key, label in STAGES:
        stage(root, python, name, config, limits, key, label)
    finish(root, python, name)


def worker(root, smoke, jobs):
    python = root / '.venv/bin/python'
    previous_handler = signal.signal(signal.SIGINT, lambda *_: STOP.set())
    sleeper = None
    if shutil.which('caffeinate'):
        sleeper = subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
    try:
        if not (root / '.software-checked').exists():
            command([str(python), '-m', 'pip', 'check'], root / 'logs/checks.log',
                    label='Checking dependencies')
            command([str(python), '-m', 'pytest', '-q', str(root / 'study/tests')],
                    root / 'logs/checks.log', label='Checking the implementation')
            command([str(python), str(root / 'study/scripts/verify_payoff_design.py')],
                    root / 'logs/checks.log', label='Checking the payoff equations')
            (root / '.software-checked').touch()
        study(root, python, 'software-check', 'smoke')
        if not smoke:
            limits = plan(root, python, 'research_protocol')
            stage(root, python, 'joint', 'research_protocol', limits, 'characterize', STAGES[0][1])
            cli(root, python, ['analyze', str(root / 'bundles/joint')],
                'Saving the original-game analysis', root / 'bundles/joint')
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                futures = [pool.submit(study, root, python, name, config) for name, config in STUDIES]
                try:
                    for future in as_completed(futures):
                        future.result()
                except BaseException:
                    STOP.set()
                    raise
        say('Finished. Results: ' + str(root / 'bundles'))
        say('Verified archives: ' + str(root / 'releases'))
        if not smoke:
            say('The cross-priority comparison and original-game/redesign spatial figures still need assembly from these results.')
        return 0
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        if sleeper is not None:
            sleeper.terminate()
            sleeper.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Private experiment folder (outside the study source)')
    parser.add_argument('--smoke', action='store_true', help='Run only the tiny software check')
    parser.add_argument('--status', action='store_true', help='Show saved stages without starting work')
    parser.add_argument('--jobs', type=int, choices=range(1, 5), default=min(4, max(1, (os.cpu_count() or 1)//4)))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = (args.output or Path.home() / 'Documents/human-ai-commons-private/experiments' /
            ('software-check' if args.smoke else 'full-study')).expanduser().resolve()
    if args.status:
        if not (root / 'bundles').exists():
            say('Not started: ' + str(root))
        else:
            for folder in sorted((root / 'bundles').iterdir()):
                say(folder.name + ': ' + str(read(folder / 'manifest.json', {}).get('stages', {})))
        return 0
    if args.worker:
        return worker(root, args.smoke, args.jobs)
    if root.is_relative_to(SOURCE) or SOURCE.is_relative_to(root):
        parser.error('Results must be outside the study source folder.')
    if any((folder / '.git').exists() for folder in (root, *root.parents)):
        parser.error('Use a private output folder outside Git repositories.')
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('This experiment folder is already running. Use --status to inspect it.')
        python = prepare(root, args.smoke)
        environment = os.environ.copy()
        environment.update(MPLBACKEND='Agg', OPENBLAS_NUM_THREADS='1',
                           OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
        environment.pop('PYTHONPATH', None)
        command_line = [str(python), str(root / 'study/run_experiments.py'), '--worker',
                        '--output', str(root), '--jobs', str(args.jobs)]
        if args.smoke:
            command_line.append('--smoke')
        process = subprocess.Popen(command_line, cwd=root / 'study', env=environment)
        try:
            return process.wait()
        except KeyboardInterrupt:
            say('Stopping after the current realization is saved. Please wait.')
            previous_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                return process.wait()
            finally:
                signal.signal(signal.SIGINT, previous_handler)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except InterruptedError as error:
        say(str(error) + ' Run the same command to resume.')
        sys.exit(130)
    except (RuntimeError, subprocess.CalledProcessError, OSError) as error:
        say(str(error))
        sys.exit(1)
