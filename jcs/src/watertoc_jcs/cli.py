import argparse
import json
from pathlib import Path
from .analysis import analyze, metric_sensitivity, verify_bundle
from .config import budget_plan, load_config
from .experiments import replay, run_stages


def main(argv=None):
    parser = argparse.ArgumentParser(description='Self-contained JCS study. plan/analyze/verify/rescore never simulate.')
    commands = parser.add_subparsers(dest='command', required=True)
    planning = commands.add_parser('plan', help='Print budget without constructing or simulating worlds')
    planning.add_argument('--config', type=Path, default=Path('configs/research_protocol.json'))
    run = commands.add_parser('run', help='Execute stages with a hard model-step limit')
    run.add_argument('--config', type=Path, default=Path('configs/smoke.json'))
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--stage', choices=['characterize', 'train', 'validate', 'test', 'robustness', 'all'], default='all')
    run.add_argument('--max-model-steps', type=int, default=10000)
    analysis = commands.add_parser('analyze', help='Tables and figures from recorded results')
    analysis.add_argument('bundle', type=Path)
    analysis.add_argument('--output', type=Path)
    verify = commands.add_parser('verify', help='Verify recorded artifact SHA256 hashes')
    verify.add_argument('bundle', type=Path)
    playback = commands.add_parser('replay', help='Rerun one recorded realization and compare its trajectory')
    playback.add_argument('bundle', type=Path)
    playback.add_argument('evaluation_id')
    playback.add_argument('--max-model-steps', type=int, default=10000)
    scoring = commands.add_parser('rescore', help='Change depletion metric using archived test trajectories')
    scoring.add_argument('bundle', type=Path)
    scoring.add_argument('--floor', type=float, required=True)
    scoring.add_argument('--tail', type=float, required=True)
    scoring.add_argument('--output', type=Path, required=True)
    exporting = commands.add_parser('export', help='Export this study and a verified completed bundle as one ZIP')
    exporting.add_argument('bundle', type=Path)
    exporting.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'plan':
            result = budget_plan(load_config(args.config))
        elif args.command == 'run':
            if args.max_model_steps < 1:
                raise ValueError('positive model-step limit required')
            result = run_stages(load_config(args.config), args.output, args.stage, args.max_model_steps)
            result = {key: result[key] for key in ['run_id', 'stages', 'model_steps_logged']}
        elif args.command == 'analyze':
            result = analyze(args.bundle, args.output)
        elif args.command == 'verify':
            result = verify_bundle(args.bundle)
        elif args.command == 'replay':
            from .provenance import read_jsonl
            records = {r['evaluation_id']: r for r in read_jsonl(args.bundle/'realizations.jsonl')}
            if records[args.evaluation_id]['steps'] > args.max_model_steps:
                raise ValueError('recorded realization exceeds replay model-step limit')
            verify_bundle(args.bundle)
            result = replay(args.bundle, args.evaluation_id)
        elif args.command == 'export':
            from .release import export_release
            result = export_release(args.bundle, args.output)
        else:
            result = metric_sensitivity(args.bundle, args.floor, args.tail, args.output)
    except (ValueError, RuntimeError, KeyError) as error:
        parser.exit(2, f'Error: {error}\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
