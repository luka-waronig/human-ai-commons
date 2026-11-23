import argparse
import json
from pathlib import Path
from .analysis import analyze, metric_sensitivity, verify_bundle
from .config import budget_plan, load_config
from .experiments import replay, run_stages


def main(argv=None):
    parser = argparse.ArgumentParser(description='Self-contained JCS study. plan/analyze/verify/rescore never simulate.', epilog='Whole experiment: python reproduce.py [research_protocol|smoke|all] [--workers N]. CPU execution and progress use the same terminal.')
    commands = parser.add_subparsers(dest='command', required=True)
    planning = commands.add_parser('plan', help='Print budget without constructing or simulating worlds')
    planning.add_argument('--config', type=Path, default=Path('configs/research_protocol.json'))
    run = commands.add_parser('run', help='Execute stages with a hard model-step limit')
    run.add_argument('--config', type=Path, default=Path('configs/smoke.json'))
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--stage', choices=['characterize', 'search', 'validate', 'test', 'robustness', 'all'], default='all')
    run.add_argument('--workers', type=int, default=8, help='CPU worker processes (default: 8)')
    run.add_argument('--max-model-steps', type=int, default=10000)
    analysis = commands.add_parser('analyze', help='Tables and figures from recorded results')
    analysis.add_argument('bundle', type=Path)
    analysis.add_argument('--output', type=Path)
    analysis.add_argument('--trajectories', action='store_true', help='Analyze saved spatial and temporal histories; never simulate')
    analysis.add_argument('--derived', type=Path, help='Reuse saved trajectory summaries for plots')
    analysis.add_argument('--workers', type=int, default=8, help='CPU workers for trajectory analysis (default: 8)')
    # legacy aliases for archived reproduction commands.
    analysis.add_argument('--publication', dest='trajectories', action='store_true', help=argparse.SUPPRESS)
    analysis.add_argument('--publication-derived', dest='derived', type=Path, help=argparse.SUPPRESS)
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
    paper = commands.add_parser('paper', help='The experiments of the paper: the original game over time, the payoff grid and the local measures')
    paper.add_argument('action', choices=['plan', 'run', 'report', 'all'])
    paper.add_argument('names', nargs='*', help='With all: the experiments to run (default: every experiment of the paper)')
    paper.add_argument('--config', type=Path, help='A configuration with a paper section (required for plan and run)')
    paper.add_argument('--output', type=Path)
    paper.add_argument('--experiment', choices=['continuity', 'interventions', 'feedback', 'search', 'search_test', 'all'], default='all')
    paper.add_argument('--workers', type=int, help='CPU worker processes (default: 1; with all, all but two cores up to 16)')
    paper.add_argument('--max-model-steps', type=int, default=0)
    audit = commands.add_parser('audit-search', help='What the saved staged search could and could not distinguish; never simulates')
    audit.add_argument('bundle', type=Path)
    audit.add_argument('--output', type=Path, required=True)
    audit.add_argument('--worlds', type=int, default=320, help='Held-out worlds read per load (default: all 320)')
    import sys
    arguments=list(sys.argv[1:] if argv is None else argv)
    arguments=['search' if a=='train' and i and arguments[i-1]=='--stage' else a for i,a in enumerate(arguments)]
    args = parser.parse_args(arguments)
    try:
        if args.command == 'plan':
            result = budget_plan(load_config(args.config))
        elif args.command == 'run':
            if args.max_model_steps < 1:
                raise ValueError('positive model-step limit required')
            from .execution import run_config
            stage='train' if args.stage=='search' else args.stage
            result = run_config(load_config(args.config), args.output, stage, args.max_model_steps, args.workers)
            result = {key: result[key] for key in ['run_id', 'stages', 'model_steps_logged']}
        elif args.command == 'analyze':
            if args.derived is not None:
                if args.output is None:
                    raise ValueError('--derived requires a separate --output directory')
                from .analysis import study_plots
                result=study_plots(args.bundle,args.derived,args.output)
            elif args.trajectories or json.loads((args.bundle/'manifest.json').read_text())['configuration_id']=='b9d187ef487b45bd64d14643861a006d185f96f0831215e105854ed9877c13df':
                from .trajectory_analysis.pipeline import analyze_trajectories
                destination=args.output or args.bundle.resolve().with_name(args.bundle.name+'-analysis')
                result=analyze_trajectories(args.bundle,destination,args.workers)
            else:
                result=analyze(args.bundle,args.output)
        elif args.command == 'verify':
            result = verify_bundle(args.bundle)
        elif args.command == 'replay':
            from .provenance import read_jsonl
            records = {r['evaluation_id']: r for r in read_jsonl(args.bundle/'realizations.jsonl')}
            if records[args.evaluation_id]['steps'] > args.max_model_steps:
                raise ValueError('recorded realization exceeds replay model-step limit')
            verify_bundle(args.bundle)
            from .execution import replay_saved
            result = replay_saved(args.bundle.resolve(), args.evaluation_id, args.max_model_steps)
        elif args.command == 'paper':
            from . import paper_experiments as protocol
            if args.action == 'all':
                from .provenance import ROOT
                if set(args.names)-set(protocol.PAPER_RUNS):
                    raise ValueError('unknown experiment of the paper; choose from '+', '.join(protocol.PAPER_RUNS))
                configs = [(name, protocol.load_protocol(ROOT/'configs'/f'{name}.json')) for name in args.names or protocol.PAPER_RUNS]
                result = protocol.run_all(configs, ROOT/'results', args.workers)
            elif args.action != 'plan' and args.output is None:
                raise ValueError('--output is required')
            elif args.action == 'report':
                # a report describes the saved run; it reads that run's own recorded configuration.
                from .paper_tables import report
                result = report(args.output)
            else:
                if args.config is None:
                    raise ValueError('--config is required')
                config = protocol.load_protocol(args.config)
                names = protocol.experiments_of(config) if args.experiment == 'all' else [args.experiment]
                if args.action == 'plan':
                    result = dict(protocol=config['protocol'], role=config['role'], worlds=config['worlds'],
                                  planned_model_steps=protocol.planned_steps(config, names), simulations_started=False)
                else:
                    result = protocol.run_protocol(config, args.output, args.experiment, args.max_model_steps, args.workers or 1)
        elif args.command == 'audit-search':
            from .paper_tables import audit_search
            result = audit_search(args.bundle, args.output, args.worlds)
        elif args.command == 'export':
            from .release import export_release
            result = export_release(args.bundle, args.output)
        else:
            result = metric_sensitivity(args.bundle, args.floor, args.tail, args.output)
    except (ValueError, RuntimeError, KeyError) as error:
        parser.exit(2, f'Error: {error}\n')
    def display(value):
        if isinstance(value,dict):return {('search' if k=='train' else k):display(v) for k,v in value.items()}
        if isinstance(value,list):return [display(v) for v in value]
        return 'search' if isinstance(value,str) and value=='train' else value
    print(json.dumps(display(result), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
