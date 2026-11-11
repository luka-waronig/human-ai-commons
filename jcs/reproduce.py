"""Run the journal experiments directly, or use individual reproduction commands."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from watertoc_jcs.cli import main

EXPERIMENTS = ('research_protocol', 'human_only_priority', 'ai_only_priority', 'delivery_reliability')


def run_experiment(name):
    from watertoc_jcs.config import load_config, budget_plan
    from watertoc_jcs.experiments import run_stages
    from watertoc_jcs.analysis import analyze, metric_sensitivity, verify_bundle

    config = load_config(ROOT / 'configs' / (name + '.json'))
    output = ROOT / 'results' / name
    plan = budget_plan(config)
    print('Experiment: ' + name, flush=True)
    print('Results: ' + str(output), flush=True)

    for stage, limit in plan['model_step_caps'].items():
        if limit:
            print('  ' + stage, flush=True)
            run_stages(config, output, stage, limit)

    verify_bundle(output)
    analyze(output)
    if name == 'research_protocol':
        for floor in (.1, .3):
            for tail in (.05, .2):
                metric_sensitivity(output, floor, tail,
                    output / 'analysis' / f'depletion-floor-{floor}-tail-{tail}.csv')
    print('Finished: ' + str(output / 'analysis'), flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        run_experiment('research_protocol')
    elif len(sys.argv) == 2 and sys.argv[1] == 'all':
        for experiment in EXPERIMENTS:
            run_experiment(experiment)
    elif len(sys.argv) == 2 and sys.argv[1] in (*EXPERIMENTS, 'smoke'):
        run_experiment(sys.argv[1])
    else:
        main()
