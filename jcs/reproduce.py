"""Single entry point for study execution, analysis, verification and replay."""
from pathlib import Path
import os
import sys

ROOT=Path(__file__).resolve().parent
# resumption selects the original package before importing any model code.
# the current execution wrapper is available as a fallback, never a replacement model.
if os.environ.get('COMMONS_SAVED_SOURCE'):
    snapshot=Path(os.environ['COMMONS_SAVED_SOURCE']).resolve()
    sys.path.insert(0,str(snapshot/'src'))
    import human_ai_commons
    human_ai_commons.__path__.append(str(ROOT/'src/human_ai_commons'))
    from human_ai_commons import provenance
    provenance.ROOT=snapshot
else:
    sys.path.insert(0,str(ROOT/'src'))

EXPERIMENTS=('research_protocol','human_only_priority','ai_only_priority','delivery_reliability')


def run_experiment(name,workers=8):
    from human_ai_commons.execution import run_experiment as execute
    return execute(name,workers)


def main(argv=None):
    import argparse
    args=list(sys.argv[1:] if argv is None else argv)
    if args and args[0]=='--resume-saved':
        import json
        from human_ai_commons.execution import run_config
        bundle=Path(args[1]);config=json.loads((bundle/'manifest.json').read_text())['configuration']
        return run_config(config,bundle,args[2],int(args[3]),int(args[4]))
    if not args or args[0] in (*EXPERIMENTS,'smoke','all') or args[0]=='--workers':
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument('experiment',nargs='?',choices=(*EXPERIMENTS,'smoke','all'),default='research_protocol')
        parser.add_argument('--workers',type=int,default=min(8,os.cpu_count() or 1))
        options=parser.parse_args(args)
        if options.workers<1:parser.error('workers must be positive')
        for name in EXPERIMENTS if options.experiment=='all' else [options.experiment]:
            run_experiment(name,options.workers)
    else:
        from human_ai_commons.cli import main as commands
        return commands(args)


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:
        print('Stopped. Completed simulations remain saved; repeat the command to resume.',flush=True)
        sys.exit(130)
    except (ValueError,RuntimeError) as error:
        print('Error: '+str(error),file=sys.stderr)
        sys.exit(2)
