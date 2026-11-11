# Strategic Incentives and Joint Welfare in Human-AI Commons: Journal of Computational Science - Special Issue

This JCS Special Issue study investigates patterns in the coevolution of behavioral and environmental parameters in a spatial and resource-bound Human-AI commons. 
This directory contains all code and documentation for reproducing the self-contained experiments that optimize payoff structures to support human and AI activity together, integrating both into a concept of social welfare, subject to restrictions on persistent local resource stress.

- [Installation and reproduction](docs/running-experiments.md)
- [Research protocol](docs/research-plan.md)
- [Model implementation](docs/implementation.md)
- [Continuity with the conference model](docs/model-continuity.md)
- [Design alternatives](docs/design-notes.md)

With Python 3.12 and the dependencies installed, run the main experiment from this folder:

```text
python reproduce.py
```

For a fresh installation with uv, follow [Set up Python once](docs/running-experiments.md#set-up-python-once). The script runs the original-game measurements, payoff search, validation, independent tests, and robustness experiments, then generates tables and figures. Results are written to `results/research_protocol/`, which is ignored by Git.

Run each comparison separately, or use `all` to run all four experiments in order:

```text
python reproduce.py human_only_priority
python reproduce.py ai_only_priority
python reproduce.py delivery_reliability
python reproduce.py all
```

Use `python reproduce.py smoke` for the tiny software check. Repeating an experiment reuses completed stages when its source, settings, and dependencies still match. The [experiment instructions](docs/running-experiments.md) cover individual stages and the optional automated launcher.

This folder can be copied and run independently of the rest of the repository.
