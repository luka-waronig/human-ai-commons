# Strategic Incentives and Joint Welfare in Human-AI Commons: Journal of Computational Science - Special Issue

This JCS Special Issue study investigates patterns in the coevolution of behavioral and environmental parameters in a spatial and resource-bound Human-AI commons. 
This directory contains all code and documentation for reproducing the experiments of the paper: whether the depleted locations of the conference study persist and what determines them, what changes to the payoffs do to the share of demand met for human and AI agents, and what three local interventions applied to AI agents change.

- [Experiments of the paper](docs/experiments.md)
- [Installation and reproduction](docs/running-experiments.md)
- [Interactive notebooks](notebooks/README.md)
- [Earlier staged payoff search](docs/research-plan.md)
- [Model implementation](docs/implementation.md)
- [Continuity with the conference model](docs/model-continuity.md)

Follow the repository's [JCS quick start](../README.md#jcs-quick-start) first. It sets up one Python 3.12 environment at the repository root for experiments, analysis and notebooks. Run all commands below from that root (`WaterToC/`, the folder containing `jcs/`).

Open the notebooks:

```sh
.venv/bin/jupyter lab jcs/notebooks/
```

Check the installation with the tiny software run:

```sh
.venv/bin/python jcs/reproduce.py smoke --workers 2
```

The experiments of the paper run with:

```text
.venv/bin/python jcs/reproduce.py paper all
```

Their results are written to `jcs/results/<name>/` for each of the six experiments, with tables under `report/`. The two companion experiments, siting in further settings and the local measures under changed assumptions, run from their own configuration files, as the [experiment instructions](docs/running-experiments.md#experiments-of-the-paper-and-search-audit) describe. One script in `figures/` draws each figure, and the scripts in `analysis/` print the numbers behind the paper's tables. The paper calls the local measures interventions.

The earlier staged payoff search is an exploratory protocol that preceded the experiments of the paper and is not reported in it. It runs with:

```text
.venv/bin/python jcs/reproduce.py --workers 8
```

The search is a substantial simulation workload. The script runs the original-game measurements, payoff search, validation, independent tests, and robustness experiments, then generates numerical summaries and plots. Independent worlds use CPU worker processes, with progress in the same terminal. Detailed spatial and temporal plots are written alongside the bundle in `jcs/results/research_protocol-analysis/plots/`. Results are written to `jcs/results/research_protocol/`, which is ignored by Git. The main Pareto figure within the bundle is `analysis/pareto_frontier.png` (also PDF); `analysis/pareto_candidates.csv` contains every tested design and its three outcomes. This run produces the frontier without the separate priority-comparison runs.

Optional comparisons can be run separately; `all` runs the search and all three comparisons in order:

```text
.venv/bin/python jcs/reproduce.py human_only_priority
.venv/bin/python jcs/reproduce.py ai_only_priority
.venv/bin/python jcs/reproduce.py delivery_reliability
.venv/bin/python jcs/reproduce.py all
```

Repeating an experiment reuses completed stages when its source, settings, and dependencies still match. The [experiment instructions](docs/running-experiments.md) cover individual stages and CPU execution.

The JCS code has no runtime dependency on `iccs2026/`.

Software is distributed under the [MIT License](LICENSE).
