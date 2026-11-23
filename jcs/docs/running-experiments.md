# Running and analyzing the JCS study

## Set up Python once

Follow the repository's [JCS quick start](../../README.md#jcs-quick-start). It installs Python 3.12, the recorded numerical dependencies and the notebook interface into a single `.venv/` at the repository root. Keep using that environment for every JCS task.

**Run every command in this guide from the repository root** (`WaterToC/`, the folder containing `jcs/`). Commands select `.venv/` explicitly; no activation or directory changes are needed. Execution supports macOS and Linux.

## Interactive notebooks

```sh
.venv/bin/jupyter lab jcs/notebooks/
```

JupyterLab opens in your browser. Open either notebook, select **Python 3** and choose **Run → Run All Cells**, or use **Shift+Enter** to run one cell at a time. The [notebook guide](../notebooks/README.md) explains both notebooks and their data requirements.

## Installation check

```sh
.venv/bin/python jcs/reproduce.py smoke --workers 2
```

The smoke configuration is a tiny software check, not research evidence. To run the JCS tests:

```sh
.venv/bin/python -m pytest jcs/tests -q
```

## Experiments of the paper and search audit

The [experiments of the paper](experiments.md) are the experiments that the journal paper reports: baseline continuity with the conference parameterization, payoff changes applied to developed states, controls of behavior-dependent renewal, a grid of the payoff family with a test of the designs it selects, local measures applied to AI agents (the paper calls them interventions), and runs under changed assumptions. Their configurations extend the research protocol, use their own random-input splits and output folders, and never write to a staged-search bundle.

```sh
.venv/bin/python jcs/reproduce.py paper all
```

This runs the six experiments in order and writes the tables of each to `jcs/results/<name>/report/`: `local_measures`, `original_game`, `original_game_long_horizon`, `model_assumptions`, `payoff_grid` and `agent_assumptions`. Together they are about 129 million model steps. The figures are then drawn one script at a time, for example `.venv/bin/python jcs/figures/plot_payoff_settings.py`, and saved to `jcs/results/figures/`; the scripts in `jcs/analysis/` print the numbers behind the paper's tables. The two companion experiments, siting in further settings and the local measures under changed assumptions, are run with the single-configuration commands below, using `jcs/configs/siting_across_settings.json` and `jcs/configs/local_measures_assumptions.json`. Names given after `all` restrict it to those experiments. It uses all but two cores, up to 16, unless `--workers` is given, and repeating it continues from the saved jobs.

Single configurations, including the exploratory pilots and diagnostics, use the explicit commands:

```sh
.venv/bin/python jcs/reproduce.py paper plan --config jcs/configs/pilot.json
.venv/bin/python jcs/reproduce.py paper run --config jcs/configs/pilot.json --output jcs/results/pilot --max-model-steps 2316000 --workers 1
.venv/bin/python jcs/reproduce.py paper report --output jcs/results/pilot
```

`plan` prints the number of model steps without simulating. `run` refuses to start above the stated limit, saves each completed world, reuses saved worlds when repeated, and refuses a changed configuration or source. `report` writes tables to `report/` inside the run folder and never simulates.

```sh
.venv/bin/python jcs/reproduce.py audit-search jcs/results/research_protocol --output jcs/results/search_audit
```

`audit-search` reads a completed staged-search bundle and its saved original-game histories. It reports the range of each search score over all evaluated payoff designs, the scores of the forced-action rules, source states by expected local load, and how many sources per world are overloaded by geometry alone. It never simulates and does not modify the bundle.

## Earlier staged payoff search

The staged payoff search is an exploratory protocol that preceded the experiments of the paper and is not reported in the paper; its [protocol](research-plan.md) describes it. It has one entry point:

```sh
.venv/bin/python jcs/reproduce.py
.venv/bin/python jcs/reproduce.py --workers 4
```

The default runs `research_protocol` with up to eight CPU worker processes. Each process calculates independent worlds; the parent saves results in index order. Each numerical library uses one thread per worker. Worker count changes execution, not the equations or indexed random inputs. `--workers 1` provides the single-worker comparison.

The other configurations are explicit choices:

```sh
.venv/bin/python jcs/reproduce.py human_only_priority --workers 4
.venv/bin/python jcs/reproduce.py ai_only_priority --workers 4
.venv/bin/python jcs/reproduce.py delivery_reliability --workers 4
.venv/bin/python jcs/reproduce.py all --workers 4
```

`all` runs these four configurations sequentially, using CPU workers within each. The main command starts the full research workload; it is not a quick check. Results are written to `jcs/results/<configuration>/` regardless of the invoking directory. Generated results are ignored by Git.

The terminal identifies the current, completed and next stages. During a payoff evaluation it reports saved simulations against that payoff's configured count, completed payoff evaluations in the current stage and the saved simulation count for the bundle. These are saved-work counts; they are not a percentage of the overall search or an estimate derived from a workload ceiling. No separate progress terminal is needed.

Ctrl+C stops after the submitted batch is saved. Wait for the shell prompt, then repeat the command to resume. Do not launch another writer against a running bundle. Closing a laptop lid can suspend computation; the runner does not promise continued work while the computer sleeps. Saved stages and existing evaluations are reused. Configuration and dependency mismatches are refused.

## Stage commands and planning

```sh
.venv/bin/python jcs/reproduce.py plan --config jcs/configs/research_protocol.json
.venv/bin/python jcs/reproduce.py run --config jcs/configs/smoke.json --output /absolute/path/to/bundle --stage characterize --max-model-steps 10000 --workers 2
.venv/bin/python jcs/reproduce.py run --config jcs/configs/smoke.json --output /absolute/path/to/bundle --stage search --max-model-steps 10000 --workers 2
```

The stages are `characterize`, `search`, `validate`, `test`, `robustness` and `all`. Search includes characterization if needed. Selection requires search; testing requires the frozen selected designs. The explicit model-step cap is checked before simulation starts. Planning reports ceilings, not completion percentages or promised runtimes. Historical saved files retain the internal `train` split and `training_*` field names to preserve the original random streams and identifiers.

For an existing bundle made with an earlier source version, resumption verifies every recorded source file in `source_snapshot/` and executes that archived model in a separate process. It does not relax source identity or overwrite the snapshot. The current execution wrapper supplies parallel scheduling and progress. A missing or modified snapshot is refused. Already completed simulation stages are not rerun.

## Analysis and plots

```sh
.venv/bin/python jcs/reproduce.py analyze jcs/results/research_protocol --output /absolute/path/to/empty-analysis --workers 8
```

For the completed research protocol, this command calculates the saved-data spatial and temporal analyses and produces the study plots. It never simulates. Use an empty output directory outside the saved bundle. The detailed analysis follows the reported protocol's phase windows and selected designs; a different protocol needs a separately specified analysis, not silently reused assumptions. Tiny or other configurations use their standard outcome analysis.

The quick-start installation includes pandas for the trajectory analysis alongside the recorded numerical environment. It reads all stored test and characterization histories, recovers the summary statistics, computes resource budgets and coverage components, estimates exploratory action responses, and checks recurrence, sensitivity and provenance. `derived/` contains numerical results; `plots/figures/` contains the PNG and vector PDF plots, with the underlying plotted values alongside them. A completed main experiment also runs this analysis after its outcome summaries and sensitivity checks. An identical completed analysis is reused only after its source and inputs are checked. If an output directory belongs to another analysis version, choose a new empty directory with the explicit analysis command. The same plots can be used in any report.

To redraw plots from already calculated summaries:

```sh
.venv/bin/python jcs/reproduce.py analyze /path/to/saved-bundle --derived /path/to/derived --output /path/to/new-plots
```

This route verifies the run identity and activity means, recomputes combined three-objective frontiers, and records the plotting-source and input hashes. It does not independently repeat trajectory extraction. Full extraction needs the saved detailed histories; the compact plotting inputs alone are insufficient for that step.

The plots cover primary and other tested trade-offs, original paired outcome intervals, spatial burdens, one-action responses, early behavior, cumulative resource budgets, activity windows, assumption sensitivity, coverage components and paired source-event changes. The CSV outputs contain the plotted numerical values. The plots retain separate human support, AI support and depletion outcomes. Ratios divide ensemble means; the original simultaneous intervals remain in difference units. Additional response and sensitivity calculations remain exploratory.

## Verification, replay and rescoring

```sh
.venv/bin/python jcs/reproduce.py verify jcs/results/research_protocol
.venv/bin/python jcs/reproduce.py replay /path/to/bundle EVALUATION_ID --max-model-steps 10000
.venv/bin/python jcs/reproduce.py rescore /path/to/bundle --floor .1 --tail .05 --output /path/to/rescored.csv
```

Verification and rescoring never simulate. Replay intentionally reruns one archived realization; the entry point selects the verified source snapshot when needed; use the recorded numerical environment. Bundle checksums cover immutable inputs and results. Regenerated outcome analyses have their own input manifest. Exporting a verified result bundle and its exact source remains available through `reproduce.py export`; the export contains the study source, the verified result bundle and a SHA256 inventory.

See the [experiments of the paper](experiments.md), [implementation](implementation.md) and [model continuity](model-continuity.md) for scientific definitions, and the [staged search protocol](research-plan.md) for the earlier search. A persistent-low-stock event is not a validated definition of the ICCS spatial trap. Frontier membership describes tested sample means, not statistical dominance or global optimality. An empty supported frontier does not prove infeasibility.
