# Running and analyzing the JCS study

## Set up Python once

Use Python 3.12 and the recorded numerical dependencies. From `jcs/`:

```sh
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python --require-hashes -r requirements.txt
uv pip install --python .venv/bin/python -e '.[analysis]'
python reproduce.py smoke --workers 2
```

The smoke configuration is a software check, not research evidence. An existing compatible environment can be used directly. On Windows, replace the activation and interpreter paths with their Windows equivalents; the execution lock currently requires macOS or Linux.

## Experiments

There is one entry point:

```sh
python reproduce.py
python reproduce.py --workers 4
```

The default runs `research_protocol` with up to eight CPU worker processes. Each process calculates independent worlds; the parent saves results in index order. Each numerical library uses one thread per worker. Worker count changes execution, not the equations or indexed random inputs. `--workers 1` provides the single-worker comparison.

The other configurations are explicit choices:

```sh
python reproduce.py human_only_priority --workers 4
python reproduce.py ai_only_priority --workers 4
python reproduce.py delivery_reliability --workers 4
python reproduce.py all --workers 4
```

`all` runs these four configurations sequentially, using CPU workers within each. The main command starts the full research workload; it is not a quick check. Results are written to `jcs/results/<configuration>/` regardless of the invoking directory. Generated results are ignored by Git.

The terminal identifies the current, completed and next stages. During a payoff evaluation it reports saved simulations against that payoff's configured count, completed payoff evaluations in the current stage and the saved simulation count for the bundle. These are saved-work counts; they are not a percentage of the overall search or an estimate derived from a workload ceiling. No separate progress terminal is needed.

Ctrl+C stops after the submitted batch is saved. Wait for the shell prompt, then repeat the command to resume. Do not launch another writer against a running bundle. Closing a laptop lid can suspend computation; the runner does not promise continued work while the computer sleeps. Saved stages and existing evaluations are reused. Configuration and dependency mismatches are refused.

## Stage commands and planning

```sh
python reproduce.py plan --config configs/research_protocol.json
python reproduce.py run --config configs/smoke.json --output /absolute/path/to/bundle --stage characterize --max-model-steps 10000 --workers 2
python reproduce.py run --config configs/smoke.json --output /absolute/path/to/bundle --stage search --max-model-steps 10000 --workers 2
```

The stages are `characterize`, `search`, `validate`, `test`, `robustness` and `all`. Search includes characterization if needed. Selection requires search; testing requires the frozen selected designs. The explicit model-step cap is checked before simulation starts. Planning reports ceilings, not completion percentages or promised runtimes. Historical saved files retain the internal `train` split and `training_*` field names to preserve the original random streams and identifiers.

For an existing bundle made with an earlier source version, resumption verifies every recorded source file in `source_snapshot/` and executes that archived model in a separate process. It does not relax source identity or overwrite the snapshot. The current execution wrapper supplies parallel scheduling and progress. A missing or modified snapshot is refused. Already completed simulation stages are not rerun.

## Analysis and plots

```sh
python reproduce.py analyze results/research_protocol --output /absolute/path/to/empty-analysis --workers 8
```

For the completed research protocol, this command calculates the saved-data spatial and temporal analyses and produces the study plots. It never simulates. Use an empty output directory outside the saved bundle. The detailed analysis follows the reported protocol's phase windows and selected designs; a different protocol needs a separately specified analysis, not silently reused assumptions. Tiny or other configurations use their standard outcome analysis.

The trajectory analysis needs pandas (`.[analysis]`) as well as the original numerical environment. It reads all stored test and characterization histories, recovers the summary statistics, computes resource budgets and coverage components, estimates exploratory action responses, and checks recurrence, sensitivity and provenance. `derived/` contains numerical results; `plots/figures/` contains the PNG and vector PDF plots, with the underlying plotted values alongside them. A completed main experiment also runs this analysis after its outcome summaries and sensitivity checks. An identical completed analysis is reused only after its source and inputs are checked. If an output directory belongs to another analysis version, choose a new empty directory with the explicit analysis command. The same plots can be used in any report.

To redraw plots from already calculated summaries:

```sh
python reproduce.py analyze /path/to/saved-bundle --derived /path/to/derived --output /path/to/new-plots
```

This route verifies the run identity and activity means, recomputes combined three-objective frontiers, and records the plotting-source and input hashes. It does not independently repeat trajectory extraction. Full extraction needs the saved detailed histories; the compact plotting inputs alone are insufficient for that step.

The plots cover primary and other tested trade-offs, original paired outcome intervals, spatial burdens, one-action responses, early behavior, cumulative resource budgets, activity windows, assumption sensitivity, coverage components and paired source-event changes. The CSV outputs contain the plotted numerical values. The plots retain separate human support, AI support and depletion outcomes. Ratios divide ensemble means; the original simultaneous intervals remain in difference units. Additional response and sensitivity calculations remain exploratory.

## Verification, replay and rescoring

```sh
python reproduce.py verify results/research_protocol
python reproduce.py replay /path/to/bundle EVALUATION_ID --max-model-steps 10000
python reproduce.py rescore /path/to/bundle --floor .1 --tail .05 --output /path/to/rescored.csv
```

Verification and rescoring never simulate. Replay intentionally reruns one archived realization; the entry point selects the verified source snapshot when needed; use the recorded numerical environment. Bundle checksums cover immutable inputs and results. Regenerated outcome analyses have their own input manifest. Exporting a verified result bundle and its exact source remains available through `reproduce.py export`; the export contains the study source, the verified result bundle and a SHA256 inventory.

See the [research protocol](research-plan.md), [implementation](implementation.md) and [model continuity](model-continuity.md) for scientific definitions. A persistent-low-stock event is not a validated definition of the ICCS spatial trap. Frontier membership describes tested sample means, not statistical dominance or global optimality. An empty supported frontier does not prove infeasibility.
