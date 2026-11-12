# Running the JCS experiments

This folder contains the JCS study. It can be copied to another location to install its dependencies, run tests, generate experiments, and analyze their outputs independently of the parent repository.

The implementation includes the payoff family, spatial simulator, three separate outcome objectives, persistent-stress restriction, constrained Pareto searches, resource-only comparators, independent validation and testing, observation controls, physical capacity checks, robustness experiments, and artifact replay. `configs/smoke.json` provides a small integration-test configuration; `configs/research_protocol.json` specifies the research experiments.

## Set up Python once

The setup below uses [uv](https://docs.astral.sh/uv/pip/environments/) to create a Python 3.12 environment and install the pinned requirements. Install uv first if it is not already available.

From the `jcs/` folder, create the environment once, activate it, and install the requirements:

```text
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python --require-hashes -r requirements.txt
uv pip check --python .venv/bin/python
python --version
```

The last command should report Python 3.12.x. `uv pip install` is an installer provided by uv; it does not require pip to be installed in the environment. A uv-created environment without pip is normal.

For each new terminal session, activate the existing environment with `source .venv/bin/activate` from the `jcs/` folder. Do not recreate a working environment. In an IDE, select `jcs/.venv/bin/python` as the interpreter.

These activation commands are for macOS/Linux. On Windows PowerShell, use `.venv\Scripts\Activate.ps1`; replace `.venv/bin/python` with `.venv\Scripts\python.exe` in explicit interpreter paths. If another correctly configured Python 3.12 environment is already in use, it can be used directly instead.

## Check the installation

Run the tiny software check first:

```text
python reproduce.py smoke
```

It should print `Finished:` and create `results/smoke/analysis/`. This checks the workflow; its output is not research evidence.

## Run the experiments

Run the main experiment:

```text
python reproduce.py
```

This runs the original-game measurements, payoff search, validation, independent testing, and robustness experiments in order. It then generates the tested design table, Pareto-frontier tables, PNG and PDF frontier figures, spatial comparisons, and depletion sensitivity analyses. This one main run is sufficient to produce the main frontier. The main command starts the full research workload; it is not a quick check. It prints stage names, but can remain quiet during a long stage.

The following comparisons are optional; none is required to generate the main frontier:

```text
python reproduce.py human_only_priority
python reproduce.py ai_only_priority
python reproduce.py delivery_reliability
```

Alternatively, `python reproduce.py all` runs the main experiment followed by these three comparisons sequentially. Choose the individual commands or `all`; both execute the same four configurations. The comparison configurations have no additional robustness variants.

If the selected Python environment already contains the required dependencies, the same scripts can be run with `python reproduce.py`. From the repository root, use `python jcs/reproduce.py`. In an IDE, select the environment above and run `reproduce.py` to start the main experiment.

## Results and restarting

Results are written relative to the script, regardless of the terminal's current folder:

| Experiment | Result folder inside `jcs/` |
| --- | --- |
| Main | `results/research_protocol/` |
| Human-only priority | `results/human_only_priority/` |
| AI-only priority | `results/ai_only_priority/` |
| Delivery protection | `results/delivery_reliability/` |

Each folder contains the resolved configuration, source record, generated worlds, evaluations, and an `analysis/` directory with tables and figures. The repository ignores generated `results/` folders. Copy the complete result folders to backup storage, including the source records and raw trajectories.

Repeating the same command reuses completed simulation stages and regenerates the analysis. During simulation stages, Ctrl+C requests a stop after the current realization is saved; wait for the script to exit before restarting. A forced kill or power loss may need recovery. Do not run two copies of the same experiment into the same output folder. Keep the model code, configurations, and numerical dependency versions unchanged while collecting one study's results. If they change, preserve the old results and use a separate study copy or the individual-stage commands below with a new output folder.

The direct script verifies result checksums and generates analyses. Replay and ZIP export are separate optional commands described below. The main run produces the original-game/frontier spatial illustrations automatically. These show the same first indexed test world under the original game and representative retained designs; the ensemble tables provide the quantitative evidence.

<details>
<summary>Individual commands for installation, stages, and analysis</summary>

## Install and check

After the setup above, from the `jcs/` folder:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_payoff_design.py
```

`requirements.txt` is a complete, hashed dependency lock for the numerical libraries, plotting, and tests. There is no Mesa or parent-project dependency. The `reproduce.py` entry point loads the package from this folder, so an editable install and a custom `PYTHONPATH` are unnecessary. A normal package installation is also supported by `pyproject.toml`.

## Reproduce the small software check

The remaining reference commands use `python` from the activated Python 3.12 environment.

```bash
python reproduce.py plan --config configs/smoke.json
python reproduce.py run --config configs/smoke.json --output results/smoke
python reproduce.py verify results/smoke
python reproduce.py analyze results/smoke
python reproduce.py rescore results/smoke --floor 0.3 --tail 0.2 --output results/smoke/analysis/metric_sensitivity.csv
```

The smoke design uses a 4 by 5 grid, six agents, and twelve time steps per realization. It exercises all workflow stages, both budget norms, and selected mechanism controls. Use `python reproduce.py plan --config configs/smoke.json` to inspect its conservative model-step bound before cache reuse. Output figures identify it as a software check.

The low-level `run` command defaults to a hard limit of 10,000 model steps. The direct commands above use the budgets calculated from each experiment configuration. With the low-level default, a production configuration is rejected **before constructing worlds or starting simulations** if its planned upper bound exceeds that limit. `plan`, `verify`, `analyze`, and `rescore` never run simulations. `replay` reruns one realization and also has a model-step limit.

## Inspect the reference before optimization

```bash
python reproduce.py run --config configs/smoke.json --output results/reference-check --stage characterize
python reproduce.py analyze results/reference-check
```

The reference stage evaluates the embedded game and forced lower/higher extraction controls on separate characterization realizations. `reference_characterization.json` and `analysis/spatial_characterization.csv` expose resource patterns, actual deliveries, and welfare together. Inspect these before interpreting a resource cutoff as a welfare failure. The same bundle can then continue through `--stage train` and later stages. The short configuration remains a software check; its spatial patterns are not journal findings.

## Plan production experiments

```bash
python reproduce.py plan --config configs/research_protocol.json
```

This reports the training, validation, test, and robustness budgets. The `run` command accepts an explicit `--max-model-steps` and a stage: `characterize`, `train`, `validate`, `test`, or `robustness`. Training runs the characterization stage first if it has not already completed. Use the same output bundle for successive stages. Validation requires training; testing requires the frozen validation design sets. Completed stages are reused. A changed configuration, source, dependency environment, or modified recorded artifact requires a new bundle.

The [research plan](research-plan.md) explains the scientific question and decision rules. The [implementation record](implementation.md) defines scheduling, random streams, world sampling, and the correspondence between experiment stages and the protocol. Robustness outputs transfer frozen policies; a transfer under changed physical conditions is not a newly optimized outcome.

`research_protocol.json` estimates the trade-offs among human support, AI support, and local depletion, subject to the payoff budget and persistent-local-stress restriction. No combined welfare score guides this search. `human_only_priority.json` and `ai_only_priority.json` reoptimize the headline setting with one activity type given all objective weight. Their physical conditions, random streams, and protection requirement match the main study. `plan` accepts each configuration without starting simulations. Their production runs use separate output bundles and explicit execution budgets. `delivery_reliability.json` repeats the three-objective Pareto workflow at the headline setting, constraining persistent unmet requests instead of low source stocks. This is a comparison of protection goals, not a redefinition of the conference spatial trap.

## Output bundle

Each run produces the resolved configuration, a budget plan and ledger, source hashes and a source snapshot, environment versions, generated worlds, candidate scores, validation shortlists, frozen policies, and per-realization metrics. Metrics include individual mean requests and deliveries, the fraction of steps each agent receives nothing, and the duration of continuous depletion at each accessible resource cell. Held-out test realizations additionally archive complete post-withdrawal stock trajectories, strategies, individual requests and deliveries, and their actual random drivers. In the Pareto workflow, robustness realizations retain outcome summaries rather than full trajectories.

`test_comparisons.json` stores a list of frozen designs for each family, separate human/AI/depletion outcomes, paired changes from the original game, observation controls, and resource-probability bounds. A design is not automatically supported as protective: inspect `simultaneous_protection_supported`. The tiny software check has too few independent realizations to support the production probability restriction.

The main outputs in `analysis/` are:

| File | Contents |
| --- | --- |
| `pareto_frontier.png` and `pareto_frontier.pdf` | Human support versus AI support, with depletion shown by colour; separate panels for pressure, budget, and norm |
| `pareto_candidates.csv` | Every tested design and original-game reference, with three mean outcomes, descriptive intervals, payoff parameters, and resource-constraint status |
| `estimated_pareto_frontier.csv` | Non-dominated test sample means among empirically admissible designs, separately by family and including the original game as an alternative |
| `pareto_frontier.csv` | The corresponding frontier restricted to designs with statistical support for the resource condition |
| `pareto_summary.json` | Whether any tested design establishes the resource condition in each problem |
| `paired_outcome_comparisons.csv` | Changes from the original game, finite-sample paired-mean intervals, and separate Hoeffding bounds |
| `training_pareto_candidates.csv` | Evaluated training designs and their three outcomes |
| `spatial_depletion.png`, `spatial_stress.png` | Matched first-world illustrations of the original game and representative retained designs |
| `observation_controls.csv` and `.png` | The same payoff matrices under local, global, and fixed observations |
| `robustness_comparisons.csv` and `robustness.png` | Outcomes and exact resource bounds for validation-selected representatives under changed conditions |
| `robustness_outcome_comparisons.csv` | Simultaneous paired outcome intervals across all robustness variants |

The original game is assessed against the tested alternatives within each family, with `is_reference` identifying its rows. It is not assigned a separate self-only frontier. Frontiers describe the finite set of tested designs. They are estimates, not proofs of global optimality or population-level dominance. The figure does not interpolate an unsupported continuous surface. Hollow markers mean the resource requirement was not established; crosses mean it was violated empirically. A supported-frontier CSV can contain only its header when no design establishes the requirement. All tested outcomes remain available in `pareto_candidates.csv`; lack of support is not proof of infeasibility.

The main tests use 320 independent realizations; each robustness variant uses 256. Robustness assesses up to three validation-selected objective extremes per family. These samples pass an exact planning check: at least 90% power per comparison to establish the 10% persistence limit when the true event rate is 2%, after the stated multiple-comparison correction. The planning rate is an assumption, not a promised property of the designs. `plan` reports these calculations and the revised workload without simulations.

The primary `simultaneous_lower` and `simultaneous_upper` columns give finite-sample confidence bounds for mean changes from the original game, with a correction across comparisons. The method uses the known outcome bounds and independent matched realizations; it also allows for unobserved rare outcomes. `simultaneous_method` identifies the method, while `bounded_lower`/`bounded_upper` retain separate Hoeffding bounds as a conservative diagnostic. Individual bootstrap intervals remain descriptive. Exact resource-probability bounds assess the persistence constraint. The unchanged direct Python command executes this configuration; no additional script is required.

`welfare_assumptions.csv` and `stress_definitions.csv` re-score frozen designs under alternative definitions. They do not establish reoptimized frontiers. `payoff_coordinates.csv` records matrix entries and strategic transitions. `spatial_characterization.csv` reports resource–strategy patterns and actual request fulfillment. Changing the depletion reference or spatial tail fraction is also available through `rescore`, without simulations. The optional single-priority configurations retain their scalar-comparison output format; their `heldout_frontier` files are not the main Pareto result.

`capacity_bounds.jsonl` records the strictly scoped zero-depletion capacity relaxation. This diagnostic does not prove that the welfare problem with its persistent-stress restriction is impossible.

To replay a recorded realization, take its `evaluation_id` from `realizations.jsonl` and pass it after the bundle path to `python reproduce.py replay`. Replay checks the source and numerical dependency versions, uses the archived world and random inputs, and compares recorded trajectories exactly. Exact replay is tested within the same numerical environment; cross-platform bitwise equivalence is not promised. SHA256 checks identify changes to archived files.

Generated `results/` folders are ignored by Git. A research release consists of the completed experiment bundle and the exact study version used to generate it.

`python reproduce.py export results/smoke --output results/smoke-release.zip` demonstrates the release export. It creates a ZIP containing one `jcs/` folder with the study files, the selected result bundle under `results/paper/`, and a SHA256 inventory. It excludes virtual environments and other result folders. Substitute the research bundle path when preparing a publication release.

## Relationship to the conference paper

The previous paper is cited in [CITATION.cff](../CITATION.cff). [Model continuity](model-continuity.md) describes the inherited mechanisms, the payoff embedding, and the new formulation. The JCS experiments generate and archive their own datasets.

Core bundle checksums cover immutable inputs and results. Regenerated files in the default `analysis/` folder have their own input manifest. Export rejects that analysis if its recorded inputs have changed; regenerate it after completing additional stages.

</details>

<details>
<summary>Optional automated launcher</summary>

On macOS/Linux, `./run-experiments` remains available for automated environment setup, a private source copy, concurrent study runs, progress updates, replay checks, and release archives. Its default workspace is `~/Documents/human-ai-commons-private/experiments/full-study/`. Repeating it resumes the saved study. It accepts `--status`, `--smoke`, `--jobs 1`, and `--output /absolute/private/path`. The direct Python commands above do not use this launcher.

</details>
