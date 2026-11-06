# Human-AI Commons: Journal of Computational Science - Special Issue

This folder contains the JCS study. It can be copied to another location to install its dependencies, run tests, generate experiments, and analyze their outputs independently of the parent repository.

The implementation includes the payoff family, spatial simulator, joint welfare calculation, persistent-stress restriction, constrained searches, resource-only comparators, independent validation and testing, observation controls, physical capacity checks, robustness experiments, and artifact replay. `configs/smoke.json` provides a small integration-test configuration; `configs/research_protocol.json` specifies the research experiments.

## Install and check

Use Python 3.12. From the `jcs/` folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements.txt
python -m pytest -q
python scripts/verify_payoff_design.py
```

`requirements.txt` is a complete, hashed dependency lock for the numerical libraries, plotting, and tests. There is no Mesa or parent-project dependency. The `reproduce.py` entry point loads the package from this folder, so an editable install and a custom `PYTHONPATH` are unnecessary. A normal package installation is also supported by `pyproject.toml`.

## Reproduce the small software check

```bash
python reproduce.py plan --config configs/smoke.json
python reproduce.py run --config configs/smoke.json --output results/smoke
python reproduce.py verify results/smoke
python reproduce.py analyze results/smoke
python reproduce.py rescore results/smoke --floor 0.3 --tail 0.2 --output results/smoke/analysis/metric_sensitivity.csv
```

The smoke design uses a 4 by 5 grid, six agents, and twelve time steps per realization. It exercises all workflow stages, both budget norms, and selected mechanism controls. It has a conservative upper bound of 4,140 model steps across the entire workflow, before cache reuse. Output figures identify it as a software check.

The runner defaults to a hard limit of 10,000 model steps. A production configuration is rejected **before constructing worlds or starting simulations** if its planned upper bound exceeds that limit. `plan`, `verify`, `analyze`, and `rescore` never run simulations. `replay` reruns one realization and also has a model-step limit.

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

This reports the training, validation, test, and robustness budgets. The `run` command accepts an explicit `--max-model-steps` and a stage: `characterize`, `train`, `validate`, `test`, or `robustness`. Training runs the characterization stage first if it has not already completed. Use the same output bundle for successive stages. Validation requires training; testing requires frozen validation winners. Completed stages are reused. A changed configuration, source, dependency environment, or modified recorded artifact requires a new bundle.

The [research plan](research-plan.md) explains the scientific question and decision rules. The [implementation record](implementation.md) defines scheduling, random streams, world sampling, and the correspondence between experiment stages and the protocol. Robustness outputs transfer frozen policies; a transfer under changed physical conditions is not a newly optimized outcome.

`research_protocol.json` maximizes joint support for human and AI activity subject to the persistent-local-stress restriction. `human_only_priority.json` and `ai_only_priority.json` reoptimize the headline setting with one activity type given all objective weight. Their physical conditions, random streams, and protection requirement match the main study. `plan` accepts each configuration without starting simulations. Their production runs use separate output bundles and explicit execution budgets. `delivery_reliability.json` holds the headline physical and welfare conditions fixed but constrains persistent unmet requests instead of low source stocks. This is a comparison of protection goals, not a redefinition of the conference spatial trap.

## Output bundle

Each run produces the resolved configuration, a budget plan and ledger, source hashes and a source snapshot, environment versions, generated worlds, candidate scores, validation shortlists, frozen policies, and per-realization metrics. Metrics include individual mean requests and deliveries, the fraction of steps each agent receives nothing, and the duration of continuous depletion at each accessible resource cell. Held-out and robustness realizations additionally archive complete post-withdrawal stock trajectories, strategies, individual requests and deliveries, and their actual random drivers.

`test_comparisons.json` contains paired welfare intervals over independent complete realizations, separate human and AI outcomes, and stress-probability bounds. A selected policy is not automatically supported as protective: inspect `simultaneous_protection_supported`. The small software check has too few independent realizations to support the main probability restriction.

`analysis/heldout_frontier.csv` reports welfare and stress results together. Red crosses in the corresponding figure mark designs whose protection requirement is not statistically established. `welfare_assumptions.csv` and `stress_definitions.csv` re-score frozen policies under alternative definitions. They do not establish reoptimized outcomes under those definitions. `payoff_coordinates.csv` records matrix entries and their derived strategic transitions. `spatial_characterization.csv` reports the resource–strategy pattern and actual request fulfillment together. `spatial_stress.png` shows continuous duration for the configured persistence basis; `spatial_depletion.png` retains the auxiliary cumulative resource-deficit map. Changing the diagnostic depletion reference or spatial tail fraction is also available through `rescore`, without simulations.

`capacity_bounds.jsonl` records the strictly scoped zero-depletion capacity relaxation. This diagnostic does not prove that the welfare problem with its persistent-stress restriction is impossible.

To replay a recorded realization, take its `evaluation_id` from `realizations.jsonl` and pass it after the bundle path to `python reproduce.py replay`. Replay checks the source and numerical dependency versions, uses the archived world and random inputs, and compares recorded trajectories exactly. Exact replay is tested within the same numerical environment; cross-platform bitwise equivalence is not promised. SHA256 checks identify changes to archived files.

Generated `results/` folders are ignored by Git. A research release consists of the completed experiment bundle and the exact study version used to generate it.

`python reproduce.py export results/smoke --output results/smoke-release.zip` demonstrates the release export. It creates a ZIP containing one `jcs/` folder with the study files, the selected result bundle under `results/paper/`, and a SHA256 inventory. It excludes virtual environments and other result folders. Substitute the research bundle path when preparing a publication release.

## Relationship to the conference paper

The previous paper is cited in [CITATION.cff](../CITATION.cff). [Model continuity](model-continuity.md) describes the inherited mechanisms, the payoff embedding, and the new formulation. The JCS experiments generate and archive their own datasets.

Core bundle checksums cover immutable inputs and results. Regenerated files in the default `analysis/` folder have their own input manifest. Export rejects that analysis if its recorded inputs have changed; regenerate it after completing additional stages.
