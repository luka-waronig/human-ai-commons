# Exploring the JCS model

Two notebooks connect the study's mathematics, implementation and saved results.

| Notebook | Purpose | Inputs |
|---|---|---|
| [Model and mathematics](01_model_and_mathematics.ipynb) | Payoff decisions, renewal and allocation, followed by a small matched spatial example | The JCS package and research configuration; no production data |
| [Exploring saved worlds](02_exploring_saved_worlds.ipynb) | Tested trade-offs, paired outcome distributions and selected spatial/time histories | A compatible JCS Pareto result bundle |

From the repository root, use the study's Python 3.12 environment to install and open the notebook interface:

```sh
uv pip install --python .venv/bin/python -e './jcs[notebooks]'
.venv/bin/jupyter lab jcs/notebooks/
```

Select the kernel belonging to this environment. Run cells in order. Sliders and selectors update plots after release, rather than executing expensive calculations continuously. Restarting the kernel and running all cells restores the initial examples.

The first notebook runs only small illustrative simulations. They are identified as examples and do not alter the research bundle. The second reads saved files and never simulates. Set `BUNDLE_PATH` to a downloaded result bundle or a compatible bundle generated with the study code. The default is `jcs/results/research_protocol/`. A fresh source checkout does not include those results.

The data explorer uses `manifest.json`, `analysis/pareto_candidates.csv`, the original outcome-comparison table when available, and `test_comparisons.json`. Detailed views additionally need the selected `worlds/<id>.npz` and `trajectories/<id>.npz` files. Summary-only bundles can still show candidate outcomes; missing inputs leave the corresponding view inactive. All input paths remain inside the selected bundle. A bounded cache holds at most four selected histories, not the whole archive.

The calculations import the existing JCS model and analysis functions. `_explore.py` provides read-only bundle access and common plotting conventions. The public notebooks contain no saved execution outputs or machine-specific paths. Numerical arrays, selected source files and the original experiment settings remain in the result bundle. Full hash verification and the complete analysis workflow are documented in the [execution guide](../docs/running-experiments.md).

Each world–driver realization is an independent observation. Source locations, actors and time points within a world are dependent. Maps illustrate individual worlds; paired distributions and simultaneous intervals answer different ensemble questions. The notebooks retain these distinctions, as well as the difference between a numerical low-stock condition and a spatial feedback mechanism.
