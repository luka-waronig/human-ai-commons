# Human-AI Commons: Interdisciplinary Chapters

This is the public parent repository of a series of interdisciplinary research chapters studying the coevolution of behavioral and environmental parameter patterns in a spatial and resource-bound Human-AI commons.

| Study | Code and experiments | Documentation |
| --- | --- | --- |
| ICCS 2026 conference paper | [iccs2026/](iccs2026/) | [Model and analysis](iccs2026/docs/model-and-analysis.md), [setup and usage](iccs2026/docs/usage.md) |
| Journal of Computational Science extension | [jcs/](jcs/) | [Experiments of the paper](jcs/docs/experiments.md), [Running the experiments](jcs/docs/running-experiments.md), [Interactive notebooks](jcs/notebooks/README.md) |


Each independent study has its own dependencies and tests. While they deliberately share the parent repository, the JCS journal study runs independently of the ICCS conference study.

## JCS quick start

Use one Python environment, `.venv/`, at the repository root: the `WaterToC/` folder containing this README and `jcs/`. Run every JCS terminal command from that folder. The commands below select this environment directly, so no activation is needed. JCS execution supports macOS and Linux.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first. For a fresh clone, create the environment once; uv supplies Python 3.12 if needed:

```sh
uv venv --python 3.12 .venv
```

If `.venv/` already exists with Python 3.12, keep it and skip that creation command. Install the study, analysis and notebook dependencies into this same environment:

```sh
uv pip install --python .venv/bin/python --require-hashes -r jcs/requirements.txt
uv pip install --python .venv/bin/python -e './jcs[analysis,notebooks]'
```

Check the installation with a tiny software run:

```sh
.venv/bin/python jcs/reproduce.py smoke --workers 2
```

Open the interactive notebooks:

```sh
.venv/bin/jupyter lab jcs/notebooks/
```

In JupyterLab, open either notebook, select **Python 3**, then choose **Run → Run All Cells**. The model notebook works without research data. The saved-results notebook needs a result bundle; see the [notebook guide](jcs/notebooks/README.md).

The experiments of the journal paper run with one command, which is a substantial simulation workload:

```sh
.venv/bin/python jcs/reproduce.py paper all
```

It runs six experiments and writes the tables of each to `jcs/results/<name>/report/`; two companion experiments run from their own configuration files. The [experiment descriptions](jcs/docs/experiments.md) specify them and the [experiment instructions](jcs/docs/running-experiments.md#experiments-of-the-paper-and-search-audit) describe their output. Results are not part of the repository. Keep using the same `.venv/` in later terminal sessions; the setup does not need repeating.

The `jcs/` folder also keeps an earlier staged payoff search, an exploratory protocol that preceded the experiments of the paper and is not reported in it. Its [protocol](jcs/docs/research-plan.md) and [instructions](jcs/docs/running-experiments.md#earlier-staged-payoff-search) describe it, and `.venv/bin/python jcs/reproduce.py` runs it.

See the [research overview](docs/overview.md) for the publications and model relationships. Citation metadata is in [CITATION.cff](CITATION.cff).

Both studies use Python 3.12. The conference study has its own setup guide. With both dependency sets installed in `.venv/`, `.venv/bin/python -m pytest -q` from the repository root runs both test suites.

Software is distributed under the [MIT License](LICENSE).
