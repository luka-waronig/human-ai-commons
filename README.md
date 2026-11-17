# Human-AI Commons: Interdisciplinary Chapters

This is the public parent repository of a series of interdisciplinary research chapters studying the coevolution of behavioral and environmental parameter patterns in a spatial and resource-bound Human-AI commons.

| Study | Code and experiments | Documentation |
| --- | --- | --- |
| ICCS 2026 conference paper | [iccs2026/](iccs2026/) | [Model and analysis](iccs2026/docs/model-and-analysis.md), [setup and usage](iccs2026/docs/usage.md) |
| Journal of Computational Science extension | [jcs/](jcs/) | [Research protocol](jcs/docs/research-plan.md), [Running the experiments](jcs/docs/running-experiments.md), [Interactive notebooks](jcs/notebooks/README.md) |


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

To run the full journal experiment, which is a substantial simulation workload:

```sh
.venv/bin/python jcs/reproduce.py
```

The script writes data, tables, and figures to `jcs/results/research_protocol/`. Add `all` to run the main experiment followed by the human-priority, AI-priority, and delivery-protection experiments. See the [experiment instructions](jcs/docs/running-experiments.md) for individual stages, saved-data analysis and verification. Keep using the same `.venv/` in later terminal sessions; the setup does not need repeating.

See the [research overview](docs/overview.md) for the publications and model relationships. Citation metadata is in [CITATION.cff](CITATION.cff).

Both studies use Python 3.12. The conference study has its own setup guide. With both dependency sets installed in `.venv/`, `.venv/bin/python -m pytest -q` from the repository root runs both test suites.

Software is distributed under the [MIT License](LICENSE).
