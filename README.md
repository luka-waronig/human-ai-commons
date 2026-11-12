# Human-AI Commons: Interdisciplinary Chapters

This is the public parent repository of a series of interdisciplinary research chapters studying the coevolution of behavioral and environmental parameter patterns in a spatial and resource-bound Human-AI commons.

| Study | Code and experiments | Documentation |
| --- | --- | --- |
| ICCS 2026 conference paper | [iccs2026/](iccs2026/) | [Model and analysis](iccs2026/docs/model-and-analysis.md), [setup and usage](iccs2026/docs/usage.md) |
| Journal of Computational Science extension | [jcs/](jcs/) | [Research protocol](jcs/docs/research-plan.md), [Running the experiments](jcs/docs/running-experiments.md) |


Each independent study has its own dependencies and tests. While they deliberately share the parent repository, the JCS journal study runs independently of the ICCS conference study.

Run the journal experiments from this directory with:

```bash
python jcs/reproduce.py
```

Use a Python 3.12 environment with the journal requirements installed. The script runs the main experiment and writes data, tables, and figures to `jcs/results/research_protocol/`. Add `all` to run the main experiment followed by the human-priority, AI-priority, and delivery-protection experiments. See the journal [experiment instructions](jcs/docs/running-experiments.md) for individual experiment names.

See the [research overview](docs/overview.md) for the publications and model relationships. Citation metadata is in [CITATION.cff](CITATION.cff).

Both studies use Python 3.12. Install the requirements from the relevant study folder. With both dependency sets installed, `python -m pytest -q` from this directory runs both test suites.

Software is distributed under the [MIT License](LICENSE).
