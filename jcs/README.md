# Strategic Incentives and Joint Welfare in Human-AI Commons: Journal of Computational Science - Special Issue

This JCS Special Issue study investigates patterns in the coevolution of behavioral and environmental parameters in a spatial and resource-bound Human-AI commons. 
This directory contains all code and documentation for reproducing the self-contained experiments that optimize payoff structures to support human and AI activity together, integrating both into a concept of social welfare, subject to restrictions on persistent local resource stress.

- [Installation and reproduction](docs/usage.md)
- [Research protocol](docs/research-plan.md)
- [Model implementation](docs/implementation.md)
- [Continuity with the conference model](docs/model-continuity.md)
- [Design alternatives](docs/design-notes.md)

Run commands from this folder:

```bash
python -m pip install --require-hashes -r requirements.txt
python -m pytest -q
python reproduce.py plan --config configs/research_protocol.json
```

This folder can be copied and run independently of the rest of the repository.
