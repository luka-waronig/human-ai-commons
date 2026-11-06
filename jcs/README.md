# Human-AI Commons: Journal of Computational Science - Special Issue

Self-contained experiments that optimize payoff structures to support human and AI activity together, subject to restrictions on persistent local resource stress.

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
