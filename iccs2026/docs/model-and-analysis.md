# Model equations and analysis reference

Regression fixtures use source commit `3c5fdc918b7f4a22dc6d9c20b460878134bfed2a`. The model and agent files at tag `v1.2` match this baseline. The [baseline manifest](../reproducibility/baseline_manifest.json) records source checksums, and the [original dependency export](../reproducibility/requirements-original.txt) records the environment.

## Replenishment rules

Let x denote local cooperation and n normalized resource level. Replenishment adds the baseline renewal rate multiplied by m, then caps the stock at capacity.

| Identifier | Feedback expression | Multiplier limits | Empty neighbourhood |
| --- | --- | --- | --- |
| repository | theta x minus 1 | 1 to 3 | baseline renewal |
| pre_floor_change | theta x minus 1 | 0.1 to 3 | baseline renewal |
| manuscript | theta x minus (1 minus x) | 0.1 to 3 | x equals zero, following Algorithm 2 |

In each row, m is the clipped value of 1 plus 0.5 n(1 minus n) times the feedback expression. Commit `11c00e4` changed the lower limit from 0.1 to 1. The repository feedback expression already appears in the earliest available model commit.

The default is `repository`; the other identifiers select sensitivity variants. At theta 3, x 0.7, n 0.4, repository and manuscript multipliers are 1.132 and 1.216 respectively.

The repository sweep densities are 0.2, 0.4, 0.5, 0.7; the chapter's Table 2 lists 0.2, 0.3, 0.4, 0.5. The sweep CLI exposes both designs and records the selection.

## Analysis and implementation

| Component | Implementation |
| --- | --- |
| Environment | Python 3.12 with pinned dependencies and a platform-aware hashed lock |
| Sensitivity inputs | AI extraction allocations and environmental parameters passed to `WaterToC` |
| Sobol estimation | One replicate mean per sampling-design row, with preserved ordering and within-configuration variance |
| Sensitivity reports | Separate outputs, estimands, and units |
| Distribution comparison | Matched background with each input fixed; descriptive KS distance |
| Moran statistic | Weights normalized on the induced masked graph |
| Spatial masks | Occupancy and occupancy-with-water evaluated separately |
| Pair correlation | Unordered-pair normalization, finite-sample correction, and exact annulus areas |
| Spatial null | Randomized strategy labels preserving positions and cooperator counts within agent types |
| Oscillation fitting | Offset-aware damped sinusoid per realization, with fit diagnostics |
| Wavelets | Centered per-run squared Morlet coefficients with edge exclusion and period limits |
| Tail diagnostics | Variance, trend, recurrence, and ADF results with undefined values retained |
| Model bounds | Finite nonnegative flows, capacity bounds, and occupancy constraints |
| Snapshots | Execution identifier, seed, feedback rule, and optional spatial recording |
| Sweeps | Stable identifiers and configuration, source, dependency, and input manifests |

Sensitivity ranges are specified by each experiment configuration.

Spatial sweeps use SHA256 seed mapping. The timeseries sweep uses the specified SHA1 seed mapping.

## Model dynamics

The original payoff rule compares row sums. With R 3, S 0, T 5, P 1, its cooperation-minus-defection sum equals 3(1 minus 2n). Before random reversal it cooperates at or below half capacity and defects above it. Human and AI agents share this response. No peer-strategy imitation or replicator selection is implemented.

Resource access has radius 1, environmental cooperation feedback radius 3, and the reported smoothed cooperation field radius 2. Agent positions remain fixed. Agents lacking accessible water retain their strategy; they record unmet requests. Global resource state includes water cells unused by agents.

Sequential withdrawals prioritize earlier agents for delivery; humans are inserted first. Given fixed preplanned nonnegative requests, final resource at each cell depends on total requests rather than withdrawal order. The allocation order determines individual delivery when requests exceed available stock.

Baseline renewal is sampled uniformly from 0.1 to 0.5 per water cell. At default density and population, expected baseline renewal is 36 units per step, whereas cooperative requested demand is 105. At theta 3 the repository multiplier is at most 1.25, giving an optimistic expected renewal bound of 45. Storage buffers finite runs; unmet demand must be distinguished from delivered consumption. This calculation is an expectation over landscapes, not a calibrated water-system prediction.

The reference replicator ODE is a different dynamical system. Its equilibrium is not a target against which ABM deviations can be attributed solely to spatial effects. The analysis reports finite-horizon dynamical and spatial statistics.

## Verification and execution

Run from `iccs2026/` using the pinned environment:

```bash
python -m pytest -q
python -m pip check
python reproducibility/capture_baseline.py
python -m water_toc.audit --replications 12 --steps 100 --output-dir results/replenishment_comparison
python -m water_toc.spatial_null --seed 42 --permutations 499
```

The baseline capture executes the untouched historical model and agent source, with its real dependencies, in a separate temporary package. Regression cases include a rectangular grid, no agents, and no resources. Tests also cover conservation accounting, occupancy, random-seed replay, logging independence, an analytical Ishigami Sobol benchmark, phase-cancelled oscillations, signed damping with an offset, masked Moran normalization, and a continuous uniform torus null. The torus null validates estimator normalization; actual lattice patterns require their own conditional null.

GitHub Actions runs the Python 3.12 test suite on Linux and macOS. Experiment manifests record configurations, dependency versions, source hashes, and input hashes.

## Sources

- [Published chapter](https://doi.org/10.1007/978-3-032-29909-3_8) and [official camera-ready manuscript](https://www.iccs-meeting.org/archive/iccs2026/papers/167870091.pdf).
- [Published supplementary information](https://doi.org/10.5281/zenodo.18713008).
- [SALib analysis workflow](https://salib.readthedocs.io/en/latest/user_guide/basics.html).
- [NumPy 2.3.1 release notes](https://github.com/numpy/numpy/releases/tag/v2.3.1), including the Apple M4 Accelerate fix.
