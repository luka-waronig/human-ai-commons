# Implementation and experiment contract

## Independent numerical model

The `human_ai_commons` package uses NumPy and SciPy directly. Geometry is represented by unique resource coordinates, fixed agent coordinates, and toroidal Chebyshev-distance incidence matrices. Neighbor cells are counted once even on a thin or small torus. There are no imports from the conference implementation.

Worlds draw Bernoulli resource locations and uniform baseline renewal. Given the landscape, agents are placed uniformly among eligible cells with remaining occupancy. In the main ensemble, eligible cells have a geometrically accessible resource. Resource layouts are redrawn only when there are no resources or insufficient eligible placement capacity. `draw_attempts` records those draws. Placement conditioning preserves the specified landscape distribution at larger grid sizes. The unconditioned-placement robustness variant allows permanent isolation. Initial cooperation is rounded to the nearest available count independently within each type and randomly assigned within that type.

The model initializes stocks at capacity. At each time step it replenishes from pre-step strategies, chooses targets using replenished stocks, plans actions, withdraws, and records post-withdrawal metrics. Simultaneous revision reads a frozen strategy array. Sequential-planning sensitivity revises once per agent in a random order while holding resource stocks fixed until every agent has planned. The focal agent is excluded from game-composition observations. Ecological feedback includes all agents in its neighborhood. The no-neighbor payoff convention is x=1/2; the empty ecological neighborhood follows the selected feedback-law convention.

Cooperative requests are scaled so their sum divided by baseline renewal on geometrically accessible cells equals the specified pressure. The AI/human ratio changes demand allocation at fixed total cooperative load. Actual delivery can be less than requested. The main `proportional` allocation divides a source's available stock in proportion to simultaneous requests when supply is insufficient. The `human_priority` comparison serves humans first and then AI, with stable order within each type. Allocation affects individual delivery, while the final stock for fixed planned requests depends on the total request at each source.

Each realization reports mean requests, mean deliveries, and the fraction of steps with no delivery for every agent, including agents without resource access. Recorded trajectories retain each agent's selected source, requests and deliveries at every step, its type, and its cooperative reference request. Each source's actual renewal and total withdrawal are also archived, allowing source-by-source balance checks. The reference request is a model quantity, not an assumed welfare target. Group totals can be reconstructed from these individual records.

Resource summaries report the fraction of observed steps below the depletion level and the longest continuous period below that level for each accessible resource cell. Durations count only the observation window after the excluded initial steps. Cumulative deficit and continuous duration describe different outcomes: the same total deficit can arise from repeated short shortages or one uninterrupted shortage.

The service target is fixed at each agent's defection-level request for the scenario. Per-step delivery divided by that target, capped at one, is averaged within each agent type. The main search keeps both group means and local depletion as separate objectives. A weighted geometric mean is retained only for optional preference comparisons and reporting. The target never follows the agent's chosen strategy. The optional `log1p` utility transforms each supported fraction before averaging. Persistent local stress is assessed with its own reference level and continuous duration; the lower resource reference in `depletion_diagnostic` defines the continuous depletion objective, separately from the stress constraint.

`additional_use_value` specifies the fraction of delivery above the fixed cooperative request credited as useful. Its primary value is one. Sensitivity values zero and one-half distinguish the valuation of additional activity from its physical extraction. Both request references remain fixed when strategies change.

The primary payoff representation stores u, v, t, and s. The optimizer works in u/(t-1) and v/(-s), which are converted to the stored matrix entries before every evaluation. Derived transition levels and slope ratios are exported separately in `payoff_coordinates.csv`. The resource-only encoding minimizes the configured norm subject to its effective relative change and active parameter bounds. The redistribution comparison enforces u+v=0 exactly.

Persistence has two explicit bases. `resource_stock` tracks accessible source cells below a configured stock reference. `request_fulfillment` tracks agents whose delivered/requested fraction is below its reference. Requests remain strategy-dependent physical quantities, while welfare targets remain fixed. Numerical equality in request fulfillment uses relative tolerance 1e-12. The shared `allowed_site_fraction` counts resource cells or agents according to that basis. Neither indicator alone is a classification of the ICCS spatial pattern.

`spatial_characterization.csv` reports mean and spatial variation of source stocks, worst-agent request fulfillment, and spatial associations. Cooperation autocorrelation is Moran's I on a symmetric radius-one agent graph with no self-links; co-located distinct agents are neighbors. Resource–cooperation association is Pearson correlation across accessible source cells between time-mean stock and time-mean cooperation within the ecological radius. These are descriptive associations, not causal effects. Undefined correlations remain missing.

## Paired randomness and archived worlds

The stream version `sha256-indexed-PCG64-v1` hashes the base seed, independent split, realization index, stream name, and time step. World generation, target ranks, action reversals, planning order, optimization, and analysis are independent streams. At each step the same agent-indexed random inputs are available even if a candidate has an empty target set. Pressure and policy are omitted from the world/trajectory stream keys so comparisons share exogenous inputs. Separate train, validation, test, and robustness labels prevent reuse of holdout worlds for selection.

Generated worlds are immutable and identified by content. Held-out trajectories archive the actual target ranks, reversal uniforms, and planning permutations, not only a seed. A source snapshot, complete resolved configuration, numerical versions, and file hashes accompany every bundle. Runtime use does not require Git history or access to any parent folder.

## Stages and controls

`characterize` evaluates the original normalized game and forced-action controls on a separate realization split before any search. It records individual deliveries and resource/strategy trajectories. Its purpose is to establish the reference pattern and measured consequences independently of optimized candidates. The stage can run and be analyzed on its own; training includes it automatically if needed.

`train` evaluates fixed grids and two rounds of Pareto refinement. It minimizes the vector of negative human support, negative AI support, and spatial-tail depletion. The original game and feasible grid nodes are evaluated explicitly. Refinement samples shrinking neighborhoods of diverse current trade-offs. Evaluations are cached and shared across budgets; no welfare weight guides the main search. The scalar differential-evolution implementation is used only by optional human-only and AI-only configurations.

`validate` assesses up to 12 candidates per family and problem on a disjoint realization set. Non-dominated feasible candidates are reduced to at most six diverse designs, preserving each objective's end. Equal-relative and redistribution candidates are included in local-family assessment. Least-violation candidates remain labelled diagnostics when no empirically feasible design is found. The frozen artifact stores `families` containing lists, not `winners` containing one design.

`test` evaluates the complete frozen sets and original game on new paired realizations. It records all three objectives and exact one-sided binomial bounds, corrected across distinct tested policy/pressure pairs. Test results never guide another search or shortlist. Analysis recomputes empirical frontiers among the tested designs using sample means; this is estimation, not proof of global optimality or population-level dominance. Individual mean and paired-change bootstrap intervals are descriptive. Primary paired-mean intervals have finite-sample coverage for independent bounded realizations; their construction is given below. The total error allowance is preallocated across main/robustness outcome and resource groups.

`robustness` transfers up to three validation-selected objective extremes per family to the listed scenario, layout, schedule, initial-condition, horizon, and grid changes, using 256 independent realizations per variant. It does not reoptimize. Resource-confidence corrections cover the complete set of variant/design pairs. Paired simultaneous outcome comparisons also cover all variants together. The representative designs are frozen before test data are observed. Metric-floor and tail-fraction changes use archived trajectories through `rescore`; changes to these outcome definitions do not establish newly optimized frontiers.

The capacity calculation is a generous static routing relaxation for a finite-window **zero-depletion** target at the auxiliary resource reference. It does not establish impossibility of meeting the persistent-stress restriction, and a passing relaxation does not establish feasible realized dynamics. Permanently isolated agents are excluded from the depletion-capacity demands: an agent that can never withdraw cannot itself force accessible cells below the floor. Its zero delivery is retained in welfare measurements.

## Confidence intervals

For each frozen design and outcome, let $D_i\in[-1,1]$ be its difference from the original game on matched realization $i$. Set $X_i=(D_i+1)/2$. For a possible mean $m>0$, the fixed mixture is

$$
E(m)=\sum_{j=1}^{9}w_j\prod_{i=1}^{N}\left(1-\rho_j+\rho_j\frac{X_i}{m}\right),
$$

where $\rho=(0.02,0.05,0.1,0.2,0.4,0.6,0.8,0.95,1)$, the first eight weights are $1/16$, and the last is $1/2$. If the true mean of $X_i$ is at most $m$, each nonnegative factor has expectation at most one. Independence between realizations therefore gives $\mathbb{E}[E(m)]\leq1$. This requires no independence between outcomes or designs evaluated on the same realization.

For $K$ distinct comparisons and group error allowance $\alpha$, Markov's inequality bounds the probability of $E(m)\geq 2K/\alpha$ by $\alpha/(2K)$. Inverting this monotone test gives a lower mean bound. Applying the same test to $1-X_i$ gives the upper bound. Combining both sides and all $K$ comparisons gives simultaneous coverage of at least $1-\alpha$. The implementation uses log products and 55 bisection steps, retaining the conservative side of the bracket. Identical-policy comparisons are known zero differences and do not consume additional comparisons. The fractions and weights are fixed before evaluation. This is an application of the [bounded-mean betting framework](https://doi.org/10.1093/jrsssb/qkad009), not a new statistical estimator.

The main sample contains 320 matched realizations. Each robustness variant uses 256. Sample counts are fixed in advance; there is no stopping based on an emerging result. Marginal bootstrap intervals and separate Hoeffding bounds are descriptive diagnostics and do not replace the primary intervals after seeing the data.

## Execution

Preflight computes workload upper bounds and exact resource-assessment power at the configured planning probability without generating worlds. Insufficient production sample sizes are rejected before simulations. Execution refuses a stage whose cap exceeds the supplied limit; the default is 10,000 model steps. Checkpoint files record completed evaluations, and bundles reject configuration, dependency, source, or checksum changes. Interrupted stages may reuse completed cached evaluations and complete once their full outputs are written.

The source version is part of the computation identity. After code changes, create a new bundle rather than mixing different implementations. Once production results exist, release their full bundle and exact source together. Reports from `analyze` identify their input hashes; regenerate them after additional stages finish.
