# Welfare and spatial traps in the coevolutionary commons

## Motivation

Human activity and AI infrastructure depend on a shared physical environment. Treating resource access for people and the operation of AI systems as independent questions can obscure this dependence. Human access cannot be assessed without considering sustained infrastructure demand, while infrastructure expansion cannot be assessed without considering the conditions that support human activity over time. Resource competition alone does not establish that AI activity has no social value; nor does the usefulness of AI remove the physical consequences of its operation.

[Malčić, Roy, and Waronig (2026)](https://doi.org/10.1007/978-3-032-29909-3_8) studied how local environmental feedback and strategic choices produce spatial traps: cooperative regions can coexist with persistent resource stress. The journal extension asks an inverse question: which admissible payoff structures best support human and AI activity together, and can those improvements be accompanied by escape from the persistent spatial patterns identified in the conference study?

The contribution is a study of coupled dynamics and their limits. Improved outcomes for both groups are a hypothesis to test, not a conclusion imposed on the simulations. The model measures resource-supported activity; it does not estimate all social benefits or harms of AI, or identify a uniquely correct social preference.

Two opposing positions motivate this question. One treats AI demand as inherently illegitimate whenever it competes with human access; the other treats the promised benefits of AI as grounds to disregard the conditions of long-term human access. We reject both positions. Useful AI activity and human well-being depend on the same physical world. The mathematical task is to establish their jointly attainable outcomes and their limits, rather than assign either an automatic priority. This is a stated philosophical position motivating the analysis, not a political conclusion inferred from a chosen welfare formula.

## Extension beyond the conference study

| Component | Conference study | Journal extension |
| --- | --- | --- |
| Strategic game | Fixed environmental interpolation | Two depleted-state payoff changes, with exact normalized reference recovery |
| Main computational task | Describe changes in cooperation, resource state, and spatial patterns | Optimize joint support for human and AI activity under explicit design and persistence restrictions |
| Mechanism comparisons | Variation in physical and feedback parameters | Equal-relative, resource-only, and unchanged-row-sum redistribution controls |
| Spatial interpretation | Identify persistent resource–strategy organization | Measure that organization alongside deliveries and welfare, and test paired changes under redesigned games |
| Reproduction record | Conference implementation and protocol | Independently runnable journal folder with indexed random inputs, individual deliveries, source-level flows, replay, and versioned release bundles |

## Payoff design

The model uses the normalized endpoint family

$$
A_{u,v}(n)=(1-n)\begin{pmatrix}t+u&v\\1&s\end{pmatrix}
+n\begin{pmatrix}1&s\\t&0\end{pmatrix}.
$$

Rows and columns are ordered cooperation, defection. The abundant-state game is held fixed. $u$ changes the depleted-state payoff for cooperating against a cooperator; $v$ changes it against a defector. Their contributions decrease linearly with abundance. The main values $t=2$ and $s=-0.5$ are obtained from the original payoffs by subtracting $P$ and dividing by $R-P$. At $u=v=0$, multiplying the normalized matrix by two and adding one to every entry recovers the conference matrix.

Require $t>1$, $s<0$, $u>-(t-1)$, and $v>s$. These preserve cooperation preference against either opponent at complete depletion and defection preference at complete abundance. Negative changes remain admissible; the study is not restricted to making cooperation more attractive.

The search uses relative changes $r_C=\frac{u}{t-1}$ and $r_D=\frac{v}{-s}$, with bounds $[-0.95,2]$. The primary budget is

$$
\sqrt{\frac{r_C^2+r_D^2}{2}}\le b.
$$

A common relative change $k$ costs $|k|$. The alternative budget is $\frac{|r_C|+|r_D|}{2}$. These bounds measure mathematical changes to incentives, not monetary costs. Main budgets are 0.25, 0.5, and 1; the reference game has zero cost.

For local cooperator fraction $x$, the expected advantage of defection is

$$
G(n,x)=x[(t-1)(2n-1)-u(1-n)]
 +(1-x)[(-s)(2n-1)-v(1-n)].
$$

An agent prefers cooperation when $G\le0$, with a configured probability of reversing that action. The derived equal-payoff resource levels are

$$
q_C=\frac{t-1+u}{2(t-1)+u},
\qquad
q_D=\frac{-s+v}{-2s+v}.
$$

These are strategic transition levels, not ecological safety limits. Equal relative changes keep them equal; unequal changes separate them. The slope ratio also changes with $u$ and $v$ and is never fixed independently.

Holding resource stock constant gives a simple discrete response calculation. If cooperation is preferable at composition $x$, the next large-population cooperation fraction is $1-\epsilon$; otherwise it is $\epsilon$. Matching incentives can reinforce either response level, whereas opposite-action incentives can alternate between them. This calculation concerns the fixed-environment synchronous response, not a replicator equation or a proof about the full spatial system.

## Matched comparison families

- Full local family: search $u$ and $v$ using actual local composition.
- Equal relative changes: $r_C=r_D$. The action depends on resources alone despite using the local observation rule.
- Optimized resource-only rule: hold $x=\frac{1}{2}$ and search $k=\frac{u+v}{t-1-s}$. Choose the least-cost matrix encoding of each $k$ under the same bounds and norm. This is stronger than unnecessarily restricting the encoding to $r_C=r_D$.
- Payoff redistribution: $u+v=0$. The original row sums are unchanged at every resource level, but local composition can change decisions. This isolates a mechanism that the original row-sum rule cannot detect.

All families use the same welfare definition and persistence requirement. Equal-relative and redistribution candidates are included in validation for the full family. The payoff objective does not alter agent targets, physical supply, or the outcome definition.

## Welfare: support for both kinds of activity

For agent $i$, fix an activity target $d_i$ before evaluating any candidate payoff structure. The reference target is the original model's higher, defection-level request: 1.5 times its cooperative request in the main scenario. The target does not decrease when an agent chooses cooperation. This allows additional delivery above the cooperative baseline to support additional activity; it does not label higher consumption as waste.

At each observed time step, calculate supported activity as

$$
s_i(t)=\min\left\{\frac{\mathrm{delivered}_i(t)}{d_i},1\right\}.
$$

$H$ is the mean of this quantity over human agents and observed steps; $A$ is the corresponding mean for AI agents. These separate measurements describe components of the same resource-coupled system. They do not assume independent dynamics.

The joint welfare index is

$$
W_\lambda=H^\lambda A^{1-\lambda},\quad 0<\lambda<1.
$$

At equal priority, $\lambda=\frac{1}{2}$, the index is $\sqrt{HA}$. Poor support for either group lowers joint support; complete loss of either group gives zero. At the endpoints, $\lambda=1$ means a human-only objective and $\lambda=0$ means an AI-only objective. The main optimization maximizes the expected index across independent realizations, not the index calculated from outcomes pooled across different realizations.

Equal priority is a declared modeling choice. It is not an estimate of the relative worth of people and infrastructure. Groups receive equal aggregate weight at $\lambda=\frac{1}{2}$ regardless of their water demand. A different group weight changes the welfare assumption. Individual deliveries and unserved periods are reported alongside group means to make their distribution available for assessment.

Sensitivity checks use $\lambda\in\{0,0.25,0.5,0.75,1\}$. A second utility calculation, $\frac{\log(1+s)}{\log(2)}$, tests diminishing benefit from additional supported activity. Re-scoring frozen policies describes their transfer to another welfare assumption. Claims about a new optimum require a new optimization under that assumption.

The fraction of additional use credited as useful is also tested at 0, 0.5, and 1. Let $c_i$ be the fixed cooperative request and $d_i$ the fixed higher request. With credited fraction $e$, supported activity is

$$
\frac{\min(\mathrm{delivered},c_i)
+e\max\{\min(\mathrm{delivered},d_i)-c_i,0\}}
{c_i+e(d_i-c_i)}.
$$

The primary $e=1$ reduces to $\frac{\mathrm{delivered}}{d_i}$, capped at one. At $e=0$ additional use above $c_i$ receives no extra credit; this is a sensitivity assumption rather than a claim that the additional use is waste. This comparison addresses uncertainty about the benefits of higher extraction without changing physical demand or strategic payoffs.

## Persistent local stress

The ICCS spatial trap is a persistent resource–strategy pattern, not a published numerical stock cutoff or a demonstrated delivery-shortage condition. Cooperative regions and chronically stressed peripheral regions are part of that description. Its welfare consequences must be measured rather than inferred from its name. A mean stock of 0.38 does not determine which locations are stressed or whether requests are being met.

The conference paper's moderately stressed equilibria are distinct from total collapse. The reference stock-based constraint measures sustained local stress separately from the auxiliary depletion score below 0.2.

For every geometrically accessible resource cell, track consecutive observed steps below a fixed resource reference. In the main specification, the reference is $n=0.5$ and the duration is 150 consecutive steps. The reference corresponds to the preferred-action change in the original game; it is not an empirically established ecological minimum. The duration is held fixed when the observation horizon changes.

A realization has a persistent local stress event if any accessible cell remains below the reference for at least the specified duration. All accessible cells remain in the assessment, including empty ones. The metric also detects uniform persistent stress; it is a protective condition broader than the particular spatial pattern reported in the conference paper. A spatial trap is investigated through resource maps, strategy patterns, persistence, and longer-horizon behavior rather than inferred from this indicator alone.

The reference stock-based restriction is

$$
\Pr(\text{persistent local stress during the observation window})\le0.1.
$$

The numerical reference, duration, and tolerated probability define the experiment; they are not universal safety standards. Reference levels $\{0.3,0.4,0.5,0.6\}$ and durations $\{75,150,300\}$ are included in recorded-trajectory sensitivity analysis. Claims about the best feasible outcome under another definition require reoptimization. Temporary shortages can remain relevant to welfare without automatically violating this persistence condition.

## Constrained optimization

The main problem maximizes expected joint welfare over admissible payoff structures, subject to the payoff-change budget and the probability restriction above. Cooperation and stored resource are explanatory measurements rather than optimization targets.

A second matched experiment, `delivery_reliability.json`, restricts persistent unmet requests instead. It tracks each agent's delivered/requested fraction against one, with a relative floating-point tolerance of $10^{-12}$, and applies the same duration and probability allowance. Both conditions use the general site-fraction setting: sites are accessible resource cells in the stock condition and agents in the delivery condition. This experiment changes the protection goal, not the definition of the historical phenomenon.

Constructed model states can maintain low reserves while serving every request. This rules out a logical identification of low stock with delivery failure; it does not establish the frequency of such states in the reference ensemble. Conversely, a group mean can conceal local shortfalls. `spatial_characterization.csv` reports stock heterogeneity, spatial clustering of cooperation, local resource–cooperation association, request fulfillment, and the worst agent's mean fulfillment. Constant fields have undefined spatial correlation and are reported as missing, not as evidence of no spatial structure.

An escape claim requires paired evidence that the resource–strategy pattern changes and welfare improves under the same physical conditions. The stock-persistence indicator alone is insufficient. If a reference spatial pattern has no measured welfare cost, the result is a persistent spatial pattern without that demonstrated cost; delivery failures must not be invented or the old phenomenon renamed to fit the objective.

During training and validation, the probability restriction is evaluated using the observed fraction of independent realizations with persistent stress. Candidate selection first prefers compliance with that restriction, then higher welfare, then smaller payoff change. If no candidate meets the empirical restriction, the selected diagnostic candidate is explicitly marked as such. An unsuccessful search does not prove that no admissible game can succeed.

Differential evolution searches the two-parameter family offline. The chosen $u$ and $v$ remain fixed throughout a realization; this is not a controller adapting after observing held-out outcomes. Its objective is one minus expected welfare; its constraints include the payoff bound and the empirical stress probability. The three one-dimensional comparators apply the same ordering. The resource-only comparator uses the least-cost encoding of its effective resource threshold, avoiding an artificial budget disadvantage. Equal-relative and zero-sum redistribution controls are retained separately.

Independent held-out realizations assess both welfare and protection. The stress probability receives an exact one-sided binomial upper confidence bound. A low observed failure fraction is distinguished from statistical support for the required probability. Main selected-policy protection claims use a correction for the number of distinct tested policies and pressures. These are finite-horizon results for the simulated distribution, not guarantees about all future trajectories.

## Experiments and interpretation

1. Run the separate characterization stage to establish the reference dynamics and map evaluated payoff structures. Record human support, AI support, joint welfare, continuous local stress, cooperation, and spatial resource patterns together.
2. Compare the local-composition family against the optimized resource-only, equal-relative, and redistribution families under matched constraints. The headline setting is pressure 1 and payoff-change budget 0.5. Other budgets and pressures describe how attainable joint support changes.
3. Compare the selected design with the embedded conference game on paired held-out worlds. Report changes in both human and AI support, including uncertainty. A claim that both improve requires evidence for both changes and support for the stress restriction.
4. Reoptimize the headline setting under human-only and AI-only priorities, using `human_only_priority.json` and `ai_only_priority.json`. Keep physical conditions and constraints matched. Compare these designs under all reported welfare priorities. This tests the consequences of competing priorities without assuming the equal-priority design dominates each group's own optimum.
5. Test frozen designs under changes in demand pressure, demand asymmetry, spatial concentration, initial strategies, feedback, grid size, and update timing. The main delivery calculation divides shortages proportionally among simultaneous requests at each source. Serving humans first is an explicit inherited-allocation comparison; welfare conclusions must state this assumption.
6. Re-score archived trajectories under alternative welfare and stress definitions. Separate these evaluations from reoptimized results. Use longer trajectories and different initial conditions to investigate whether apparent improvements persist.

The intended finding is a characterization of when payoff redesign can improve joint support, what trade-offs remain, and where persistent local stress prevents otherwise attractive activity levels. A negative result, or an improvement limited to a narrow regime, remains part of that characterization. The model cannot establish that a particular social priority is morally correct, or that a real location can support a specified infrastructure project.

## Reproducibility and scope

The main configuration uses 2,000 steps, excluding the first 500; 8 separate characterization realizations, 8 training, 16 validation, and 64 held-out realizations; and disjoint indexed random inputs. Candidate policies are frozen before held-out evaluation. All targets, welfare assumptions, stress definitions, random inputs, source versions, individual outcomes, and analysis settings are archived. Replaying a realization checks the recorded individual deliveries and welfare results as well as the physical trajectory.

The `smoke.json` configuration is a short software check, not evidence for scientific conclusions. Production experiments require an explicit execution budget. [Running the experiments](running-experiments.md) describes commands and [implementation](implementation.md) specifies the event order and output records.

## Discussion questions tied to the evidence

- Joint gains: determine whether changing the payoff structure improves support for both groups relative to the embedded conference game, and where gains require a trade-off. An improved combined score alone does not establish gains for both groups.
- Competing priorities: explain how human-only, AI-only, and joint objectives change selected incentives under the same resource conditions. A larger joint score under its own optimizer is not independent evidence that its normative weights are correct.
- Spatial traps and consequences: distinguish a persistent resource–strategy pattern from demonstrated lost activity, unmet requests, or limited recovery. State whether the same conclusion survives both protection specifications and different definitions of additional-use benefit.
- Mathematical limits: examine the limits imposed by spatial access, demand pressure, the permitted payoff changes, and the fixed abundant-state game. Failure to find a successful design is not a proof that none exists.
- Political interpretation: discuss why shared physical dependence complicates both automatic rejection of AI demand and disregard for human access. Identify which conclusions follow from the simulations and which remain philosophical commitments. The model measures supported activity, not the full social value of an AI system or a person's life.
