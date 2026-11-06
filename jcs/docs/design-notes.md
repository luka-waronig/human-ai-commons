# Alternative formulations for the journal extension

**Based on:** *Emergent Spatial Traps in the Coevolutionary Commons: Human-AI Cooperation under Environmental Feedback* (ICCS 2026)

**Scope:** Alternative payoff and welfare formulations. The executable study is specified in the [research protocol](research-plan.md).  
**Core aim:** turn the ICCS discovery of spatial traps into a constrained optimization and feasibility study of when strategic governance can, and cannot, prevent persistent localized depletion.

---

# Working title

## When Can Governance Prevent Spatial Traps?
### Feasibility Boundaries in Human–AI Resource Commons under Environmental Feedback

Alternative:

## Beyond Managed Scarcity: Constrained Optimization of Spatial Coevolutionary Commons

The feasibility-boundary formulation concerns the distinction between **governance-correctable** and **physically infeasible** regimes.

---

# One-sentence thesis

The ICCS paper showed that high cooperation can coexist with persistent local resource stress. The journal extension asks a harder question:

> **For a given spatial resource system and relative AI demand, does there exist any admissible strategic payoff structure that can sustain useful human and AI activity without allowing persistent localized depletion—and, if so, what does that feasible payoff region look like?**

This is the central move beyond the conference paper.

The paper is therefore not primarily about maximizing water, maximizing cooperation, or finding one numerically “best” payoff matrix. It is about identifying the **feasible operating envelope** of a spatial coevolutionary commons and determining when strategic redesign is sufficient versus when the physical demand on the resource system is too high for governance alone to solve.

---

# Draft abstract

Spatial commons can remain globally functional while becoming locally degraded. In our previous work, a spatial coevolutionary game between human and AI-infrastructure agents produced managed-scarcity regimes in which relatively high cooperation coexisted with persistent local resource stress. Here we reformulate this problem as a constrained multiobjective optimization over the strategic payoff structure itself. Rather than maximizing cooperation or mean resource abundance, we search for payoff configurations that preserve human and AI service while satisfying hard local resource constraints and suppressing persistent localized depletion. To avoid dependence on weak real-world calibration, the system is analysed in dimensionless form using relative demand pressure, AI-to-human demand ratios, normalized storage and renewal, and spatial concentration. This allows us to ask whether a given physical resource regime is **governance-correctable**—meaning that at least one admissible payoff structure satisfies all system constraints—or **physically infeasible**, meaning that no strategic design in the considered game family can do so. The resulting Pareto and feasibility surfaces characterize how the strategic design space contracts as resource pressure increases and identify the transition beyond which incentive redesign can no longer substitute for physical capacity. The framework converts the descriptive discovery of spatial traps into a computational method for locating the boundary between correctable governance failure and irreducible resource insufficiency.



---

# 1. Motivation and step beyond the ICCS paper

The ICCS study fixed the payoff structure and asked what dynamics emerge as environmental feedback strength, water capacity, and resource density change. It found that:

- greater aggregate resource abundance can weaken cooperation;
- high cooperation can coexist with a moderately stressed environment;
- cooperation and resource availability organize spatially;
- some configurations settle into a managed-scarcity regime rather than recovering to high abundance;
- local feedback and spatial structure are therefore essential to the dynamics.

The extension concerns strategic design and feasibility rather than an enlarged physical-parameter sweep.

The natural next question is:

> If the spatial trap is an emergent consequence of the game-environment coupling, can the game itself be redesigned so that the trap is no longer an admissible or Pareto-optimal outcome?

A still deeper question follows:

> Is every spatial trap correctable through strategic redesign, or are there resource-demand regimes in which no payoff structure can simultaneously protect local resource minima, human service, and useful AI activity?

This gives the extended paper a substantially different scientific purpose.

The ICCS paper was **forward analysis**:

\[
\text{fixed strategic game}
\rightarrow
\text{spatial dynamics}
\rightarrow
\text{emergent traps}.
\]

The journal paper becomes **constrained design and feasibility analysis**:

\[
\text{family of strategic games}
\rightarrow
\text{spatial dynamics}
\rightarrow
\text{physical objectives and constraints}
\rightarrow
\text{feasible / infeasible strategic regimes}.
\]

Every simulation still runs forward in time. The “inverse” aspect is only the research question: we search over admissible strategic structures to discover which ones produce acceptable physical system behaviour.

---

# 2. Scientific question

## Primary research question

> **Under what combinations of relative resource demand, environmental responsiveness, and spatial concentration can strategic payoff redesign prevent managed-scarcity traps while sustaining both human and AI activity?**

## Secondary questions

1. Does there exist a critical normalized demand pressure beyond which **no admissible payoff structure** satisfies the local resource and behavioural constraints?
2. How does the critical pressure change with the AI-to-human demand ratio?
3. How does spatial concentration of demand or resource access shift the feasibility boundary?
4. Is there a regime in which stronger strategic responsiveness stops helping or produces oscillatory / unstable behaviour?
5. Do Pareto-optimal games share a common structural property, such as a characteristic transition point at which extraction restraint becomes individually advantageous?

These questions produce interpretable conclusions even without geographic calibration.

---

# 3. What is deliberately *not* being claimed

This is not a calibrated municipal water-demand model.

Geographically calibrated claims, such as the following, are outside the model scope:

> “A city can safely support a 100 MW data centre consuming X litres per day.”

Without detailed local hydrology, utility-network data, water rights, cooling technology, seasonal variability, and infrastructure constraints, such a claim would be unjustified.

Instead, the study seeks **dimensionless structural results**:

> How do relative demand, renewable supply, storage, spatial concentration, and strategic response determine whether safe coexistence is possible at all?

This is a complex-systems contribution rather than a site-specific forecast.

A later empirical study could calibrate the dimensionless quantities to a particular municipality.

---

# 4. Published model retained as the forward system

Continuity with the ICCS model rests on the following shared mechanisms.

The shared mechanisms are:

- two-dimensional spatial environment;
- human and AI-infrastructure agent types;
- local resource access;
- normalized water state \(n_j(t)\in[0,1]\);
- local environmental regeneration;
- environmental feedback strength \(\theta\);
- stochastic strategy deviation;
- human versus AI extraction asymmetry;
- \(K_{\max}\), resource density, and spatial layout as environmental conditions.

The published payoff formulation is

\[
A(n)
=
(1-n)
\begin{bmatrix}
T & P\\
R & S
\end{bmatrix}
+
n
\begin{bmatrix}
R & S\\
T & P
\end{bmatrix}.
\]

The ICCS model uses fixed values

\[
(R,S,T,P)=(3,0,5,1).
\]

The journal extension replaces this single game by a **minimal parameterized family of games**.

The ICCS payoff configuration provides the strategic reference for regression checks.

---

# 5. Parameterizing the strategic game

## 5.1 Why not optimize four arbitrary payoff numbers?

The absolute values of \(R,S,T,P\) are not the most meaningful degrees of freedom. Strategic behaviour depends on **relative payoff differences**.

This alternative parameterizes the *advantages of defection versus cooperation* in the depleted and replete environmental regimes.

Let

\[
\delta_C^{(0)} = T_0-R_0 < 0,
\qquad
\delta_D^{(0)} = P_0-S_0 < 0
\]

denote the defection advantage in a depleted environment \(n=0\). Negative values mean that cooperation is individually preferable.

Similarly,

\[
\delta_C^{(1)} = T_1-R_1 > 0,
\qquad
\delta_D^{(1)} = P_1-S_1 > 0
\]

denote the defection advantage in a replete environment \(n=1\).

Because a common positive scaling of all payoff differences is redundant for deterministic best-response ordering, set

\[
\delta_C^{(0)}=-1.
\]

Then define

\[
\alpha = -\delta_D^{(0)} > 0,
\qquad
\beta = \delta_C^{(1)} > 0,
\qquad
\gamma = \delta_D^{(1)} > 0.
\]

The environment-dependent incentive differences become

\[
\boxed{
\delta_C(n)=-(1-n)+\beta n
}
\]

and

\[
\boxed{
\delta_D(n)=-\alpha(1-n)+\gamma n.
}
\]

Thus the optimization searches

\[
\boxed{
\mathbf p=(\alpha,\beta,\gamma).
}
\]

The published game becomes a particular point in this family after normalization.

A two-parameter subfamily follows by imposing an additional relation on the three normalized endpoint differences.

---

# 6. Local strategy evaluation

Local strategy evaluation uses both local water state \(n_i\) and local cooperation composition \(x_i\).

Let

\[
x_i(t)\in[0,1]
\]

be the fraction of cooperators in the relevant local interaction neighborhood.

The expected payoff advantage of defection is

\[
\Delta\pi_i
=
x_i\,\delta_C(n_i)
+
(1-x_i)\,\delta_D(n_i).
\]

Substituting,

\[
\Delta\pi_i
=
x_i[-(1-n_i)+\beta n_i]
+
(1-x_i)[-\alpha(1-n_i)+\gamma n_i].
\]

The indifference boundary \(\Delta\pi_i=0\) is

\[
\boxed{
n^*(x)
=
\frac{x+\alpha(1-x)}
{x+\alpha(1-x)+\beta x+\gamma(1-x)}.
}
\]

Interpretation:

> \(n^*(x)\) is the normalized local resource state at which cooperation and defection are equally attractive for a given local strategic composition.

The scientific question is whether feasible, trap-free systems consistently require the strategic transition to occur before local resource stress becomes severe.

---

# 7. Dimensionless physical description

Dimensionless ratios describe the physical system without dependence on absolute litres.

## 7.1 AI-to-human demand ratio

\[
\boxed{
\chi
=
\frac{Q_{\mathrm{AI}}}{Q_{\mathrm{H}}}
}
\]

where \(Q_{\mathrm{AI}}\) and \(Q_{\mathrm{H}}\) are aggregate baseline AI and human demands over the same period.

## 7.2 Total demand pressure

\[
\boxed{
\Pi
=
\frac{Q_{\mathrm{H}}+Q_{\mathrm{AI}}}
{R_{\mathrm{renew}}}
}
\]

where \(R_{\mathrm{renew}}\) is total effective renewable resource input over the same period.

The definition follows the resource update and realized model quantities.

## 7.3 Storage buffer

\[
\boxed{
B
=
\frac{K_{\mathrm{tot}}}
{Q_{\mathrm{H}}+Q_{\mathrm{AI}}}
}
\]

which measures normalized storage relative to demand.

## 7.4 Spatial concentration

A normalized concentration statistic can describe how unevenly demand is imposed across accessible resource cells:

\[
C_Q
=
\frac{
\sum_j s_j^2 - 1/M
}{
1-1/M
},
\]

where \(s_j\) is the fraction of total demand assigned to resource zone \(j\), and \(M\) is the number of active resource zones.

This gives the physical parameter space

\[
(\Pi,\chi,B,C_Q,\theta,\ldots)
\]

without geographic hydrological calibration.

---

# 8. The welfare problem

The following quantities are excluded as stand-alone welfare objectives:

- mean water level;
- total water stored;
- cooperation fraction;
- the numerical payoff values themselves.

Water and cooperation are system conditions, not the ultimate objective.

The proposed multiobjective welfare vector is

\[
\boxed{
\mathbf W(\mathbf p)
=
\left[
B_{\mathrm{AI}},
H_{\mathrm{human}},
-L_{\mathrm{PLD}}
\right].
}
\]

These three components are calculated from physical ABM outputs, independently from the strategic payoff values.

---

# 9. AI benefit

The ABM does not simulate compute throughput. Resource-supported activity is distinct from measured compute output.

Instead define normalized **AI operational service**:

\[
s_{i,t}^{A}
=
\min
\left(
\frac{q_{i,t}^{\mathrm{served}}}
{q_{i,t}^{\mathrm{target}}},
1
\right).
\]

Then

\[
\boxed{
B_{\mathrm{AI}}
=
\frac{1}{N_A T}
\sum_{i\in A}\sum_t s_{i,t}^{A}.
}
\]

This measures the fraction of requested AI operational demand successfully supported by the resource system.

---

# 10. Human welfare

Define human welfare through satisfaction of essential household demand:

\[
s_{i,t}^{H}
=
\min
\left(
\frac{q_{i,t}^{\mathrm{served}}}
{q_{i,t}^{\mathrm{essential}}},
1
\right).
\]

Then

\[
\boxed{
H_{\mathrm{human}}
=
\frac{1}{N_H T}
\sum_{i\in H}\sum_t s_{i,t}^{H}.
}
\]

Normalizing the essential requirement to the original cooperative human demand is one reference choice; its sensitivity is part of the welfare specification.

---

# 11. Persistent Local Depletion (PLD)

The physical quantity measured here is local resource depletion: some locations remain persistently depleted even when global averages look acceptable.

For resource cell \(j\), define cumulative normalized deficit below a safe state \(n_{\mathrm{safe}}\):

\[
D_j
=
\frac{1}{T}
\sum_{t=1}^{T}
\max
\left(
0,
n_{\mathrm{safe}}-n_j(t)
\right).
\]

Then define **Persistent Local Depletion** as the average deficit among the worst affected fraction \(q\) of cells:

\[
\boxed{
L_{\mathrm{PLD}}
=
\operatorname{CVaR}_{q}
\left(
D_1,\ldots,D_M
\right).
}
\]

For example, \(q=0.10\) evaluates the worst 10% of resource cells.

This targets the physical pathology that defines managed scarcity: acceptable global behaviour coexisting with persistent localized depletion.

The values of \(n_{\mathrm{safe}}\) and \(q\) are sensitivity parameters rather than universal empirical constants.

---

# 12. Hard constraints

## 12.1 Local resource floor

\[
\boxed{
\Pr
\left[
\min_{j,t} n_j(t)\ge n_{\min}
\right]
\ge 1-\varepsilon_n.
}
\]

This makes local collapse unacceptable even if global mean resource level remains high.

## 12.2 Minimum cooperation condition

If retained:

\[
\boxed{
\Pr
\left[
\bar{x}_{\mathrm{post}}\ge x_{\min}
\right]
\ge 1-\varepsilon_x.
}
\]

Cooperation is not maximized; it is only a feasibility condition. Its necessity is assessed separately from the physical constraints.

---

# 13. The constrained multiobjective problem

\[
\boxed{
\max_{\alpha,\beta,\gamma}
\left(
B_{\mathrm{AI}},
H_{\mathrm{human}},
-L_{\mathrm{PLD}}
\right)
}
\]

subject to

\[
\Pr
\left[
\min_{j,t}n_j(t)\ge n_{\min}
\right]
\ge 1-\varepsilon_n,
\]

and, if retained,

\[
\Pr
\left[
\bar{x}_{\mathrm{post}}\ge x_{\min}
\right]
\ge 1-\varepsilon_x.
\]

There is intentionally **no single scalar welfare weight**.

The result is a feasible Pareto set of strategic payoff structures.

---

# 14. The deeper question: can governance solve the problem at all?

For a physical scenario

\[
\mathbf z
=
(\Pi,\chi,B,C_Q,\theta,\ldots),
\]

define

\[
\mathcal F(\mathbf z)
=
\left\{
\mathbf p:
\mathbf p
\text{ satisfies all hard constraints under }\mathbf z
\right\}.
\]

Then classify:

\[
\boxed{
\mathcal F(\mathbf z)\neq\varnothing
}
\]

as **governance-correctable**, versus

\[
\boxed{
\mathcal F(\mathbf z)=\varnothing
}
\]

as **physically infeasible within the modeled intervention class**.

The feasible set describes the admissible designs beyond a single optimum.

---

# 15. Critical demand boundary

Increase normalized demand pressure \(\Pi\) for fixed

\[
(\chi,B,C_Q,\theta,\ldots).
\]

Define

\[
\boxed{
\Pi_c
=
\sup
\left\{
\Pi:
\mathcal F(\Pi,\chi,B,C_Q,\theta,\ldots)
\neq\varnothing
\right\}.
}
\]

\(\Pi_c\) is the maximum normalized demand pressure that remains governance-correctable within the modeled game family.

The study then asks how \(\Pi_c\) changes with

\[
\chi,\quad C_Q,\quad B,\quad \theta.
\]

The practical interpretation is simple:

> Strategic governance may enlarge the safe operating envelope, but beyond some physical load no strategic design can compensate for insufficient resource capacity.

---

# 16. Interpretation of the Pareto set

For a fixed physical scenario, the Pareto surface shows the best achievable combinations of

\[
B_{\mathrm{AI}},
\qquad
H_{\mathrm{human}},
\qquad
L_{\mathrm{PLD}}.
\]

As \(\Pi\) increases, the relevant result is not merely movement along a generic Pareto front. It is the **contraction and eventual disappearance of the feasible Pareto set**.

Conceptually:

1. **Low pressure:** many payoff structures satisfy all constraints.
2. **Intermediate pressure:** the feasible Pareto surface contracts and genuine trade-offs emerge.
3. **Critical pressure:** only a narrow set of strategic configurations remains feasible.
4. **Beyond \(\Pi_c\):** no feasible strategic design remains.

---

# 17. Regime classification

A phase diagram summarizes the candidate regimes:

\[
(\Pi,\chi)
\longrightarrow
\begin{cases}
\text{robustly feasible}\\
\text{strategically sensitive}\\
\text{managed scarcity}\\
\text{oscillatory}\\
\text{physically infeasible}
\end{cases}
\]

for different values of \(C_Q\), \(B\), and \(\theta\).

A further hypothesis is a partial collapse of these transitions under a small number of dimensionless groups.

Such a relation would provide a reduced description of the simulated regime boundaries.

---

# 18. Hypotheses

**H1 — Feasibility boundary.** There exists a critical normalized demand pressure \(\Pi_c\) above which no admissible payoff structure satisfies local resource constraints while maintaining human and AI service.

**H2 — Demand asymmetry.** Increasing \(\chi=Q_{\mathrm{AI}}/Q_{\mathrm{H}}\) reduces the feasible strategic region unless compensated by greater storage, renewal, or spatial distribution.

**H3 — Spatial concentration.** At fixed total demand pressure, increasing \(C_Q\) reduces the feasible region because aggregate abundance cannot protect the most heavily loaded local resource zones.

**H4 — Diminishing governance returns.** Strategic redesign substantially expands the feasible region under moderate resource pressure but has diminishing effect near physical insufficiency.

**H5 — Over-response may destabilize.** Highly responsive payoff structures may replace persistent depletion with oscillatory or unstable dynamics, so stronger incentives are not necessarily monotonically better.

These are hypotheses, not conclusions.

---

# 19. Computational study design

## Stage A — Freeze and reproduce the ICCS model

A fixed source revision defines the software baseline.

Reference comparison:

> The original payoff configuration and environmental parameters provide the basis for assessing qualitative trends and representative statistics.

## Stage B — Add the parameterized game family

Add a small game module exposing

```python
PayoffParameters(alpha, beta, gamma)
```

and functions for

```python
delta_c(n, params)
delta_d(n, params)
expected_payoff_gap(n, x_local, params)
```

## Stage C — Record physical service and local resource trajectories

Record:

- requested human demand;
- delivered human demand;
- requested AI demand;
- delivered AI demand;
- per-cell water state \(n_j(t)\);
- local cooperation \(x_j(t)\);
- global cooperation;
- strategy transitions;
- spatial cluster statistics.

The return value of `consume_water_at()` is the amount actually delivered, permitting direct measurement of unmet demand.

## Stage D — Reparameterize physical scenarios

Create scenario generation in terms of

\[
\Pi,\chi,B,C_Q,\theta
\]

instead of reporting only raw \(K_{\max}\), resource density, and allocation values.

## Stage E — Map phase space before optimizing

Use Sobol or Latin-hypercube sampling over the payoff and physical spaces to establish:

- collapse regions;
- managed scarcity;
- feasible coexistence;
- disconnected feasible regions;
- oscillatory regimes.

This phase-space characterization precedes optimization.

## Stage F — Constrained multiobjective optimization

Candidate method for the alternative multiobjective formulation:

> **Constrained noisy multiobjective Bayesian optimization**

Reason:

- only a few payoff parameters;
- each candidate needs multiple stochastic ABM replications;
- objectives and constraints are noisy;
- the output is a Pareto set.

One implementation option is BoTorch with noisy expected hypervolume improvement.

Sampling and evolutionary baselines provide comparisons across optimization methods.

## Stage G — Find the feasibility boundary

For fixed \((\chi,B,C_Q,\theta)\):

1. start at low \(\Pi\);
2. optimize the payoff family;
3. determine whether any candidate satisfies all constraints;
4. increase \(\Pi\);
5. repeat until feasibility disappears;
6. refine the transition to estimate \(\Pi_c\).

The outer continuation characterizes the feasibility boundary independently of the choice of inner search algorithm.

## Stage H — Out-of-sample validation

Validate headline findings with:

- unseen random seeds;
- longer horizons;
- unseen intermediate environmental values;
- different initial cooperation fractions;
- multiple grid sizes;
- alternative spatial demand layouts.

---

# 20. Alternative optimization pseudocode

```python
for physical_scenario in scenario_grid:

    feasible = []

    for demand_pressure in increasing_pi_values:

        def evaluate_game(payoff_params):
            results = [
                run_simulation(
                    scenario=physical_scenario,
                    demand_pressure=demand_pressure,
                    payoff_params=payoff_params,
                    seed=seed,
                )
                for seed in common_random_seeds
            ]

            return compute_objectives_and_constraints(results)

        pareto_set = constrained_multiobjective_optimize(
            evaluate_game,
            payoff_bounds=PAYOFF_BOUNDS,
        )

        feasible_candidates = [
            p for p in pareto_set
            if satisfies_constraints(p)
        ]

        if feasible_candidates:
            feasible.append((demand_pressure, feasible_candidates))
        else:
            break

    pi_critical = refine_feasibility_boundary(feasible)
```

---

# 21. Proposed result figures

**Figure 1 — From discovery to design.** Fixed ICCS game → spatial trap; parameterized games → ABM → physical objectives and constraints → feasible operating envelope.

**Figure 2 — Strategic payoff family.** \(\delta_C(n)\), \(\delta_D(n)\), and \(n^*(x)\), with the ICCS game marked as the reference point.

**Figure 3 — Same global mean, different local outcome.** Two simulations with similar global mean resource state but different PLD.

**Figure 4 — Pareto surfaces at increasing demand pressure.** Low, medium, near-critical, and supercritical \(\Pi\), showing contraction of the feasible set.

**Figure 5 — Critical feasibility boundary.** \(\Pi_c\) versus \(\chi\) for several spatial concentration levels \(C_Q\).

**Figure 6 — Complex-systems regime diagram.** Robust coexistence, sensitive coexistence, managed scarcity, oscillatory adaptation, infeasible.

**Figure 7 — Robustness.** Grid size, seeds, initial conditions, selected \(\theta\) values.

---

# 22. Levels of analysis

Parameter estimates:

> “The optimizer found \(\alpha=1.24,\beta=0.61,\gamma=0.83\).”

Feasibility hypothesis:

> “Across environmental regimes, the feasible payoff region contracts predictably with normalized demand pressure and disappears at a critical boundary determined by AI-to-human demand and spatial concentration.”

Reduced-dimensional hypothesis:

> “The feasibility transition approximately collapses onto a low-dimensional relation between relative demand pressure and spatial concentration.”

Conditional interpretation, requiring evidence of the stated boundary:

> **Below the feasibility boundary, strategic governance can prevent managed scarcity without eliminating useful AI activity. Above it, no payoff structure in the tested family prevents persistent local depletion; physical demand reduction, additional renewable supply, greater storage, altered siting, or technological change becomes necessary.**

---

# 23. Extensions beyond the ICCS formulation

The journal paper adds:

1. a parameterized strategic family rather than one fixed payoff matrix;
2. a physical multiobjective welfare formulation independent of game-payoff values;
3. Persistent Local Depletion as a local-tail metric that prevents spatial failure being hidden in global averages;
4. hard local constraints rather than maximizing water or cooperation;
5. dimensionless physical ratios rather than weak absolute calibration;
6. constrained Pareto analysis;
7. a feasibility boundary separating governance-correctable from physically infeasible regimes;
8. a complex-systems phase diagram;
9. longer-horizon and out-of-sample robustness;
10. direct continuity with the published model.

---

# 24. Interpretation for AI infrastructure and urban development

The framework represents distributed resource-access zones rather than one literal reservoir shared by all agents.

The abstraction is a spatially distributed system of effective local resource-access zones.

The model separates two situations.

## Governance-correctable

The physical system has sufficient underlying capacity, but the strategic response permits persistent local depletion.

In reality, this is the regime where operating rules, demand adaptation, pricing, allocation, siting, or other institutions could in principle matter.

## Physically infeasible

Even the best admissible strategic response cannot satisfy the local resource constraints at the proposed normalized demand pressure.

The implication is not that a specific policy has failed. It is that governance cannot manufacture physical capacity. Some combination of lower demand, greater efficiency, alternative sources, additional storage or renewal, altered siting, or different technology becomes necessary.

Mechanism-specific policy prescriptions are outside the scope of an abstraction without those mechanisms.

---

# 25. Limitations

- stylized and dimensionless rather than geographically calibrated;
- AI operational service is a normalized abstraction, not measured compute output;
- local resource cells represent effective supply zones rather than literal reservoirs;
- tariffs, permits, water rights, and regulatory institutions are not explicitly modeled unless added later;
- noise, animal impacts, electricity-grid effects, and embodied water are outside the present model;
- the feasibility boundary applies to the modeled game/environment family, not universally to real water systems;
- \(n_{\min}\), \(n_{\mathrm{safe}}\), and the PLD tail fraction require sensitivity analysis;
- payoff redundancy and identifiability depend on the chosen update rule.

---

# 26. Computational evaluation sequence

1. Freeze the published ICCS model and add regression tests.
2. Refactor game logic into a standalone payoff module.
3. Implement the normalized payoff parameterization.
4. Make local strategic composition enter payoff evaluation.
5. Record requested versus delivered human and AI demand.
6. Record complete per-cell water trajectories.
7. Implement PLD and hard constraints.
8. Implement \(\Pi,\chi,B,C_Q\).
9. Run a coarse Sobol design before any optimizer.
10. Inspect the regime structure manually.
11. Add constrained multiobjective optimization.
12. Estimate \(\Pi_c\) through an outer feasibility search.
13. Validate headline results out of sample.

The **phase structure and feasibility transition** are the scientific objects; optimization is the computational method used to investigate them.

---

# 27. Candidate conclusion if supported by the simulations

> The central limitation of governance in a spatial commons is not whether cooperation can be made more attractive in principle, but whether sufficient physical slack remains for strategic adaptation to matter. By parameterizing the coevolutionary game and optimizing it under local resource constraints, we identify a transition between governance-correctable and physically infeasible regimes. Below this transition, alternative payoff structures can eliminate managed-scarcity traps while maintaining both human and AI service. As normalized demand pressure and spatial concentration increase, the feasible strategic region contracts and ultimately disappears. Beyond this point, no strategic redesign in the modeled family can prevent persistent local depletion. The practical implication is conditional: strategic governance can enlarge the safe operating envelope of shared resource systems, but it cannot substitute indefinitely for physical capacity, efficiency, or appropriate infrastructure siting.

This interpretation is conditional on simulation evidence; it is not a reported finding.

---

# 28. Core contribution in one paragraph

The journal extension transforms the ICCS paper from a **discovery paper** into a **design-and-limits paper**. The previous work established that spatially localized feedback can generate managed-scarcity traps even when aggregate cooperation remains high. The extended work determines whether those traps can be removed by redesigning the strategic payoff structure, how the best attainable human/AI outcomes change with dimensionless resource pressure, and where the feasible strategic region disappears entirely. The final contribution is therefore not a particular optimized payoff vector. It is a computational characterization of the boundary between **resource systems that can still be stabilized through strategic governance** and **resource systems whose demand has exceeded what strategic adaptation can physically repair**.
