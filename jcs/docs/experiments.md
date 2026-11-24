# Experiments of the paper: the original game over time, payoff changes, and local measures

This protocol is separate from the staged payoff search described in the [research protocol](research-plan.md). It has its own configurations (those in `configs/` with a `paper` section), its own random-input splits, its own output folders and its own analysis. Each configuration extends the study configuration and adds a `paper` section; the world, the reference scenario, the payoff family and the base seed are read with the study's own loaders, so both protocols describe the same model. It never writes to a search bundle. The scenario options that only these experiments switch on stay out of the search's evaluation identifiers while they hold their defaults, so the search's reproduction path is unchanged.

The configuration, designs, windows and comparisons below were fixed before any simulation of this protocol was run. They were informed by reading saved histories of the original game from the staged search (48 held-out worlds per load); that reading is exploratory and is reproduced by `audit-search` on all 320 worlds. A 12-world pilot (`configs/pilot.json`) checked that the measurements respond to the contrasts and gave the variation needed to size the runs; it supports no confirmatory claim. The six experiments of the paper each declare their comparisons in advance, with one 5% allowance per experiment.

## Question

Can changing the payoff structure weaken the spatial feedback processes that sustain undesirable outcomes, and what activity trade-offs accompany that change? The protocol addresses three linked parts: which resource–behavior relationships of the conference study recur under the journal assumptions; how payoff changes alter those dynamics; and what that does to human and AI activity, when, and for whom.

## Quantities defined before simulation

The expected ordinary load of a source is defined from geometry and requests. An agent chooses uniformly among its accessible sources that hold stock. With all of them available, its ordinary request is spread equally over its access set. Summing these shares over the agents that can reach a source and dividing by the source's baseline renewal gives its expected ordinary load. The load depends on geometry and requests. It does not depend on payoffs, because source selection does not.

Sources are grouped into three classes. With intensified requests equal to 1.5 ordinary requests and a largest renewal multiplier $m_{\max}$ (1.25 for the reference law at $\theta=3$):

- underused: load $\times 1.5 \le 1$. Expected requests stay within baseline renewal even if every user intensifies.
- overloaded: load $> m_{\max}$. Expected ordinary requests exceed the largest renewal the law can produce.
- contested: in between. Choices can decide whether the stock is held.

Action reversals and the randomness of source selection blur both boundaries. The classes organize the analysis; they are not thresholds of the model.

Stock is a fraction of capacity. The share of demand met (fields `human_activity` and `ai_activity`) is delivered resource divided by the fixed intensified-use target, capped at one, averaged over agents of a type; fully delivered ordinary use gives two-thirds. Deliveries are also reported in resource units, because conditions with different requests have different targets. Renewal is reported as actual renewal, the part added by the cooperation term, and the part lost because a source was full.

## Hypotheses stated in advance

1. Structure: The long-run stock of a source follows its expected ordinary load. Overloaded sources stay near empty under every admissible payoff design.
2. Capacity: With requests and renewal fixed, capacity changes how long the initial stored volume lasts, and has little effect on long-run stock fraction, cooperation or deliveries. If so, a relationship between capacity and cooperation observed at 100 steps from full stocks is produced by the initial stored volume: it should appear when sources start full and not when they start with equal volume.
3. Start dependence: If low stocks are self-maintained through behavior-dependent renewal, late source states depend on where stocks started, and that dependence disappears when renewal no longer reads current behavior. If late states do not depend on the start, persistence reflects load, not a self-maintaining process.
4. Reach of payoff changes: Shifting the switching level moves contested sources toward the new level and changes activity mainly while stored volume is released or rebuilt. Sustained differences are small and limited to the renewal added by cooperation. Overloaded sources are unaffected.
5. Participation dependence: Unequal relative changes differ from equal ones only where local composition is mixed.
6. Recovery: After a common temporary disturbance, the return of stocks and activity is compared between changed and unchanged payoffs. No direction is assumed.

A sustained difference of 0.01 in a group's activity (one percentage point of the fixed target) is declared as the smallest difference of interest. Smaller differences are reported with their intervals and described as small.

## Experiment A: baseline continuity and physical drivers (original game only)

A ladder moves from the conference parameterization to the journal reference one change at a time, on the same landscapes and random inputs:

| Condition | Change from the previous row | Held fixed |
| --- | --- | --- |
| `conference_like` | fixed requests (human 0.1, AI 2.0), unconditioned placement, human-first allocation, reference renewal law | landscape, capacity 20, full initial stocks |
| `conference_like_published_law` | renewal law of the published Algorithm 2 | everything else |
| `conditioned_placement` | every agent placed with access (reference renewal law) | requests, allocation |
| `proportional_allocation` | shortages shared in proportion to requests | requests, placement |
| `journal_reference` | ordinary requests rescaled so that their total equals accessible baseline renewal | placement, allocation |

The last step changes request magnitudes by the factor recorded as `relative_load`. It is a demand change, not a change of notation: writing the same numerical requests in either form gives identical histories, which is a unit test.

Storage is varied at capacities 10, 20 and 40 on the same landscape, renewal and requests, under both request forms. Each capacity starts full (initial volume grows with capacity), with 10 units per source (equal initial volume), and empty. Supply is varied through source density 0.2 and 0.5, which draws a new landscape; with fixed requests the relative load changes, and with scaled requests the requests change. Every row records capacity, initial stored volume, total ordinary request, accessible baseline renewal and their ratio.

Windows: steps 80–100 (the last fifth of the conference horizon), 0–100, 400–500 and 1500–2000.

## Experiment B: payoff changes after development and from initialization

The journal reference world runs 1000 steps under the original game. Its complete state (stocks, strategies, absolute step) is then continued for 1500 steps under unchanged payoffs and under each design, with the same indexed random inputs at the same absolute steps. Each design is also applied from initialization. An unchanged continuation is compared with the uninterrupted run in every world and must be identical.

Designs were chosen by the decision rule they produce, not by any outcome:

| Design | $r_C$, $r_D$ | Switching level against cooperators, defectors | Role |
| --- | --- | --- | --- |
| `threshold_up`, `threshold_up_small` | 1, 1; 0.5, 0.5 | 0.667; 0.6 | ordinary use starts at higher stock; composition irrelevant |
| `threshold_down`, `threshold_down_large` | −0.5, −0.5; −0.9, −0.9 | 0.333; 0.091 | ordinary use starts at lower stock |
| `join_cooperators` | 1, 0 | 0.667, 0.5 | ordinary use is favored more where others use ordinarily |
| `offset_defectors` | 0, 1 | 0.5, 0.667 | ordinary use is favored more where others intensify |
| `offsetting_toward_cooperators` | 0.45, −0.9 | 0.592, 0.091 | $u+v=0$ |
| `offsetting_toward_defectors` | −0.45, 0.9 | 0.355, 0.655 | $u+v=0$ |
| `searched_equal_0345`, `searched_lower_bound` | 0.345, 0.345; −0.95, −0.95 | 0.574; 0.048 | designs retained by the staged search, as comparators |
| `always_ordinary`, `always_intensified` | — | — | the range any choice rule can span |

Windows after the change: steps 0–50, 50–150, 150–500 and 500–1500. Designs applied from initialization are compared with unchanged payoffs both at equal time since the change and at the same absolute steps.

## Experiment C: what behavior-dependent renewal contributes

Two controls change how renewal reads behavior. `fixed_cooperation` keeps the stock dependence of the renewal multiplier but replaces current neighborhood cooperation by a time-invariant value per source, taken from the same world's coupled run. It removes the dependence of renewal on current behavior. Requests, access, source selection, payoffs, choices and the stock dependence of renewal remain free. `baseline_only` sets the multiplier to one; it also removes the stock dependence and the renewal added by cooperation. Neither control changes what agents observe.

Start dependence is tested first. The reference world runs 3000 steps from full and from empty stocks with the same strategies and random inputs, under coupled renewal, both controls, and four designs. Per source, late stocks (steps 2000–3000) of the two starts are compared; the same comparison at steps 1000–2000 and the drift within one run give the scale of slow convergence and noise.

Design effects are then compared under each renewal rule. From the developed state, unchanged payoffs and two threshold designs continue under coupled renewal and both controls. A design effect that is the same under `fixed_cooperation` does not operate through behavior-dependent renewal.

Recovery is tested last. From the developed state, two common disturbances start at the change: renewal reduced to a quarter for 200 steps, and all stocks emptied once. Each design is compared with itself undisturbed, and unchanged payoffs supply the reference recovery. Windows: 0–200, 200–400, 400–800 and 800–1500 steps.

## Follow-up chosen after the reference pilot

`configs/pilot_strong_feedback.json` repeats the pilot on its first eight worlds with renewal responsiveness $\theta=10$, the largest value of the conference sweep, where $m_{\max}=2.125$. It was added after the reference pilot showed no start dependence at $\theta=3$, to learn whether that result is specific to weak renewal feedback. Its designs, windows and measurements are unchanged. It is exploratory in the same sense as the reference pilot, and its choice depended on the reference pilot's outcome.

## Diagnostics that fixed the final design

`configs/diagnostics.json` and `configs/diagnostics_long.json` were run on the eight pilot worlds after the two pilots, each for its feedback part (`--experiment feedback`). They are exploratory and chose what the experiments of the paper contain:

- three renewal responsiveness values ($\theta$ = 3, 5, 10) crossed with four loads (ordinary demand at 0.75, 1 and 1.5 times accessible baseline renewal, and the conference's fixed requests), to locate where late source states depend on the starting stocks;
- a 12,000-step version of the start-dependence test at $\theta=10$, to see whether the dependence is still shrinking;
- three designs outside the inherited family, described below.

## Designs outside the inherited payoff family

The inherited family moves the stock level at which an agent switches to ordinary use; it reads the stock of the source the agent selected. Three additions test whether that is the limiting assumption. Each is compared with the inherited designs under identical conditions, and none is reported as an admissible design of the inherited family.

- Payoff reads the stock the agent's behavior affects (`consequence_half`, `consequence_full`): renewal at a source reads the behavior of every agent within the feedback radius, while the inherited payoff reads only the selected source. With weight $w$ the payoff's environmental state becomes $(1-w)$ times the selected source's stock plus $w$ times the mean stock of all sources whose renewal the agent's behavior enters. This implements the conference paper's proposal to tie incentives to consequences at the spatial scale of the feedback. Weights 0.5 and 1 were fixed in advance.
- Reliability of the response (`no_reversals`, `reversals_5_percent`, `reversals_20_percent`): the payoffs are unchanged and the probability of acting against them is 0, 0.05 or 0.2 instead of 0.1. In this model the size of a payoff difference does not affect choices, only its sign does; these designs bound what a stronger incentive of the same shape could achieve if it made choices more reliable.
- Ordinary use almost everywhere, with reversals kept (`very_high_threshold`, switching level 0.95, outside the admissible bounds): together with `no_reversals` it separates the two things the forced rule `always_ordinary` changes at once.

## Sustained-delivery ceiling

For each world and request vector the report gives the largest total delivery per step that any choice of sources and actions could sustain: a transport problem on the access network in which each agent receives at most its fixed target, each source gives at most its renewal, and stocks do not change. It is computed with baseline renewal and with the largest renewal multiplier. No payoff design, source-selection rule or allocation rule can sustain more with the same access, requests and renewal.

## The experiments on 64 worlds

`configs/original_game.json` uses 64 worlds of a new split (`paper-original-game`) that no earlier run touched. Its settings, designs, windows and comparisons were fixed before any of those worlds was simulated; the same file was checked end to end on two worlds of a separate throwaway split. Every experiment of this section had been run once before with earlier code, on other splits of worlds and with the same settings and declared comparisons; the runs reported here repeat them on worlds that no earlier run simulated.

| Part | Question | Settings | Conditions |
| --- | --- | --- | --- |
| A | Which conference relationships recur? | reference | ladder, storage and supply conditions as above |
| C, start dependence | Is a low state self-maintaining, and where? | all 12 ($\theta$ 3, 5, 10 × four loads) | coupled renewal and both renewal controls; three designs in four settings |
| C, controls and recovery | What does behavior-dependent renewal contribute, and do stocks return after a disturbance? | reference; $\theta=10$ at loads 0.75 and 1 | as above, with `consequence_full` added to recovery |
| B | What do payoff changes do, when, and to whom? | reference (also from initialization); loads 0.75 and 1.5; $\theta$ 5 and 10; conference requests at $\theta$ 3 and 10 | 8 inherited designs, 2 designs kept by the staged search, 2 forced rules, 3 reliability designs, 1 diagnostic, 2 consequence designs |

The file declares 33 paired contrasts. They share one 5% error allowance: each interval is a paired $t$ interval at confidence $1-0.05/33$, and a distribution-free bounded-mean interval with the same simultaneous coverage is reported beside it. A contrast is read as a difference when its interval excludes zero, and as smaller than the smallest difference of interest when its interval lies inside ±0.01. All other tables of the final run are descriptive.

`configs/original_game_long_horizon.json` is a companion on the first 32 worlds of the same split. It repeats the start-dependence test over 12,000 steps at $\theta=10$ and loads 0.75 and 1, with two declared contrasts of its own, to decide whether a dependence seen after 3000 steps is still present much later. It was declared together with the final run.

`configs/model_assumptions.json` is a second companion on the same 64 worlds. It repeats seven key designs at the reference load and $\theta=3$ under seven changed assumptions, one at a time: human-first allocation, the published renewal law, sequential planning, unconditioned placement, an AI-to-human demand ratio of 5 and of 40, and a 40 × 40 grid with 400 agents. It declares 42 contrasts of its own. Its configuration was written while the final run was in progress and before any result of that split was read.

`configs/agent_assumptions.json` is a third companion on the same 64 worlds. It tests assumptions under which the main findings could hold by construction. Agents choose a source in proportion to its stock (`stock_weighted_choice`) or always take the fullest source they can reach (`fullest_choice`), at $\theta=3$ and $\theta=10$, instead of choosing uniformly. The higher request is 2 or 3 times the lower one instead of 1.5 (`higher_request_2x`, `higher_request_3x`). Agents follow a graded response in which the probability of the higher request is $1/(1+e^{-G/\tau})$ for payoff difference $G$ and $\tau$ of 0.5, 0.25 or 0.1, so that larger differences are followed more reliably (`graded_response_050`, `graded_response_025`, `graded_response_010`); the reversal probability is then unused. Sequential planning is repeated with two designs that depend on what others do. One design is added in every cell: `consequence_reach`, in which the payoff reads the mean stock of the sources the agent can reach instead of all sources its behavior affects. Ten designs run in twelve cells, and start dependence is measured in eight. The file declares 28 contrasts sharing one 5% allowance. It is 24.6 million model steps. It was designed after the main experiments, to test them, and is treated as exploratory. A first complete run of eleven cells showed that the fullest-source rule broke ties by source index; the rule now breaks ties at random, two cells were added (stock-weighted choice under the fixed requests, and at $\theta=10$ with load 0.75), and the whole run was repeated. Only the repeated run is reported.

## Experiment D: a map of the payoff family and a test of its selection

The staged search scored designs only after the first 500 steps, used a depletion score that did not respond to any design, and required that no accessible source stay below half capacity for 150 steps. The `audit-search` command shows that this requirement cannot be met: overloaded sources exist in every world at loads of 0.75 and above, and source selection does not depend on payoffs. Experiment D repeats the search with scores that respond and a requirement that can be met. `configs/payoff_grid.json` fixes all of the following before any of its worlds is simulated.

The designs are the 324 points of the staged study's grid: 17 equally spaced relative changes per axis between the admissible bounds, with zero added, for the payoff of ordinary use against ordinary users and against intensifying users. The two forced rules are run beside them as outer bounds and are never selected. Each design continues the same developed state of the original game for 1500 steps with matched random inputs, in the reference setting and at $\theta=10$, on 32 worlds of the split `paper-payoff-grid`.

Three objectives are maximized: human activity and AI activity over all 1500 steps after the change, and the stock of contested sources over steps 500 to 1500. Scoring the whole period keeps the early release or rebuilding of stored volume in the comparison. The requirement is that a design leaves no larger share of contested sources nearly empty over steps 500 to 1500 than the original game does, where nearly empty means a mean stock below one tenth of capacity. The original game meets this requirement by definition.

Within the largest budget of the staged study, a design is retained when it meets the requirement and no other such design is at least as good on all three objectives and better on one. At most eight retained designs per setting are selected: the best on each objective, then those that spread widest in the scaled objectives. The selection is written to `selected_designs.json` and is not changed by later results. The selected designs and the original game are then run on 64 worlds of a second split, `paper-payoff-grid-test`, that the map never used. Their paired differences from the original game share one 5% error allowance across all selected designs, settings, objectives and the requirement.

The map itself involves no selection: every design is reported with its paired difference from the original game for each outcome and window. Statements about selected designs rest on the test worlds only.

## Local measures applied to AI agents

The conference study closes by naming three measures that tie use to local conditions: spatially differentiated pricing, locally adaptive caps, and siting rules that limit expansion into residential areas. A change of $u$ and $v$ is a uniform rule that acts on how much each agent requests. The three measures act on something else: where demand goes at each step, how low a source may fall, and where demand sits. `configs/local_measures.json` tests them with the original game's payoffs unchanged. The paper calls these measures interventions.

The measures bind AI agents only (`regulated: ai`), because AI agents are the large users: with a demand ratio of 20 they account for 95% of requests. Human agents keep choosing a source at random and are never restricted by a cap.

| Measure | Definition | Strengths |
| --- | --- | --- |
| Price on scarcity | an AI agent chooses among the sources it can reach with probability proportional to $e^{\kappa n}$, where $n$ is the stock level after renewal | $\kappa$ of 1 and 4, and the limit in which the fullest source is always taken |
| Local cap | the requests of AI agents at a source are cut back, in proportion, to the stock above a floor; the source is then shared as without a cap | floors of 10%, 25% and 50% of capacity |
| Siting | AI agents are placed where the sources can carry them: a cell is acceptable when the highest expected load on the sources reachable from it stays at or below one, or is the lowest available; among acceptable cells the one sharing the fewest sources with human agents is taken. With a share below one, those whose sources carry the highest load are the ones moved | half and all AI agents |

Three properties of these definitions matter for reading the results. The cap limits what AI agents draw; human agents may still draw a source below the floor. An AI agent under a cap still chooses its source at random, so it can arrive at a capped source and receive nothing, which the price together with the cap addresses. The conference study speaks of siting that limits expansion into residential areas. Keeping AI agents entirely off the sources that human agents use is not simulated: human agents reach about two thirds of all sources and AI agents account for 95% of requests, so full separation would confine AI agents to the sources that no human agent reaches, which hold about a third of the renewal. The siting rule therefore puts carrying capacity first and separation second. The rule does not separate the two groups. Under random placement 87% of AI agents reach at least one source that human agents use, and under siting 88%. Sited AI agents spread over more sources, so the share of human-used sources that AI agents also reach rises from 70% to 88%, and the baseline renewal at sources that only human agents reach falls from 7.7 to 2.0 units per step. Sources, human agents and initial strategies stay as generated. The report gives the number of reachable sources, the reachable baseline renewal and the sustained-delivery ceiling for each placement beside the effect.

The model contains no money. The price is represented by its effect on the choice of source, and its burden is reported as the scarcity of the source each agent draws from (`human_scarcity`, `ai_scarcity` and the same by exposure group). Under siting, requests keep the size in resource units that they have under random placement, so that both placements face the same demand; the reachable renewal differs between placements, so the same demand is another load on it.

Three further conditions complete the set: the price at $\kappa=4$ together with the 25% cap; human agents served first when a source runs short (`small_users_first`), which is the allocation of the conference study, under which the AI agents at a short source are served in a fixed order, so that only group means are read for this condition; and the price at $\kappa=4$ applied to every agent (`price_4_everyone`), which measures what is added by extending a measure to human agents.

Each setting changes one thing in the baseline of the conference study, except that full siting is also run at $\theta=10$. Each has a stated reason.

| Setting | Reason |
| --- | --- |
| reference ($\theta=3$, density 0.3, capacity 20, load 1) | the baseline of the conference study with requests scaled to renewal |
| $\theta=10$ | the upper end of the conference sweep, where behavior-dependent renewal is largest |
| source density 0.2 and 0.5 | the ends of the conference sweep; density sets how many sources an agent can reach, which a price depends on |
| AI share of demand 50% and 20% (demand ratio 1 and 0.25) | whether the results require AI agents to dominate demand |
| fixed requests of the conference study | its own demand, at which most sources are overloaded |
| siting of half and of all AI agents; of all AI agents also at $\theta=10$ | the third measure, which is a property of the world and therefore a setting |

Capacity is not varied: Experiment A shows that it changes how long the initial stock lasts and not the state that is reached. The price, the cap and the allocation rule are switched on after 1000 steps of the original game and followed for 1500 steps with matched random inputs, as in Experiment B. Siting holds from the first step, and its effect is the difference between the original game under sited and under random placement of the same world.

The outcomes are the demand met for human and for AI agents, the share of reachable sources that are nearly empty, and the scarcity measure above. Demand met counts every agent at its higher request, so a lower request delivered in full counts as two thirds. To tell asking less from receiving less, each group's requests and its share of agents on the lower request are recorded, and the share of requests that is delivered is reported beside the demand met. With a floor at or above one tenth of capacity the share of nearly empty sources falls largely by construction, so the cap is judged on the demand met. Start dependence is measured at $\theta=10$ for the original game, the price at $\kappa=4$ and the 25% cap.

The file declares 19 contrasts with one 5% allowance: the price at $\kappa=4$, the 25% cap and the siting of all AI agents, each for both groups in the reference setting and at $\theta=10$; human agents served first; the price extended to every agent against the price on AI agents; and three start-dependence contrasts. The contrasts on the price, the cap and the allocation rule use all 1500 steps after the change, so that an early cost is counted. The four siting contrasts compare the same 1500 steps of the original game in two developed worlds; no change occurs within them. The run uses 64 worlds of the split `paper-local-measures`, which no earlier run simulated, and is 13.8 million model steps.

A pilot of six worlds on the separate split `mechanism-interventions-pilot` preceded this file. In it the price and the cap applied to every agent; it served to choose the strengths and to check the run end to end. Its numbers are not evidence, and its output is kept with the local results and is not part of the repository.

## Siting across settings

The local measures run applies the price and the cap in seven settings and siting in two of them, the reference setting and $\theta=10$. `configs/siting_across_settings.json` adds siting of all AI agents in the other five: source density 0.2 and 0.5, an AI share of demand of 50% and 20%, and the fixed requests of the conference study. It uses the same 64 worlds as the local measures run. Each of the five settings is simulated with random and with sited placement, under the original game, the price at $\kappa=4$ and the 25% cap; the rows with random placement repeat rows of the local measures run and agree with them. The file declares ten comparisons with one 5% allowance: demand met for human and for AI agents under sited against random placement, in each setting, over the same 1500 steps. It is 5.1 million model steps. It was designed after the main experiments, to close a gap in the settings of siting, and is treated as exploratory.

## Local measures under changed assumptions

The two companions on changed assumptions repeat payoff designs, not the measures. `configs/local_measures_assumptions.json` repeats the measures under the same assumptions, one at a time, at the reference load and $\theta=3$: the price at $\kappa=4$, the 25% cap and both together, each with random and with sited placement of all AI agents.

| Assumption | Changed to |
| --- | --- |
| allocation | human agents served first when a source runs short (`human_first_allocation`) |
| renewal | the renewal law of the conference study, under which renewal can fall below its baseline (`published_renewal_law`) |
| planning | agents decide one after another (`sequential_planning`) |
| placement | agents placed without regard to access (`unconditioned_placement`) |
| demand split | an AI-to-human demand ratio of 5 and of 40 (`ai_demand_ratio_5`, `ai_demand_ratio_40`) |
| grid | 40 × 40 cells with 200 agents of each kind (`larger_grid`) |
| higher request | 2 and 3 times the lower one (`higher_request_2x`, `higher_request_3x`) |
| response | graded, with $\tau$ of 0.5, 0.25 and 0.1 (`graded_response_050`, `graded_response_025`, `graded_response_010`) |

Each of these twelve cells has a sited twin, named `sited_` followed by its name, that differs only in where the AI agents are placed. Choice of source in proportion to stock, or of the fullest source, is not combined with the measures. The price itself sets how AI agents choose a source, and the price and the cap restrict AI agents only, so human agents would return to random choice and the cell would no longer hold its assumption for them.

The file declares 96 comparisons with one 5% allowance. In each of the twelve cells, the price, the cap and both together are compared with the original game, and sited with random placement under the original game, each for human and for AI agents over the 1500 steps after the change. It uses the same 64 worlds as the local measures run and is 14.6 million model steps. It was designed after the main experiments and is treated as exploratory.

## Replication and inference

The independent unit is the world with its trajectory inputs. All conditions of one world share them, so comparisons are paired within worlds, and several continuations of one baseline count as one unit. Pilot tables give means, standard errors and descriptive $t$ intervals over 12 worlds without correction for the number of comparisons. Each experiment of the paper uses its own split of worlds, a stated family of comparisons and its own error allowance. A world in which the outcome of a comparison is undefined, such as the stock of contested sources in a world that has none, is left out of that comparison; the table of comparisons gives the number of worlds each one used.

## Commands

Run from the repository root.

```sh
.venv/bin/python jcs/reproduce.py paper plan --config jcs/configs/pilot.json
.venv/bin/python jcs/reproduce.py paper run --config jcs/configs/pilot.json --output jcs/results/pilot --max-model-steps 2316000 --workers 1
.venv/bin/python jcs/reproduce.py paper report --output jcs/results/pilot
.venv/bin/python jcs/reproduce.py audit-search jcs/results/research_protocol --output jcs/results/search_audit
```

Every experiment of the paper:

```sh
.venv/bin/python jcs/reproduce.py paper all
```

This runs the six experiments in the order `local_measures`, `original_game`, `original_game_long_horizon`, `model_assumptions`, `payoff_grid`, `agent_assumptions`, into `jcs/results/<name>/`, each with its planned number of model steps, and writes the tables of each report. The companion on siting across settings is run by its own two commands: `.venv/bin/python jcs/reproduce.py paper run --config jcs/configs/siting_across_settings.json --output jcs/results/siting_across_settings --workers 16 --max-model-steps 5120000`, then `.venv/bin/python jcs/reproduce.py paper report --output jcs/results/siting_across_settings`. The companion on the measures under changed assumptions is run in the same way with `--config jcs/configs/local_measures_assumptions.json`, `--output jcs/results/local_measures_assumptions` and `--max-model-steps 14592000`. Figures are drawn afterwards by the scripts in `jcs/figures/`, and the numbers used in the paper's tables are printed by the scripts in `jcs/analysis/`. Names given after `all` restrict it to those experiments, for example `paper all payoff_grid`. It uses all but two cores, up to 16, unless `--workers` is given. Repeating the command after an interruption continues from the saved jobs. A results folder made with another configuration or other source code is refused, so the results of an earlier version are moved aside before a run is repeated, and the package is left unchanged while a run is in progress.

`plan` and `report` never simulate. The planned number of steps is exact for every experiment except the test of selected designs, which is planned at its largest possible size. `run` refuses to start above the stated model-step limit, reuses completed worlds, and refuses a changed configuration or source. Each world and setting is one job; `--workers` sets how many run at once, each with one numerical thread, and saved results do not depend on the worker count. `audit-search` reads the saved staged-search bundle and never simulates.
