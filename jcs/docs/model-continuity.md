# Model continuity with ICCS 2026

The preceding publication is Ivana Malčić, Debraj Roy, and Luka Waronig, *Emergent Spatial Traps in the Coevolutionary Commons: Human-AI Cooperation Under Environmental Feedback*, Computational Science – ICCS 2026 Workshops, pages 95–109. [DOI](https://doi.org/10.1007/978-3-032-29909-3_8). [Official camera-ready paper](https://www.iccs-meeting.org/archive/iccs2026/papers/167870091.pdf). [Supplementary material](https://doi.org/10.5281/zenodo.18713008).

The JCS study inherits the spatial resource setting, two extraction-intensity classes, binary choices, environmental payoff feedback, and local resource renewal. It introduces a parameterized endpoint-game family, expected payoffs against local strategic composition, and constrained Pareto optimization of human support, AI support, and local depletion. Persistent local stress restricts acceptable designs; cooperation and the remaining spatial measurements describe their dynamics.

The original payoff interpolation is

\[
A(n)=(1-n)\begin{bmatrix}5&1\\3&0\end{bmatrix}
+n\begin{bmatrix}3&0\\5&1\end{bmatrix}.
\]

The normalized journal family is A(n)=(1-n)[[t+u,v],[1,s]]+n[[1,s],[t,0]], with t=2 and s=-0.5. At u=v=0, the original matrix equals 2*A(n)+1, where one is added to every entry. This common positive scaling and offset preserve the retained action comparison. The conference implementation compares row sums; the journal engine evaluates expected payoffs against frozen local composition. At the original point both rules prefer cooperation below n=1/2 and defection above it, with cooperation at ties. Exact matrix recovery does not make the full journal experiment identical to the conference experiment.

Only the depleted endpoint is redesigned. Its cooperation conditions are u>-(t-1) and v>s. Derived transition levels and the relative slopes follow from u and v; they are not independently chosen threshold parameters. Under u+v=0 the original row-sum rule is unchanged at every n. This control isolates the response to local strategic composition.

Higher extraction is treated as potentially supporting additional activity, rather than assumed to be waste. Welfare is evaluated relative to fixed higher-request targets, not relative to strategy-dependent requests. The main journal configuration shares a source's available stock in proportion to simultaneous requests. The inherited human-first withdrawal order is retained as an explicit comparison because it can affect welfare even when the final resource stock is unchanged.

With local cooperation x and normalized stock n, the `repository` renewal multiplier is

\[
m=\operatorname{clip}(1+0.5n(1-n)(\theta x-1),1,3).
\]

The `manuscript` variant implements Algorithm 2 of the camera-ready paper: theta x minus (1 minus x), with a lower clipping limit of 0.1. The JCS reference configuration uses `repository`; `manuscript` provides a feedback-law robustness check. The `pre_floor_change` variant uses the repository expression with a 0.1 floor.

The journal experiments use the JCS implementation, numerical configuration, dependency lock, and generated output bundles.

The conference description of a spatial trap concerns persistent resource–strategy organization. The journal code separately measures resource persistence, request fulfillment, and welfare. Neither a fixed stock cutoff nor delivery failure is silently substituted for the historical phenomenon. The stock-based and delivery-based constraints are explicitly different experiments.
