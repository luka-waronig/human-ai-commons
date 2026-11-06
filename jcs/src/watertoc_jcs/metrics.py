from dataclasses import dataclass
import numpy as np
from scipy.stats import beta


@dataclass(frozen=True)
class WelfareSpec:
    human_weight: float = .5
    utility: str = 'linear'
    additional_use_value: float = 1

    def __post_init__(self):
        if not np.isfinite(self.human_weight) or not 0 <= self.human_weight <= 1:
            raise ValueError('human welfare weight must be in [0,1]')
        if self.utility not in ['linear', 'log1p']:
            raise ValueError('unknown delivery utility')
        if not np.isfinite(self.additional_use_value) or not 0 <= self.additional_use_value <= 1:
            raise ValueError('additional-use value must be in [0,1]')


def supported_activity(delivered, baseline, target, additional_use_value=1):
    """Value baseline delivery and a declared share of additional delivery."""
    WelfareSpec(additional_use_value=additional_use_value)
    delivered, baseline, target = (np.asarray(value, dtype=float) for value in [delivered, baseline, target])
    if (not np.all(np.isfinite(delivered) & (delivered >= 0))
            or not np.all(np.isfinite(baseline) & (baseline > 0))
            or not np.all(np.isfinite(target) & (target >= baseline))):
        raise ValueError('finite deliveries and positive ordered activity references required')
    if additional_use_value == 1:
        return np.minimum(delivered/target, 1)
    if additional_use_value == 0:
        return np.minimum(delivered/baseline, 1)
    extra = np.maximum(np.minimum(delivered, target)-baseline, 0)
    value = np.minimum(delivered, baseline)+additional_use_value*extra
    return value/(baseline+additional_use_value*(target-baseline))


def joint_welfare(human, ai, human_weight=.5):
    """Combine normalized support for both kinds of activity.

    The weight expresses an assumed relative priority, not an estimated social value.
    At equal priority this is the geometric mean: sqrt(human * ai).
    """
    WelfareSpec(human_weight)
    human, ai = np.asarray(human, dtype=float), np.asarray(ai, dtype=float)
    if human.shape != ai.shape or not all(np.all(np.isfinite(v) & (v >= 0) & (v <= 1)) for v in [human, ai]):
        raise ValueError('matching normalized human and AI outcomes required')
    if human_weight == 0:
        return ai
    if human_weight == 1:
        return human
    return human**human_weight * ai**(1-human_weight)


class Service:
    def __init__(self, targets, is_ai, spec, baseline=None):
        self.targets = np.array(targets, dtype=float, copy=True)
        self.is_ai = np.asarray(is_ai)
        if (self.targets.ndim != 1 or not np.all(np.isfinite(self.targets) & (self.targets > 0))
                or self.is_ai.dtype != np.bool_ or self.is_ai.shape != self.targets.shape
                or not self.is_ai.any() or self.is_ai.all()):
            raise ValueError('positive fixed targets and both agent types required')
        if baseline is None and spec.additional_use_value != 1:
            raise ValueError('baseline requests required when varying the value of additional use')
        self.baseline = self.targets.copy() if baseline is None else np.array(baseline, dtype=float, copy=True)
        if (self.baseline.shape != self.targets.shape or not np.all(np.isfinite(self.baseline)
                & (self.baseline > 0) & (self.baseline <= self.targets))):
            raise ValueError('positive baseline no larger than each target required')
        self.spec, self.count = spec, 0
        self.totals = np.zeros_like(self.targets)

    def update(self, delivered):
        delivered = np.asarray(delivered, dtype=float)
        if delivered.shape != self.targets.shape or not np.all(np.isfinite(delivered) & (delivered >= 0)):
            raise ValueError('one finite nonnegative delivery per agent required')
        supported = supported_activity(delivered, self.baseline, self.targets, self.spec.additional_use_value)
        if self.spec.utility == 'log1p':
            supported = np.log1p(supported)/np.log(2)
        self.totals += supported
        self.count += 1

    def finish(self):
        if self.count == 0:
            raise ValueError('empty service observation window')
        means = self.totals/self.count
        human, ai = float(means[~self.is_ai].mean()), float(means[self.is_ai].mean())
        return dict(human_service=human, ai_service=ai,
                    welfare=float(joint_welfare(human, ai, self.spec.human_weight)),
                    agent_mean_service=means.tolist(), agent_service_target=self.targets.tolist())


@dataclass(frozen=True)
class StressSpec:
    reference: float = .5
    duration_steps: int = 150
    allowed_site_fraction: float = 0
    maximum_probability: float = .1
    basis: str = 'resource_stock'

    def __post_init__(self):
        if (not 0 < self.reference <= 1
                or isinstance(self.duration_steps, bool) or not isinstance(self.duration_steps, int) or self.duration_steps < 1
                or not 0 <= self.allowed_site_fraction < 1 or not 0 <= self.maximum_probability < 1):
            raise ValueError('invalid persistent local stress specification')
        if self.basis not in ['resource_stock', 'request_fulfillment']:
            raise ValueError('unknown persistence basis')


class LocalStress:
    def __init__(self, cells, horizon, spec):
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
            raise ValueError('positive integer observation horizon required')
        self.spec, self.horizon = spec, horizon
        if spec.duration_steps > horizon:
            raise ValueError('stress duration cannot exceed the observation horizon')
        self.tracker = Depletion(cells, MetricSpec(spec.reference, 1))
        self.duration_steps = spec.duration_steps

    def update(self, normalized):
        if self.tracker.count >= self.horizon:
            raise ValueError('stress observation exceeds declared horizon')
        self.tracker.update(normalized)

    def finish(self):
        if self.tracker.count != self.horizon:
            raise ValueError('incomplete stress observation horizon')
        durations = self.tracker.longest_continuous_depletion_steps
        affected = durations >= self.duration_steps
        fraction = float(affected.mean())
        result = dict(persistent_local_stress=bool(fraction > self.spec.allowed_site_fraction),
                      persistent_site_fraction=fraction, stress_duration_steps=self.duration_steps,
                      stress_basis=self.spec.basis, site_longest_continuous_stress_steps=durations.tolist())
        if self.spec.basis == 'resource_stock':
            result.update(persistent_cell_fraction=fraction, cell_longest_continuous_stress_steps=durations.tolist())
        return result


def request_fulfillment(delivered, requested):
    delivered, requested = np.asarray(delivered), np.asarray(requested)
    if (delivered.shape != requested.shape or not np.all(np.isfinite(requested) & (requested > 0))
            or not np.all(np.isfinite(delivered) & (delivered >= 0))):
        raise ValueError('matching deliveries and positive requests required')
    ratio = np.minimum(1, delivered/requested)
    return np.where(np.isclose(delivered, requested, rtol=1e-12, atol=0), 1., ratio)


def probability_upper_bound(events, confidence=.95):
    """One-sided exact binomial upper bound over independent realizations."""
    events = np.asarray(events)
    if events.ndim != 1 or not len(events) or events.dtype != np.bool_ or not 0 < confidence < 1:
        raise ValueError('nonempty boolean events and confidence in (0,1) required')
    failures = int(events.sum())
    return 1. if failures == len(events) else float(beta.ppf(confidence, failures+1, len(events)-failures))


def upper_tail_mean(values, fraction):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)) or not 0 < fraction <= 1:
        raise ValueError('finite nonempty vector and tail fraction in (0,1] required')
    count = fraction*len(values)
    whole = int(np.floor(count))
    ordered = np.sort(values)[::-1]
    total = ordered[:whole].sum()
    if whole < len(ordered):
        total += (count-whole)*ordered[whole]
    return float(total/count)


@dataclass(frozen=True)
class MetricSpec:
    floor: float = .2
    tail: float = .1

    def __post_init__(self):
        if not 0 < self.floor <= 1 or not 0 < self.tail <= 1:
            raise ValueError('floor and tail fraction must lie in (0,1]')


class Depletion:
    def __init__(self, cells, spec):
        if cells < 1:
            raise ValueError('the accessible resource set must not be empty')
        self.spec, self.count = spec, 0
        self.deficits = np.zeros(cells)
        self.duration = np.zeros(cells)
        self.continuous_depletion_steps = np.zeros(cells, dtype=int)
        self.longest_continuous_depletion_steps = np.zeros(cells, dtype=int)
        self.minimum_resource = np.inf

    def update(self, normalized):
        normalized = np.asarray(normalized)
        if normalized.shape != self.deficits.shape or not np.all(np.isfinite(normalized)):
            raise ValueError('invalid resource observation')
        self.count += 1
        depleted = normalized < self.spec.floor
        self.deficits += np.maximum(self.spec.floor-normalized, 0)/self.spec.floor
        self.duration += depleted
        self.continuous_depletion_steps = np.where(depleted, self.continuous_depletion_steps+1, 0)
        self.longest_continuous_depletion_steps = np.maximum(
            self.longest_continuous_depletion_steps, self.continuous_depletion_steps)
        self.minimum_resource = min(self.minimum_resource, float(normalized.min()))

    def finish(self):
        if self.count == 0:
            raise ValueError('empty observation window')
        deficits = self.deficits/self.count
        return dict(loss=upper_tail_mean(deficits, self.spec.tail),
                    mean_cumulative_deficit=float(deficits.mean()),
                    mean_depleted_fraction=float((self.duration/self.count).mean()),
                    longest_continuous_depletion_steps=int(self.longest_continuous_depletion_steps.max()),
                    cell_longest_continuous_depletion_steps=self.longest_continuous_depletion_steps.tolist(),
                    cell_depleted_fraction=(self.duration/self.count).tolist(),
                    minimum_accessible_resource=float(self.minimum_resource),
                    observation_steps=self.count, scored_cells=len(deficits))


def rescore(stocks, capacity, accessible, burn, spec):
    stocks = np.asarray(stocks)
    if stocks.ndim != 2 or not 0 <= burn < len(stocks):
        raise ValueError('invalid trajectory window')
    acc = Depletion(int(np.sum(accessible)), spec)
    for water in stocks[burn:]:
        acc.update(water[accessible]/capacity[accessible])
    return acc.finish()


def paired_interval(reductions, generator, confidence=.95, samples=10000):
    values = np.asarray(reductions, dtype=float)
    if len(values) < 2 or not np.all(np.isfinite(values)) or not 0 < confidence < 1:
        raise ValueError('at least two finite paired realizations required')
    means = values[generator.integers(0, len(values), (samples, len(values)))].mean(axis=1)
    alpha = (1-confidence)/2
    low, high = np.quantile(means, [alpha, 1-alpha])
    return dict(mean_reduction=float(values.mean()), lower=float(low), upper=float(high),
                confidence=confidence, paired_realizations=len(values), bootstrap_samples=samples)
