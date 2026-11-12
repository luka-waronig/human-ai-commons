from dataclasses import dataclass
import numpy as np
from .provenance import identifier


@dataclass(frozen=True)
class Policy:
    u: float = 0.
    v: float = 0.
    t: float = 2.
    s: float = -.5
    observation: str = 'local'
    forced: str | None = None

    def __post_init__(self):
        if not all(np.isfinite(value) for value in [self.u, self.v, self.t, self.s]):
            raise ValueError('finite payoff parameters required')
        if self.t <= 1 or self.s >= 0 or self.u <= -(self.t-1) or self.v <= self.s:
            raise ValueError('t>1, s<0, u>-(t-1), and v>s required')
        if self.observation not in ['local', 'global', 'fixed'] or self.forced not in [None, 'C', 'D']:
            raise ValueError('unknown observation or forced-action rule')

    @classmethod
    def from_relative(cls, r_c, r_d, t=2., s=-.5, observation='local', forced=None):
        return cls(float((t-1)*r_c), float(-s*r_d), t, s, observation, forced)

    @property
    def relative(self):
        return np.array([self.u/(self.t-1), self.v/(-self.s)])

    @property
    def id(self):
        return identifier(self)[:24]

    @property
    def q_c(self):
        return (self.t-1+self.u)/(2*(self.t-1)+self.u)

    @property
    def q_d(self):
        return (-self.s+self.v)/(-2*self.s+self.v)

    @property
    def omega(self):
        return (-2*self.s+self.v)/(2*(self.t-1)+self.u)

    def gap(self, n, x):
        a, b = self.t-1, -self.s
        baseline = x*a+(1-x)*b
        change = x*self.u+(1-x)*self.v
        return (2*n-1)*baseline-(1-n)*change

    def matrix(self, n):
        return (1-n)*np.array([[self.t+self.u, self.v], [1., self.s]])+n*np.array([[1., self.s], [self.t, 0.]])

    def cost(self, norm='L2'):
        r = self.relative
        if norm == 'L2':
            return float(np.linalg.norm(r)/np.sqrt(2))
        if norm == 'L1':
            return float(np.abs(r).mean())
        raise ValueError('unknown budget norm')

    @property
    def regime(self):
        first, second = self.relative
        return 'resource_only' if first == second else ('coordination' if first > second else 'anti_coordination')


def resource_only_encoding(k, t=2., s=-.5, bounds=(-.95, 2.), norm='L2'):
    """Least-cost encoding of the fixed-composition rule, including active bounds.

    k=(u+v)/(t-1-s). This rule's environmental transition is (1+k)/(2+k).
    """
    low, high = bounds
    if not low <= k <= high:
        raise ValueError('effective relative change is outside the box')
    a, b = t-1, -s
    Policy(t=t, s=s)
    z = (a+b)*k
    lower, upper = max(low, (z-b*high)/a), min(high, (z-b*low)/a)
    first = float(np.clip(a*z/(a*a+b*b), lower, upper))
    if norm == 'L1':
        options = [lower, upper, first, float(np.clip(0, lower, upper)), float(np.clip(z/a, lower, upper))]
        first = min(options, key=lambda r: (abs(r)+abs((z-a*r)/b), r*r+((z-a*r)/b)**2, r))
    elif norm != 'L2':
        raise ValueError('unknown budget norm')
    return Policy.from_relative(first, (z-a*first)/b, t, s, 'fixed')


def effective_change_interval(budget, t=2., s=-.5, norm='L2', bounds=(-.95, 2.)):
    endpoints = []
    for edge in bounds:
        good, bad = 0., edge
        if resource_only_encoding(edge, t, s, bounds, norm).cost(norm) <= budget:
            good = edge
        else:
            for _ in range(60):
                mid = (good+bad)/2
                if resource_only_encoding(mid, t, s, bounds, norm).cost(norm) <= budget:
                    good = mid
                else:
                    bad = mid
        endpoints.append(good)
    return tuple(endpoints)
