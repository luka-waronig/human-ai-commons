"""Spatial patterns and deliveries assessed separately from failure conditions."""
import numpy as np
from .metrics import request_fulfillment
from .world import torus_distances


def spatial_autocorrelation(values, adjacency):
    values, weights = np.asarray(values, dtype=float), np.asarray(adjacency, dtype=float)
    if values.ndim != 1 or weights.shape != (len(values), len(values)) or not np.all(np.isfinite(values)):
        raise ValueError('finite values and matching spatial weights required')
    if np.any(weights < 0) or not np.all(np.isfinite(weights)) or np.any(np.diag(weights) != 0):
        raise ValueError('nonnegative finite spatial weights with no self-links required')
    if len(values) < 2 or np.ptp(values) == 0 or weights.sum() == 0:
        return None
    centered = values-values.mean()
    return float(len(values)/weights.sum()*(centered @ weights @ centered)/(centered @ centered))


def spatial_summary(world, stocks, strategies, requested, delivered):
    stocks, strategies, requested, delivered = (np.asarray(v) for v in [stocks, strategies, requested, delivered])
    steps, agents, cells = len(stocks), len(world.positions), len(world.resources)
    if (steps == 0 or stocks.shape != (steps, cells) or strategies.shape != (steps, agents)
            or requested.shape != (steps, agents) or delivered.shape != (steps, agents)):
        raise ValueError('matching nonempty recorded trajectories required')
    accessible = world.access.any(axis=0)
    normalized = stocks[:, accessible]/world.capacity[accessible]
    average_stock = normalized.mean(axis=0)
    neighbors = torus_distances(world.positions, world.positions, world.spec.width, world.spec.height) <= 1
    np.fill_diagonal(neighbors, False)
    ecological = torus_distances(world.resources[accessible], world.positions, world.spec.width, world.spec.height) <= world.spec.feedback_radius
    counts = ecological.sum(axis=1)
    local_cooperation = np.divide(ecological @ strategies.mean(axis=0), counts,
                                 out=np.zeros(len(counts)), where=counts > 0)
    valid = counts > 0
    association = None
    if valid.sum() > 1 and np.ptp(local_cooperation[valid]) > 0 and np.ptp(average_stock[valid]) > 0:
        association = float(np.corrcoef(local_cooperation[valid], average_stock[valid])[0, 1])
    fulfillment = request_fulfillment(delivered, requested)
    current, longest = np.zeros(agents,dtype=int), np.zeros(agents,dtype=int)
    for shortfall in fulfillment < 1:
        current = np.where(shortfall,current+1,0)
        longest = np.maximum(longest,current)
    result = dict(mean_accessible_stock=float(normalized.mean()),
                  spatial_sd_of_mean_stock=float(average_stock.std()),
                  minimum_time_mean_local_stock=float(average_stock.min()),
                  cooperation_spatial_autocorrelation=spatial_autocorrelation(strategies.mean(axis=0), neighbors),
                  local_cooperation_stock_association=association,
                  mean_requested_fraction_delivered=float(fulfillment.mean()),
                  fraction_agent_steps_with_unmet_request=float((fulfillment < 1).mean()),
                  worst_agent_continuous_shortage_steps=int(longest.max()),
                  worst_agent_mean_request_fulfillment=float(fulfillment.mean(axis=0).min()))
    for name, group in [('human', ~world.kinds), ('ai', world.kinds)]:
        result[f'{name}_mean_request_fulfillment'] = float(fulfillment[:, group].mean()) if group.any() else None
    return result
