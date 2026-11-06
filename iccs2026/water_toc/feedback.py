"""Identified replenishment laws for controlled feedback comparisons."""

import numpy as np

FEEDBACK_RULES = ("repository", "pre_floor_change", "manuscript")


def replenishment_multiplier(n, cooperation, theta, rule="repository"):
    if rule not in FEEDBACK_RULES:
        raise ValueError(f"feedback_rule must be one of {FEEDBACK_RULES}")
    if rule == "manuscript":
        feedback = theta * cooperation - (1 - cooperation)
    else:
        feedback = theta * cooperation - 1.0
    floor = 1.0 if rule == "repository" else 0.1
    return np.clip(1.0 + 0.5 * n * (1 - n) * feedback, floor, 3.0)
