# agent/mmde_policy.py
"""
MMDE = Multi-Metric Decision Engine
A handcrafted baseline that ranks actions using
weighted normalized metrics:
    - latency
    - deadline slack
    - priority
    - congestion / regime quality
    - compute cost vs upload cost
"""

import numpy as np

# Each agent environment returns obs = [regime_onehot(3), priority_code(1)]
# but full info is inside the 'info' dict the agent receives.

REGIME_SCORE = {
    "good": 1.0,
    "nominal": 0.7,
    "degraded": 0.3,
    "outage": 0.05
}

PRIORITY_W = {
    "high": 1.5,
    "medium": 1.0,
    "low": 0.7
}

# Action IDs
ACT_LOCAL = 0
ACT_EDGE  = 1
ACT_CLOUD = 2

def mmde_score_local(info):
    lat = info["latency_ms"]
    dead = info["deadline_ms"]
    pr = PRIORITY_W[info["priority"]]
    slack = max(0.0, (dead - lat) / dead)

    return (
        0.25 * slack
        + 0.25 * pr
        + 0.25 * 1.0     # no upload penalty
        + 0.25 * 0.5     # local compute may be slow; down-weight
    )


def mmde_score_edge(info):
    lat = info["latency_ms"]
    dead = info["deadline_ms"]
    pr = PRIORITY_W[info["priority"]]
    reg = REGIME_SCORE.get(info["regime"], 0.2)
    slack = max(0.0, (dead - lat) / dead)

    return (
        0.30 * reg       # network quality
        + 0.30 * slack
        + 0.30 * pr
        + 0.10 * 0.8     # compute speed advantage on edge
    )


def mmde_score_cloud(info):
    lat = info["latency_ms"]
    dead = info["deadline_ms"]
    pr = PRIORITY_W[info["priority"]]
    reg = REGIME_SCORE.get(info["regime"], 0.2)
    slack = max(0.0, (dead - lat) / dead)

    return (
        0.25 * reg
        + 0.25 * slack
        + 0.35 * pr      # high priority → cloud more recommended
        + 0.15 * 1.0     # cloud compute speed is highest
    )


def mmde_decide(info):
    """Return action 0/1/2 (local, edge, cloud)."""
    S_local = mmde_score_local(info)
    S_edge  = mmde_score_edge(info)
    S_cloud = mmde_score_cloud(info)

    scores = np.array([S_local, S_edge, S_cloud], dtype=np.float32)
    best = np.argmax(scores)

    # Tie-breaking with small noise
    if np.sum(scores == scores[best]) > 1:
        best = np.argmax(scores + 1e-3 * np.random.randn(3))

    return int(best)
