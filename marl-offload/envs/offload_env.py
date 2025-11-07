import os, random, numpy as np
from typing import Dict, Any, Tuple, List

from common.config import (
    AGENT_NAMES, AGENT_PROFILES, N_ACTIONS,
    PROC_MS_PER_SEC, ALPHA_LAT_MS, BETA_ENERGY, GAMMA_SUCC, LAMBDA_TARDY,
    REGIME_SEQUENCE,
)
from common.video_tasks import list_videos_for_agent, task_from_video
from common.net_regimes import sample_link_for_agent, transfer_time_seconds

class OffloadEnv:
    """
    Whole-video tasks:
      - Each agent consumes a queue of video files (one file = one task).
      - Action: 0 local, 1 edge (GPU), 2 cloud.
      - Latency = upload_time (if offloaded) + compute_time (scaled by duration).
      - Reward: soft-deadline + priority.
    """
    def __init__(self, seed=0):
        random.seed(seed); np.random.seed(seed)
        self.agents: List[str] = list(AGENT_NAMES)
        self.regime_idx = 0
        self.t = 0
        self.video_queues: Dict[str, List[Dict[str,Any]]] = {}
        self._init_queues()

    def _init_queues(self):
        self.video_queues = {}
        for a in self.agents:
            paths = list_videos_for_agent(a)
            tasks = [task_from_video(p) for p in paths]
            # loop forever by cycling through videos
            self.video_queues[a] = tasks

    def current_regime(self):
        return REGIME_SEQUENCE[self.regime_idx % len(REGIME_SEQUENCE)]

    def _compute_ms(self, agent: str, action: int, duration_s: float) -> float:
        prof = AGENT_PROFILES[agent]
        key = (
            "local_low" if (action==0 and prof=="low") else
            "local_high" if (action==0 and prof=="high") else
            "edge" if action==1 else
            "cloud"
        )
        k = PROC_MS_PER_SEC[key]
        noise = np.random.uniform(0.9, 1.1)
        return float(k * duration_s * noise)

    def _transfer_ms(self, agent: str, action: int, payload_mb: float, regime: str) -> float:
        if action == 0:  # local
            return 0.0
        link = "drone_edge" if action==1 else "drone_cloud"
        bw, rtt, loss = sample_link_for_agent(agent, link, regime)
        xfer_s = transfer_time_seconds(payload_mb, bw, rtt)
        # crude retry on loss
        if np.random.rand() < loss:
            xfer_s *= 1.5
        return float(xfer_s * 1000.0)

    def reset(self):
        self.t = 0
        self.regime_idx = 0
        self._init_queues()
        return self.observe()

    def step(self, action_by_agent: Dict[str,int]):
        reg = self.current_regime()
        rewards = {}
        infos = {}
        done = False; truncated = False

        # One task per agent (cycle through queue)
        for a in self.agents:
            task = self.video_queues[a][ self.t % len(self.video_queues[a]) ]
            act = int(action_by_agent.get(a, 0))

            xfer_ms = self._transfer_ms(a, act, task["payload_mb"], reg)
            comp_ms = self._compute_ms(a, act, task["duration_s"])
            total_ms = xfer_ms + comp_ms

            tardy_ms = max(0.0, total_ms - task["deadline_ms"])
            succ = 1.0 if tardy_ms <= 1e-6 else 0.0

            r = -ALPHA_LAT_MS*total_ms - LAMBDA_TARDY*tardy_ms + GAMMA_SUCC*succ - BETA_ENERGY*0.0
            r *= task["priority_w"]

            rewards[a] = float(r)
            infos[a] = {
                "latency_ms": float(total_ms),
                "deadline_ms": float(task["deadline_ms"]),
                "tardy_ms": float(tardy_ms),
                "succ": bool(succ),
                "priority": task["priority"],
                "action": act,
                "regime": reg,
                "payload_mb": float(task["payload_mb"]),
                "duration_s": float(task["duration_s"]),
                "video": task["video_path"],
            }

        self.t += 1
        if self.t % 100 == 0:
            self.regime_idx += 1

        return self.observe(), rewards, done, truncated, infos

    def observe(self) -> Dict[str, Any]:
        reg = self.current_regime()
        reg_ohe = {
            "good":    [1,0,0,0],
            "nominal": [0,1,0,0],
            "degraded":[0,0,1,0],
            "outage":  [0,0,0,1],
        }[reg]
        return {a: np.array(reg_ohe, dtype=np.float32) for a in self.agents}

    def agent_list(self): return list(self.agents)
