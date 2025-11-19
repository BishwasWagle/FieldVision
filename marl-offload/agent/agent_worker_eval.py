# agent/agent_worker_eval.py
"""
Modified agent worker that supports baseline algorithms for evaluation
"""
import os, time, json, zmq, torch
import numpy as np
from typing import Dict

from envs.offload_env import OffloadEnv
from common.message import agent_push_ctx
from common.models import ActorDiscrete

# Get configuration from environment
ALGO = os.environ.get("ALGO", "MMDE")
BASELINE_TYPE = os.environ.get("BASELINE_TYPE", "")
AGENT_ID = os.environ.get("AGENT_ID", "drone_0")
OBS_DIM = 4
N_ACTIONS = 3

class BaselineAgent:
    """Implements baseline policies for evaluation"""

    def __init__(self, baseline_type: str):
        self.baseline_type = baseline_type
        self.step_count = 0

    def select_action(self, obs, info=None):
        """Select action based on baseline policy"""
        self.step_count += 1

        if self.baseline_type == "LOCAL":
            return 0
        elif self.baseline_type == "EDGE":
            return 1
        elif self.baseline_type == "CLOUD":
            return 2
        elif self.baseline_type == "RANDOM":
            return np.random.randint(0, 3)
        elif self.baseline_type == "ROUND_ROBIN":
            return self.step_count % 3
        elif self.baseline_type == "LATENCY_GREEDY":
            # Use regime information to make decision
            if info and "regime" in info:
                regime = info["regime"]
                if regime == "outage":
                    return 0  # Stay local in outage
                elif regime == "good":
                    return 2  # Use cloud in good conditions
                else:
                    return 1  # Use edge otherwise
            return 1  # Default to edge
        else:
            return 0  # Default to local

def select_action(actor, obs_vec):
    """Select action using neural network policy"""
    x = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = actor(x)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
    return int(a.item())

def mmde_decision(info):
    """MMDE decision logic based on weighted metrics"""
    # Simplified MMDE logic
    latency = info.get("latency_ms", 100)
    deadline = info.get("deadline_ms", 200)
    priority = info.get("priority", "medium")
    regime = info.get("regime", "nominal")

    # Score each action
    scores = np.zeros(3)

    # Local: good for outages, low latency
    if regime == "outage":
        scores[0] = 1.0
    else:
        scores[0] = 0.3

    # Edge: balanced option
    if regime in ["good", "nominal"]:
        scores[1] = 0.7
    else:
        scores[1] = 0.4

    # Cloud: good for high priority, good conditions
    if priority == "high" and regime == "good":
        scores[2] = 0.9
    elif regime == "good":
        scores[2] = 0.6
    else:
        scores[2] = 0.2

    # Add noise for tie-breaking
    scores += np.random.randn(3) * 0.01

    return int(np.argmax(scores))

def main():
    """Main agent worker loop"""
    ctx, push, sub = agent_push_ctx()
    env = OffloadEnv(seed=42 + hash(AGENT_ID) % 9973)
    obs_dict = env.reset()

    # Initialize policy based on algorithm type
    if ALGO == "BASELINE":
        policy = BaselineAgent(BASELINE_TYPE)
        print(f"[Agent {AGENT_ID}] Starting with BASELINE policy: {BASELINE_TYPE}")
    elif ALGO in ("PPO", "MAPPO"):
        actor = ActorDiscrete(OBS_DIM, N_ACTIONS)
        actor.eval()
        print(f"[Agent {AGENT_ID}] Starting with {ALGO}")
    else:  # MMDE or other
        print(f"[Agent {AGENT_ID}] Starting with {ALGO}")

    step = 0
    info_dict = {}  # Store latest info for decision making

    while True:
        # Handle async weight updates from Trainer (for PPO/MAPPO)
        if ALGO in ("PPO", "MAPPO"):
            try:
                while True:
                    msg = sub.recv_json(flags=zmq.NOBLOCK)
                    if msg.get("type") == "weights" and msg.get("algo") == ALGO:
                        from common.broadcast import blob_to_state
                        state = blob_to_state(msg["blob"].encode("ascii"))
                        actor.load_state_dict(state, strict=False)
                        print(f"[Agent {AGENT_ID}] Loaded {ALGO} weights v{msg.get('version')}")
            except zmq.Again:
                pass

        # Select action based on algorithm
        act_by_agent: Dict[str, int] = {a: 0 for a in env.agent_list()}

        if ALGO == "BASELINE":
            # Use baseline policy
            obs_vec = np.array(obs_dict[AGENT_ID], dtype=np.float32)
            action = policy.select_action(obs_vec, info_dict.get(AGENT_ID))
        elif ALGO in ("PPO", "MAPPO"):
            # Use neural network policy
            obs_vec = np.array(obs_dict[AGENT_ID], dtype=np.float32)
            action = select_action(actor, obs_vec)
        elif ALGO == "MMDE":
            # Use MMDE heuristic
            if AGENT_ID in info_dict:
                action = mmde_decision(info_dict[AGENT_ID])
            else:
                action = np.random.randint(0, 3)
        else:
            action = 0

        act_by_agent[AGENT_ID] = int(action)

        # Step environment
        next_obs_dict, reward_dict, term, trunc, info_dict = env.step(act_by_agent)

        obs_vec = np.array(obs_dict[AGENT_ID], dtype=np.float32)
        next_vec = np.array(next_obs_dict[AGENT_ID], dtype=np.float32)
        rew = float(reward_dict[AGENT_ID])

        # Get additional metrics from info
        agent_info = info_dict.get(AGENT_ID, {})

        # Push transition to Trainer/Logger
        message = {
            "algo": BASELINE_TYPE if ALGO == "BASELINE" else ALGO,
            "agent": AGENT_ID,
            "step": step,
            "obs": obs_vec.tolist(),
            "action": int(action),
            "reward": rew,
            "next_obs": next_vec.tolist(),
            "done": False,
            # Include additional metrics for evaluation
            "latency_ms": agent_info.get("latency_ms"),
            "deadline_ms": agent_info.get("deadline_ms"),
            "tardy_ms": agent_info.get("tardy_ms"),
            "succ": agent_info.get("succ"),
            "priority": agent_info.get("priority"),
            "regime": agent_info.get("regime"),
            "payload_mb": agent_info.get("payload_mb"),
            "duration_s": agent_info.get("duration_s"),
            "video": agent_info.get("video"),
        }

        push.send_json(message)

        obs_dict = next_obs_dict
        step += 1

        # Rate limiting
        if step % 1000 == 0:
            time.sleep(0.1)
            print(f"[Agent {AGENT_ID}] Step {step}, Algo: {ALGO if ALGO != 'BASELINE' else BASELINE_TYPE}")

if __name__ == "__main__":
    main()
