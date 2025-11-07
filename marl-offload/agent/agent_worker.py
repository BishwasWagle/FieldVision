# agent/agent_worker.py
import os, time, json, zmq, torch
import numpy as np
from typing import Dict

from envs.offload_env import OffloadEnv
from common.message import agent_push_ctx
from common.models import ActorDiscrete

ALGO = os.environ.get("ALGO","MMDE")  # "MMDE" | "PPO" | "MAPPO"
AGENT_ID = os.environ.get("AGENT_ID","drone_0")
OBS_DIM = 4
N_ACTIONS = 3

def select_action(actor, obs_vec):
    x = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = actor(x)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
    return int(a.item())

def main():
    ctx, push, sub = agent_push_ctx()
    env = OffloadEnv(seed=42 + hash(AGENT_ID)%9973)
    obs_dict = env.reset()

    # Local actor only used for PPO/MAPPO
    actor = ActorDiscrete(OBS_DIM, N_ACTIONS)
    actor.eval()

    print(f"[Agent {AGENT_ID}] Starting with ALGO={ALGO}")
    step = 0

    while True:
        # Handle async weight updates from Trainer
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

        # Build joint action (single-process controls only itself; others default local=0)
        act_by_agent: Dict[str,int] = {a:0 for a in env.agent_list()}

        if ALGO in ("PPO","MAPPO"):
            # local obs vector for this agent (env.observe() returns dict of OHE regime)
            obs_vec = np.array(obs_dict[AGENT_ID], dtype=np.float32)
            action = select_action(actor, obs_vec)
        elif ALGO == "MMDE":
            # keep MMDE from previous file if you want; otherwise random
            action = np.random.randint(0,3)
        else:
            action = 0

        act_by_agent[AGENT_ID] = int(action)

        next_obs_dict, reward_dict, term, trunc, info_dict = env.step(act_by_agent)

        obs_vec = np.array(obs_dict[AGENT_ID], dtype=np.float32)
        next_vec = np.array(next_obs_dict[AGENT_ID], dtype=np.float32)
        rew = float(reward_dict[AGENT_ID])

        # Push transition to Trainer
        push.send_json({
            "algo": ALGO,
            "agent": AGENT_ID,
            "step": step,
            "obs": obs_vec.tolist(),
            "action": int(action),
            "reward": rew,
            "next_obs": next_vec.tolist(),
            "done": False,  # episodic logic omitted in this minimal env
        })

        obs_dict = next_obs_dict
        step += 1
        if step % 1000 == 0:
            time.sleep(0.1)

if __name__ == "__main__":
    main()
