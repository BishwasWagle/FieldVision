# trainer/mappo_trainer.py
import os, time, json, signal, sys, zmq, torch, numpy as np
from collections import defaultdict, deque

from common.message import trainer_pull_ctx
from common.broadcast import state_to_blob
from common.config import CKPT_DIR, AGENT_NAMES, N_ACTIONS, ROLLOUT_HORIZON
from common.utils import ensure_dirs, init_run_dir
from common.models import ActorDiscrete, CentralCritic

from trainer.aggregator import append_rollout_record, aggregate_to_csv
from trainer.plotter import make_plots

# ===== Hyperparams (aligned with PPO) =====
GAMMA = 0.99
LAMBDA = 0.95
CLIP_RATIO = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
TRAIN_ITERS = 4
MINIBATCH = 2048

OBS_DIM = 4                              # per-agent obs
GLOBAL_DIM = OBS_DIM * len(AGENT_NAMES)  # concat of all agents' obs

def compute_gae(rewards, values, dones, gamma=GAMMA, lam=LAMBDA):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nextnonterm = 1.0 - float(dones[t])
        nextvalue = values[t+1] if t+1 < T else 0.0
        delta = rewards[t] + gamma*nextvalue*nextnonterm - values[t]
        lastgaelam = delta + gamma*lam*nextnonterm*lastgaelam
        adv[t] = lastgaelam
    returns = adv + values[:T]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, returns

def run_mappo_trainer():
    ensure_dirs(CKPT_DIR)
    device = torch.device("cpu")
    
    # CREATE A NEW RUN FOLDER AND POINT EVERYTHING TO IT
    paths = init_run_dir("MAPPO")
    print(f"[MAPPO] Writing to {paths['RUN_DIR']}")

    actor = ActorDiscrete(OBS_DIM, N_ACTIONS).to(device)
    critic = CentralCritic(GLOBAL_DIM).to(device)
    opt_actor = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)

    ctx, pull, pub = trainer_pull_ctx()
    print("[MAPPO] Trainer online. Waiting for transitions...")

    # --- Buffers for trajectories (shared actor; centralized critic)
    buf_obs_local, buf_act, buf_rew, buf_done, buf_next_local = [], [], [], [], []
    buf_global, buf_next_global = [], []
    buf_len = 0
    version = 0
    step_counter = 0

    # Track latest obs per agent to build global vectors
    latest_obs  = {a: np.zeros(OBS_DIM, dtype=np.float32) for a in AGENT_NAMES}
    latest_next = {a: np.zeros(OBS_DIM, dtype=np.float32) for a in AGENT_NAMES}

    def build_global(o_dict):
        parts = [o_dict[a] for a in AGENT_NAMES]
        return np.concatenate(parts, axis=0)

    def broadcast_weights():
        blob = state_to_blob(actor.state_dict())
        pub.send_json({"type": "weights", "algo": "MAPPO", "version": version, "blob": blob.decode("ascii")})

    # Initial broadcast
    broadcast_weights()
    last_broadcast = time.time()

    # === Stop guards (configurable via env; 0 disables)
    MAX_UPDATES = int(os.environ.get("MAX_UPDATES", "0"))
    MAX_STEPS   = int(os.environ.get("MAX_STEPS",   "0"))
    RUN_SECONDS = int(os.environ.get("RUN_SECONDS", "0"))
    PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "5000"))

    t0 = time.time()
    STOP_REQUESTED = False

    def finalize_and_snapshot(msg: str):
        try:
            aggregate_to_csv()
            make_plots()
        except Exception as e:
            print(f"[Trainer] finalize snapshot error: {e}", file=sys.stderr)
        print(f"[Trainer] {msg}  → CSV + figs saved under runs/current/")

    def _handle_signal(signum, frame):
        nonlocal STOP_REQUESTED
        STOP_REQUESTED = True
        print(f"[Trainer] Signal {signum} received → will stop after current mini-batch.")
    signal.signal(signal.SIGINT,  _handle_signal)   # Ctrl-C
    signal.signal(signal.SIGTERM, _handle_signal)   # kill

    try:
        while True:
            msg = pull.recv_json()

            if msg.get("algo") != "MAPPO":
                continue

            # ---- Log a row for plotting/CSV (safe defaults if missing)
            append_rollout_record({
                "agent":    msg.get("agent",""),
                "step":     int(msg.get("step", 0)),
                "reward":   float(msg.get("reward", 0.0)),

                # keep raw values; if missing, let them be None -> becomes NaN in CSV
                "latency_ms": msg.get("latency_ms", None),
                "regime":     msg.get("regime", None),
                "priority":   msg.get("priority", None),
                "succ":       msg.get("succ", None),
                "tardy_ms":   msg.get("tardy_ms", None),

                "algo":     "MAPPO",   # or "PPO" in ppo_trainer
                "video":      msg.get("video", None),
                "payload_mb": msg.get("payload_mb", None),
                "duration_s": msg.get("duration_s", None),
            })


            # ---- Unpack transition for training
            agent = msg["agent"]
            obs   = np.array(msg["obs"], dtype=np.float32)
            nxt   = np.array(msg["next_obs"], dtype=np.float32)
            act   = int(msg["action"])
            rew   = float(msg["reward"])
            done  = bool(msg.get("done", False))

            # Update trackers and build global snapshots
            latest_obs[agent]  = obs
            latest_next[agent] = nxt
            glob      = build_global(latest_obs)
            glob_next = build_global(latest_next)

            # Fill buffers
            buf_obs_local.append(obs)
            buf_act.append(act)
            buf_rew.append(rew)
            buf_done.append(done)
            buf_next_local.append(nxt)
            buf_global.append(glob)
            buf_next_global.append(glob_next)
            buf_len += 1
            step_counter += 1

            # Periodic progress + periodic re-broadcast of weights (keep agents in sync)
            if PRINT_EVERY and (step_counter % PRINT_EVERY == 0):
                elapsed = int(time.time() - t0)
                print(f"[MAPPO] steps={step_counter} updates={version} elapsed={elapsed}s")
                # also roll CSV quickly so you can tail it live
                try:
                    aggregate_to_csv()
                except Exception:
                    pass

            if time.time() - last_broadcast > 60:
                broadcast_weights()
                last_broadcast = time.time()

            # ---- Train when horizon reached
            if buf_len >= ROLLOUT_HORIZON:
                # tensors
                obs_t  = torch.tensor(np.stack(buf_obs_local), dtype=torch.float32, device=device)
                act_t  = torch.tensor(buf_act, dtype=torch.long, device=device)
                glob_t = torch.tensor(np.stack(buf_global), dtype=torch.float32, device=device)

                rew_np  = np.array(buf_rew, dtype=np.float32)
                done_np = np.array(buf_done, dtype=np.float32)

                with torch.no_grad():
                    v = critic(glob_t).cpu().numpy()
                    glob_next_t = torch.tensor(np.stack(buf_next_global), dtype=torch.float32, device=device)
                    v_next = critic(glob_next_t).cpu().numpy()
                values = np.concatenate([v, v_next[-1:]], axis=0)
                adv, ret = compute_gae(rew_np, values, done_np)

                with torch.no_grad():
                    logits = actor(obs_t)
                    dist   = torch.distributions.Categorical(logits=logits)
                    old_logp = dist.log_prob(act_t)

                idx = np.arange(buf_len)
                for _ in range(TRAIN_ITERS):
                    np.random.shuffle(idx)
                    for start in range(0, buf_len, MINIBATCH):
                        mb = idx[start:start+MINIBATCH]
                        mb_obs  = obs_t[mb]
                        mb_glob = glob_t[mb]
                        mb_act  = act_t[mb]
                        mb_adv  = torch.tensor(adv[mb], dtype=torch.float32, device=device)
                        mb_ret  = torch.tensor(ret[mb], dtype=torch.float32, device=device)
                        mb_old  = old_logp[mb]

                        # critic on global obs
                        v = critic(mb_glob)
                        v_loss = ((v - mb_ret)**2).mean() * VF_COEF
                        opt_critic.zero_grad(); v_loss.backward(); opt_critic.step()

                        # actor on local obs
                        logits = actor(mb_obs)
                        dist   = torch.distributions.Categorical(logits=logits)
                        logp   = dist.log_prob(mb_act)
                        ratio  = torch.exp(logp - mb_old)
                        unclipped = ratio * mb_adv
                        clipped   = torch.clamp(ratio, 1-CLIP_RATIO, 1+CLIP_RATIO) * mb_adv
                        pi_loss = -torch.min(unclipped, clipped).mean()
                        ent     = dist.entropy().mean()
                        loss    = pi_loss - ENT_COEF * ent
                        opt_actor.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                        opt_actor.step()

                # clear & broadcast
                buf_obs_local.clear(); buf_act.clear(); buf_rew.clear(); buf_done.clear(); buf_next_local.clear()
                buf_global.clear(); buf_next_global.clear()
                buf_len = 0
                version += 1
                broadcast_weights()
                print(f"[MAPPO] Updated -> version {version}, total steps {step_counter}")

                # snapshot metrics/figs after each update
                try:
                    aggregate_to_csv()
                    make_plots()
                except Exception:
                    pass

                # --- Stop guards tied to updates / time / signal
                if MAX_UPDATES and version >= MAX_UPDATES:
                    finalize_and_snapshot(f"Reached MAX_UPDATES={MAX_UPDATES}. Stopping.")
                    break
                if RUN_SECONDS and (time.time() - t0) >= RUN_SECONDS:
                    finalize_and_snapshot(f"Reached RUN_SECONDS={RUN_SECONDS}s. Stopping.")
                    break
                if STOP_REQUESTED:
                    finalize_and_snapshot("Stop requested by signal.")
                    break

            # --- Stop guards that may trip between updates
            if MAX_STEPS and step_counter >= MAX_STEPS:
                finalize_and_snapshot(f"Reached MAX_STEPS={MAX_STEPS}. Stopping.")
                break
            if RUN_SECONDS and (time.time() - t0) >= RUN_SECONDS:
                finalize_and_snapshot(f"Reached RUN_SECONDS={RUN_SECONDS}s. Stopping.")
                break
            if STOP_REQUESTED:
                finalize_and_snapshot("Stop requested by signal.")
                break

    except KeyboardInterrupt:
        finalize_and_snapshot("Interrupted (KeyboardInterrupt).")
    finally:
        finalize_and_snapshot("Run finished.")

if __name__ == "__main__":
    run_mappo_trainer()


# # any combination; 0 (unset) disables a guard
# export MAX_UPDATES=1500
# export MAX_STEPS=2000000
# export RUN_SECONDS=500
# export PRINT_EVERY=10000
# python -m trainer.mappo_trainer
