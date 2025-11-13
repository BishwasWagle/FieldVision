# trainer/ppo_trainer.py
import os, time, json, signal, sys, zmq, torch, random
import numpy as np
from collections import defaultdict, deque

from common.message import trainer_pull_ctx
from common.broadcast import state_to_blob
from common.config import CKPT_DIR, AGENT_NAMES, N_ACTIONS, ROLLOUT_HORIZON
from common.utils import ensure_dirs
from common.models import ActorDiscrete, Critic

from trainer.aggregator import append_rollout_record, aggregate_to_csv
from trainer.plotter import make_plots

# ===== PPO hyperparams =====
GAMMA = 0.99
LAMBDA = 0.95
CLIP_RATIO = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
TRAIN_ITERS = 4
MINIBATCH = 2048

OBS_DIM = 4  # env obs per agent (regime one-hot)

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

def run_ppo_trainer():
    ensure_dirs(CKPT_DIR)
    device = torch.device("cpu")

    paths = init_run_dir("PPO")
    print(f"[PPO] Writing to {paths['RUN_DIR']}")

    actor = ActorDiscrete(OBS_DIM, N_ACTIONS).to(device)
    critic = Critic(OBS_DIM).to(device)
    opt_actor = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)

    ctx, pull, pub = trainer_pull_ctx()
    print("[PPO] Trainer online. Waiting for transitions...")

    # Per-agent buffers (shared across all agents)
    buf_obs, buf_act, buf_rew, buf_done, buf_next = [], [], [], [], []
    buf_len = 0
    step_counter = 0
    version = 0

    def broadcast_weights():
        blob = state_to_blob(actor.state_dict())
        pub.send_json({"type": "weights", "algo": "PPO", "version": version, "blob": blob.decode("ascii")})

    last_broadcast = time.time()
    broadcast_weights()

    # === Stop guards (env-configurable; 0 disables) ===
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
            print(f"[PPO] finalize snapshot error: {e}", file=sys.stderr)
        print(f"[PPO] {msg}  → CSV + figs saved under runs/current/")

    def _handle_signal(signum, frame):
        nonlocal STOP_REQUESTED
        STOP_REQUESTED = True
        print(f"[PPO] Signal {signum} received → will stop after current mini-batch.")
    signal.signal(signal.SIGINT,  _handle_signal)   # Ctrl-C
    signal.signal(signal.SIGTERM, _handle_signal)   # kill

    try:
        while True:
            msg = pull.recv_json()
            if msg.get("algo") != "PPO":
                continue

            # ---- Log a row for plotting/CSV (safe defaults if missing)
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

            # ---- Transition
            obs = np.array(msg["obs"], dtype=np.float32)
            act = int(msg["action"])
            rew = float(msg["reward"])
            done = bool(msg.get("done", False))
            nxt = np.array(msg["next_obs"], dtype=np.float32)

            buf_obs.append(obs); buf_act.append(act); buf_rew.append(rew); buf_done.append(done); buf_next.append(nxt)
            buf_len += 1
            step_counter += 1

            # Progress + quick CSV roll for live tailing
            if PRINT_EVERY and (step_counter % PRINT_EVERY == 0):
                elapsed = int(time.time() - t0)
                print(f"[PPO] steps={step_counter} updates={version} elapsed={elapsed}s")
                try:
                    aggregate_to_csv()
                except Exception:
                    pass

            # Periodic re-broadcast to keep late agents synced
            if time.time() - last_broadcast > 60:
                broadcast_weights()
                last_broadcast = time.time()

            # Train when we have enough samples
            if buf_len >= ROLLOUT_HORIZON:
                # to tensors
                obs_t = torch.tensor(np.stack(buf_obs), dtype=torch.float32, device=device)
                act_t = torch.tensor(buf_act, dtype=torch.long, device=device)
                rew_np = np.array(buf_rew, dtype=np.float32)
                done_np = np.array(buf_done, dtype=np.float32)

                with torch.no_grad():
                    val = critic(obs_t).cpu().numpy()
                    nxt_t = torch.tensor(np.stack(buf_next), dtype=torch.float32, device=device)
                    val_next = critic(nxt_t).cpu().numpy()
                values = np.concatenate([val, val_next[-1:]], axis=0)
                adv, ret = compute_gae(rew_np, values, done_np)

                # old logp
                with torch.no_grad():
                    logits = actor(obs_t)
                    dist = torch.distributions.Categorical(logits=logits)
                    old_logp = dist.log_prob(act_t)

                idx = np.arange(buf_len)
                for _ in range(TRAIN_ITERS):
                    np.random.shuffle(idx)
                    for start in range(0, buf_len, MINIBATCH):
                        mb = idx[start:start+MINIBATCH]
                        mb_obs = obs_t[mb]
                        mb_act = act_t[mb]
                        mb_adv = torch.tensor(adv[mb], dtype=torch.float32, device=device)
                        mb_ret = torch.tensor(ret[mb], dtype=torch.float32, device=device)
                        mb_old = old_logp[mb]

                        # critic
                        v = critic(mb_obs)
                        v_loss = ((v - mb_ret)**2).mean() * VF_COEF
                        opt_critic.zero_grad(); v_loss.backward(); opt_critic.step()

                        # actor
                        logits = actor(mb_obs)
                        dist = torch.distributions.Categorical(logits=logits)
                        logp = dist.log_prob(mb_act)
                        ratio = torch.exp(logp - mb_old)
                        unclipped = ratio * mb_adv
                        clipped = torch.clamp(ratio, 1-CLIP_RATIO, 1+CLIP_RATIO) * mb_adv
                        pi_loss = -torch.min(unclipped, clipped).mean()
                        ent = dist.entropy().mean()
                        loss = pi_loss - ENT_COEF * ent
                        opt_actor.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                        opt_actor.step()

                # clear buffer & bump version
                buf_obs.clear(); buf_act.clear(); buf_rew.clear(); buf_done.clear(); buf_next.clear()
                buf_len = 0
                version += 1
                broadcast_weights()
                print(f"[PPO] Updated -> version {version}, total steps {step_counter}")

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
    run_ppo_trainer()



# any combination; unset or 0 disables a guard
# export MAX_UPDATES=1200
# export MAX_STEPS=1500000
# export RUN_SECONDS=5400
# export PRINT_EVERY=10000
# python -m trainer.ppo_trainer
