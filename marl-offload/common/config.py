from dataclasses import dataclass
from typing import Dict, List

# Trainer wiring (unchanged)
TRAINER_IP = "172.31.82.206"
ZMQ_PORT_ROLLOUT = 5555
ZMQ_PORT_CMD     = 5556

AGENT_NAMES   = ["drone_0","drone_1","drone_2","drone_3"]
AGENT_PROFILES= {"drone_0":"low","drone_1":"low","drone_2":"high","drone_3":"high"}

# === Workload: VIDEO tasks (one file = one task) ===
# Each agent will pick from its local /data/videos folder by default.
# You can pin specific files per agent if you want deterministic mapping:
VIDEO_ROOT = "data/videos"
# Optional: per-agent explicit lists (leave {} to auto-scan VIDEO_ROOT on each agent)
AGENT_VIDEOS: Dict[str, List[str]] = {
    "drone_0": ["DJI_0604_30s.MOV","DJI_0604_45s.MOV"],
    "drone_1": ["DJI_0604_60s.MOV"],
    "drone_2": ["DJI_0604_90s.MOV", "DJI_0604_30s.MOV"],
    "drone_3": ["DJI_0604_120s.MOV", "DJI_0604_45s.MOV"],
}

# === Compute time model for whole videos ===
# Processing time is proportional to video duration (seconds):
#   compute_ms ≈ k[target] * duration_secs
# These are “effective” constants that fold model complexity + hardware speed.
PROC_MS_PER_SEC = {
    "local_low":  60.0,   # ~35 ms per second of video on low drone
    "local_high": 45.0,   # ~15 ms per second of video on high drone
    "edge":        30.0,   # edge faster
    "cloud":       15.0,   # cloud fastest
}

# === Deadlines / priorities (still used) ===
DEADLINE_BOUNDS = {"high": (20_000, 60_000), "medium": (60_000, 120_000), "low": (120_000, 240_000)}
PRIORITY_WEIGHTS = {"high": 1.5, "medium": 1.0, "low": 0.7}

# === Reward shaping (unchanged) ===
ALPHA_LAT_MS = 0.0008
BETA_ENERGY  = 0.0
GAMMA_SUCC   = 1.0
LAMBDA_TARDY = 0.0015

# === Regime schedules per LINK TYPE (drone-drone, drone-edge/GPU, drone-cloud) ===
REGIME_SEQUENCE = ["nominal"]*300 + ["degraded"]*200 + ["good"]*300

# === Per-agent congestion intensities (multipliers applied to sampled link stats)
# Heavier congestion → lower BW, higher RTT, slightly higher loss.
AGENT_CONGESTION = {
    # levels: "light", "medium", "heavy"
    "drone_0": "heavy",
    "drone_1": "medium",
    "drone_2": "light",
    "drone_3": "light",
}
CONGESTION_MULTS = {
    "light":  {"bw_mult": 1.0,  "rtt_mult": 1.0,  "loss_mult": 1.0},
    "medium": {"bw_mult": 0.75, "rtt_mult": 1.3,  "loss_mult": 1.2},
    "heavy":  {"bw_mult": 0.5,  "rtt_mult": 1.8,  "loss_mult": 1.5},
}

# PPO/MAPPO
N_ACTIONS = 3
ROLLOUT_HORIZON = 1024
TOTAL_STEPS     = 200_000

# Paths on trainer
RUN_DIR = "./runs/current"
RAW_DIR = f"{RUN_DIR}/raw"
CKPT_DIR= f"{RUN_DIR}/checkpoints"
FIG_DIR = f"{RUN_DIR}/figs"
LOG_CSV = f"{RUN_DIR}/marl_logs.csv"
