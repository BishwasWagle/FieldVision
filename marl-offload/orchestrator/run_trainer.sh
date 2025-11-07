#!/usr/bin/env bash
set -e
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

mkdir -p runs/current/raw runs/current/checkpoints runs/current/figs

# Start trainer (MAPPO wiring; also used by PPO stub)
python -u trainer/mappo_trainer.py &
TRAINER_PID=$!

echo "[Trainer] PID=$TRAINER_PID"
echo "[Trainer] Tail logs with: tail -f runs/current/marl_logs.csv (after first aggregate)"
