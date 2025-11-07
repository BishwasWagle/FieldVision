#!/usr/bin/env bash
set -e
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Set your identity and algorithm for this agent
export AGENT_ID=${AGENT_ID:-drone_0}
export ALGO=${ALGO:-MMDE}     # or PPO or MAPPO later

python -u agent/agent_worker.py
