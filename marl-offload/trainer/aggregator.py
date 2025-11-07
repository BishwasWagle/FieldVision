import os, json, time, pandas as pd
from common.utils import ensure_dirs
from common.config import RAW_DIR, LOG_CSV

def append_rollout_record(rec: dict):
    # rec should include agent, step, reward, latency_ms, regime, action, priority, succ, tardy_ms, algo
    os.makedirs(RAW_DIR, exist_ok=True)
    path = f"{RAW_DIR}/steps.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(rec)+"\n")

def aggregate_to_csv():
    # read jsonl -> aggregate per episode if you add episode id; here simple rolling append
    path = f"{RAW_DIR}/steps.jsonl"
    if not os.path.exists(path): return
    rows = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    if not rows: return
    df = pd.DataFrame(rows)
    # keep a light CSV to plot quickly
    use_cols = ["agent","step","reward","latency_ms","regime","action","priority","succ","tardy_ms","algo"]
    df[use_cols].to_csv(LOG_CSV, index=False)
