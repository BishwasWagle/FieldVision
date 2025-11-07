import os, re, random, math
from typing import Dict, Any, List
from common.config import VIDEO_ROOT, AGENT_VIDEOS, DEADLINE_BOUNDS, PRIORITY_WEIGHTS

_DUR_RE = re.compile(r'_(\d+)s(?:\.|$)', re.IGNORECASE)

def list_videos_for_agent(agent: str) -> List[str]:
    explicit = AGENT_VIDEOS.get(agent)
    if explicit:
        return [os.path.join(VIDEO_ROOT, f) for f in explicit]
    # auto-scan VIDEO_ROOT
    files = [f for f in os.listdir(VIDEO_ROOT) if f.lower().endswith((".MOV",".mp4",".mkv",".avi"))]
    files.sort()
    return [os.path.join(VIDEO_ROOT, f) for f in files]

def parse_duration_seconds(path: str) -> float:
    """Try to infer duration from filename like DJI_0604_60s.MOV; fallback: size-based guess."""
    name = os.path.basename(path)
    m = _DUR_RE.search(name)
    if m:
        return float(m.group(1))
    # fallback: rough heuristic (very coarse)
    sz_mb = os.path.getsize(path) / (1024*1024.0)
    return max(10.0, min(300.0, sz_mb / 5.0))  # assume ~5 MB/s encoded rate

def task_from_video(path: str) -> Dict[str, Any]:
    size_mb = os.path.getsize(path) / (1024*1024.0)
    duration_s = parse_duration_seconds(path)
    p = random.choices(["high","medium","low"], weights=[0.2,0.5,0.3], k=1)[0]
    dl_lo, dl_hi = DEADLINE_BOUNDS[p]
    deadline_ms = random.uniform(dl_lo, dl_hi)
    return {
        "video_path": path,
        "payload_mb": size_mb,     # upload size to edge/cloud
        "duration_s": duration_s,  # compute cost scales with this
        "priority": p,
        "priority_w": PRIORITY_WEIGHTS[p],
        "deadline_ms": deadline_ms,
    }
