import os, csv, shutil, time

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def write_csv_row(path, header, row_dict):
    write_header = (not os.path.exists(path))
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header: w.writeheader()
        w.writerow(row_dict)

def _safe_mkdir(p):
    os.makedirs(p, exist_ok=True)
    return p

def _replace_symlink(link_path: str, target_path: str):
    # Remove old symlink or file/dir named the same
    if os.path.islink(link_path) or os.path.exists(link_path):
        try:
            if os.path.islink(link_path):
                os.unlink(link_path)
            elif os.path.isdir(link_path):
                # don't delete old run data; only remove if it's an empty placeholder
                if not os.listdir(link_path):
                    os.rmdir(link_path)
            else:
                os.remove(link_path)
        except Exception:
            pass
    os.symlink(os.path.abspath(target_path), link_path)

def init_run_dir(algo: str) -> dict:
    """
    Create a timestamped run directory and wire env vars so
    aggregator/plotter write to the right place. Also updates
    a convenient 'runs/latest' symlink.
    """
    root = "runs"
    _safe_mkdir(root)

    # Allow manual run names; else timestamp
    run_tag = os.environ.get("RUN_NAME") or time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(root, f"{run_tag}_{algo.upper()}")

    raw_dir = _safe_mkdir(os.path.join(run_dir, "raw"))
    fig_dir = _safe_mkdir(os.path.join(run_dir, "figs"))
    ckpt_dir = _safe_mkdir(os.path.join(run_dir, "checkpoints"))

    # Update env so aggregator/plotter pick it up
    os.environ["RUN_DIR"] = run_dir
    os.environ["RAW_DIR"] = raw_dir
    os.environ["FIG_DIR"] = fig_dir
    os.environ["CKPT_DIR"] = ckpt_dir
    os.environ["LOG_CSV"] = os.path.join(run_dir, "marl_logs.csv")

    # Maintain a convenient 'latest' pointer
    _replace_symlink(os.path.join(root, "latest"), run_dir)

    return {
        "RUN_DIR": run_dir,
        "RAW_DIR": raw_dir,
        "FIG_DIR": fig_dir,
        "CKPT_DIR": ckpt_dir,
        "LOG_CSV": os.path.join(run_dir, "marl_logs.csv"),
    }
