import os, csv

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def write_csv_row(path, header, row_dict):
    write_header = (not os.path.exists(path))
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header: w.writeheader()
        w.writerow(row_dict)
