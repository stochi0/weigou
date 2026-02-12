"""Extract metrics from training log files and create summary CSVs."""
import re
import csv
import argparse
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def parse_folder_name(folder_name: str) -> Dict[str, Optional[int]]:
    patterns = {
        "dp": r"dp(\d+)",
        "tp": r"tp(\d+)",
        "pp": r"pp(\d+)",
        "micro_batch_size": r"mbs(\d+)",
        "grad_acc": r"ga(\d+)",
        "seq_len": r"sl(\d+)",
    }
    return {
        k: int(m.group(1)) if (m := re.search(p, folder_name)) else None
        for k, p in patterns.items()
    }


def from_readable_format(s: Union[str, float]) -> float:
    if not isinstance(s, str):
        return float(s)
    s = s.strip().upper()
    try:
        return float(s)
    except ValueError:
        pass
    multipliers = {"T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3}
    if not s:
        return 0.0
    suffix = s[-1]
    if suffix in multipliers:
        return float(s[:-1]) * multipliers[suffix]
    raise ValueError(f"Unknown format: {s}")


def parse_log_line(line: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        tokens_match = re.search(r"Tokens/s/GPU:\s*([\d.]+[KMBT]?)", line)
        mfu_match = re.search(r"MFU:\s+(\d+\.\d+)%", line)
        mfu = float(mfu_match.group(1)) if mfu_match else None
        tokens = from_readable_format(tokens_match.group(1)) if tokens_match else None
        return mfu, tokens
    except Exception as e:
        print(f"Parse error for line: {line.strip()}: {e}")
        return None, None


def process_file(filepath: Path) -> Tuple[Optional[int], Optional[int]]:
    tokens_values, mfu_values = [], []

    try:
        with open(filepath) as f:
            for line in f:
                if "[rank" in line or re.search(r"\[default\d+\]:\[rank \d+\]", line):
                    mfu, tokens = parse_log_line(line)
                    if tokens is not None:
                        tokens_values.append(tokens)
                    if mfu is not None:
                        mfu_values.append(mfu)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None, None

    if len(tokens_values) < 4 and len(mfu_values) < 4:
        return None, None

    avg_tokens = int(round(np.mean(tokens_values[3:]))) if len(tokens_values) > 3 else None
    avg_mfu = int(round(np.mean(mfu_values[3:]))) if len(mfu_values) > 3 else None
    return avg_mfu, avg_tokens


def write_csv(data: Dict[str, Any], output_path: Path) -> None:
    if not data:
        return
    fieldnames = [
        "run_name", "status", "dp", "tp", "pp",
        "micro_batch_size", "grad_acc", "seq_len",
        "avg_tokens_s_gpu", "avg_mfu",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(data)


def read_status(status_file: Path) -> Optional[str]:
    try:
        if status_file.exists():
            return status_file.read_text().strip()
    except Exception:
        pass
    return None


def process_subdirs(input_folder: Path) -> List[Path]:
    processed = []
    for filepath in input_folder.rglob("*.out"):
        dir_path = filepath.parent
        output_csv = dir_path / "metrics.csv"
        status_file = dir_path / "status.txt"

        avg_mfu, avg_tokens = process_file(filepath)
        if avg_mfu is None and avg_tokens is None:
            continue

        params = parse_folder_name(dir_path.name)
        params["run_name"] = dir_path.name
        params["status"] = read_status(status_file) or ""
        params["avg_tokens_s_gpu"] = avg_tokens if avg_tokens is not None else -1
        params["avg_mfu"] = avg_mfu if avg_mfu is not None else -1

        write_csv(params, output_csv)
        processed.append(dir_path)
        print(f"Processed {filepath} -> metrics.csv")

    return processed


def aggregate_metrics(input_folder: Path) -> None:
    for top_dir in input_folder.iterdir():
        if not top_dir.is_dir():
            continue
        rows = []
        for subdir in top_dir.iterdir():
            if not subdir.is_dir():
                continue
            metrics_file = subdir / "metrics.csv"
            if metrics_file.exists():
                try:
                    with open(metrics_file) as f:
                        rows.append(next(csv.DictReader(f)))
                except Exception as e:
                    print(f"Error reading {metrics_file}: {e}")
        if rows:
            output = top_dir / "global_metrics.csv"
            fieldnames = list(rows[0].keys())
            with open(output, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()
                csv.DictWriter(f, fieldnames=fieldnames).writerows(rows)
            print(f"Created {output} with {len(rows)} entries")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", type=str, help="Folder containing experiment subfolders")
    args = parser.parse_args()

    path = Path(args.input_folder)
    if not path.exists():
        print(f"Directory not found: {path}")
        return

    print("Creating metrics.csv files...")
    process_subdirs(path)
    print("\nAggregating metrics...")
    aggregate_metrics(path)


if __name__ == "__main__":
    main()
