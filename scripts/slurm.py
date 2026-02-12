"""Submit and manage Slurm jobs for Weigou experiments."""
import os
import subprocess
from enum import Enum
from pathlib import Path
from typing import List, Optional

from jinja2 import Template

from weigou.config import WeigouConfig


class Status(Enum):
    INIT = "init"
    PENDING = "pending"
    RUNNING = "running"
    FAIL = "fail"
    OOM = "oom"
    TIMEOUT = "timeout"
    COMPLETED = "completed"


class Job:
    def __init__(self, root_path: Path, qos: str) -> None:
        self.root_path = root_path
        self.name = root_path.name
        self.config_path = root_path / "config.json"
        self.qos = qos
        self.status_path = root_path / "status.txt"
        self._init_status()
        self.status = self.get_status()

    def _init_status(self) -> None:
        if not self.status_path.exists():
            self.status_path.write_text(Status.INIT.value)

    def get_status(self) -> Status:
        status = self.status_path.read_text().strip()
        if status not in {s.value for s in Status}:
            raise ValueError(f"Invalid status: {status}")
        return Status(status)

    def set_status(self, status: Status) -> Status:
        self.status_path.write_text(status.value)
        self.status = status
        return status


class Scheduler:
    def __init__(self, inp_dir: str, qos: str) -> None:
        self.inp_dir = Path(inp_dir)
        self.qos = qos
        self.jobs = self._discover_jobs()

    def _discover_jobs(self) -> List[Job]:
        paths = []
        for root, dirs, files in os.walk(self.inp_dir):
            if not dirs:
                path = Path(root).resolve()
                if "profiler" in str(path):
                    path = Path(str(path).replace("/profiler", ""))
                paths.append(path)
        return [Job(p, self.qos) for p in sorted(set(paths))]

    def filter_by_status(self, status: Status, keep: bool = True) -> List[Job]:
        if keep:
            return [j for j in self.jobs if j.status == status]
        return [j for j in self.jobs if j.status != status]

    def create_slurm_script(self, job: Job) -> None:
        if not job.config_path.exists():
            print(f"Config not found: {job.config_path}")
            return

        config = WeigouConfig.load(str(job.config_path))
        d = config.distributed
        world_size = d.tp_size * d.cp_size * d.pp_size * d.dp_size

        max_gpu_per_node = 8
        assert world_size <= max_gpu_per_node or world_size % max_gpu_per_node == 0
        nodes = max(1, world_size // max_gpu_per_node)
        procs = min(max_gpu_per_node, world_size // nodes)
        assert nodes * procs == world_size

        template_path = Path(__file__).resolve().parent.parent / "slurm" / "base_job.slurm"
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        ctx = {
            "nodes": nodes,
            "n_proc_per_node": procs,
            "root_path": str(job.root_path),
            "config": str(job.config_path),
            "qos": job.qos,
        }

        script = Template(template_path.read_text()).render(ctx)
        output_path = job.root_path / "job.slurm"
        output_path.write_text(script)
        print(f"Created {output_path}")

    def launch_with_deps(self, jobs: List[Job], env: dict) -> None:
        prev_id = None
        for job in jobs:
            cmd = ["sbatch", "--parsable"]
            if prev_id is not None:
                cmd.append(f"--dependency=afterany:{prev_id}")
            cmd.append(str(job.root_path / "job.slurm"))

            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Failed to submit {job.name}: {result.stderr}")
                continue
            job.set_status(Status.PENDING)
            prev_id = result.stdout.strip()

    def print_status(self) -> None:
        counts = {s.value: 0 for s in Status}
        for job in self.jobs:
            try:
                counts[job.get_status().value] += 1
            except Exception as e:
                print(f"Error for {job.name}: {e}")

        print(f"{'Status':<10} | {'Count':<6}")
        print("-" * 18)
        for status, count in counts.items():
            print(f"{status.capitalize():<10} | {count:<6}")
        print("-" * 18)
        print(f"{'Total':<10} | {sum(counts.values()):<6}")


def submit(inp_dir: str, qos: str, hf_token: str, nb_arrays: int = 0, only: Optional[str] = None) -> None:
    scheduler = Scheduler(inp_dir, qos)
    env = os.environ.copy()
    env["HUGGINGFACE_TOKEN"] = hf_token

    if only:
        status_map = {s.value: s for s in Status}
        if only not in status_map:
            print(f"Invalid status: {only}. Options: {list(status_map.keys())}")
            return
        scheduler.jobs = scheduler.filter_by_status(status_map[only], keep=True)
        if not scheduler.jobs:
            print(f"No jobs with status '{only}'")
            return
        print(f"Submitting {len(scheduler.jobs)} jobs with status '{only}'")

    scheduler.jobs = scheduler.filter_by_status(Status.COMPLETED, keep=False)

    if nb_arrays > 0:
        n = len(scheduler.jobs)
        base, extra = n // nb_arrays, n % nb_arrays
        sizes = [base + (1 if i < extra else 0) for i in range(nb_arrays)]
        start = 0
        for i, size in enumerate(sizes):
            if size == 0:
                continue
            batch = scheduler.jobs[start : start + size]
            print(f"Launching array {i + 1} with {size} jobs")
            for job in batch:
                scheduler.create_slurm_script(job)
            scheduler.launch_with_deps(batch, env)
            start += size
    else:
        for job in scheduler.jobs:
            scheduler.create_slurm_script(job)
            subprocess.run(["sbatch", str(job.root_path / "job.slurm")], env=env)
            job.set_status(Status.PENDING)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--inp_dir", type=str, required=True, help="Jobs directory")
    parser.add_argument("--qos", type=str, required=True)
    parser.add_argument("--hf_token", type=str, required=True)
    parser.add_argument("--nb_slurm_array", type=int, default=0)
    parser.add_argument("--only", type=str, default=None, help="Filter by status")

    args = parser.parse_args()
    submit(args.inp_dir, args.qos, args.hf_token, args.nb_slurm_array, args.only)


if __name__ == "__main__":
    main()
