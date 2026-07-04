from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_command(command: list[str], timeout: int = 20) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return {"available": False, "command": command, "error": "command not found"}
    except subprocess.TimeoutExpired:
        return {"available": True, "command": command, "error": "timed out"}
    return {
        "available": True,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def parse_gpu_summary(output: str) -> dict[str, Any]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return {"available": False}
    first = [part.strip() for part in lines[0].split(",")]
    if len(first) < 5:
        return {"available": False, "raw": output}
    name, util, memory_used, memory_total, temperature = first[:5]
    return {
        "available": True,
        "name": name,
        "utilization_percent": float(util),
        "memory_used_mb": float(memory_used),
        "memory_total_mb": float(memory_total),
        "temperature_c": float(temperature),
    }


def gpu_status(idle_util_percent: float, idle_memory_mb: float) -> dict[str, Any]:
    query = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    summary_result = run_command(query)
    summary = parse_gpu_summary(summary_result.get("stdout", ""))
    compute_result = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ]
    )
    visible_processes = [
        line.strip()
        for line in compute_result.get("stdout", "").splitlines()
        if line.strip() and "No running processes" not in line
    ]
    compute_processes = [process for process in visible_processes if "[N/A]" not in process]
    graphics_or_unknown_processes = [process for process in visible_processes if "[N/A]" in process]
    summary["compute_processes"] = compute_processes
    summary["graphics_or_unknown_processes"] = graphics_or_unknown_processes
    if summary.get("available"):
        summary["idle_for_robotics_work"] = (
            summary["utilization_percent"] <= idle_util_percent
            and summary["memory_used_mb"] <= idle_memory_mb
            and not compute_processes
        )
    summary["raw_query"] = summary_result
    summary["raw_compute_process_query"] = compute_result
    return summary


def kaggle_status(limit: int) -> dict[str, Any]:
    credential_path = Path.home() / ".kaggle" / "kaggle.json"
    status: dict[str, Any] = {
        "credential_file_present": credential_path.exists(),
        "python_module_available": False,
        "recent_kernels": [],
    }
    module_check = run_command([sys.executable, "-m", "kaggle", "--version"], timeout=20)
    status["python_module_available"] = bool(module_check.get("available")) and module_check.get("returncode") == 0
    status["version_check"] = module_check
    if not status["credential_file_present"] or not status["python_module_available"]:
        return status

    list_result = run_command(
        [sys.executable, "-m", "kaggle", "kernels", "list", "--mine", "--page-size", str(limit)],
        timeout=30,
    )
    status["kernel_list"] = list_result
    refs = parse_kernel_refs(list_result.get("stdout", ""), limit)
    for ref in refs:
        kernel = {"ref": ref}
        kernel["status"] = run_command([sys.executable, "-m", "kaggle", "kernels", "status", ref], timeout=30)
        status["recent_kernels"].append(kernel)
    return status


def parse_kernel_refs(output: str, limit: int) -> list[str]:
    refs: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("ref ") or stripped.startswith("-"):
            continue
        first = stripped.split()[0]
        if "/" in first:
            refs.append(first)
        if len(refs) >= limit:
            break
    return refs


def print_summary(report: dict[str, Any]) -> None:
    gpu = report["gpu"]
    if gpu.get("available"):
        idle = "idle" if gpu.get("idle_for_robotics_work") else "busy"
        print(
            f"GPU: {gpu['name']} | {gpu['utilization_percent']:.0f}% util | "
            f"{gpu['memory_used_mb']:.0f}/{gpu['memory_total_mb']:.0f} MiB | {idle}"
        )
        if gpu.get("compute_processes"):
            print("Compute processes:")
            for process in gpu["compute_processes"]:
                print(f"- {process}")
        else:
            print("Compute processes: none with allocated CUDA memory reported by nvidia-smi")
        graphics_count = len(gpu.get("graphics_or_unknown_processes", []))
        if graphics_count:
            print(f"Graphics/WDDM processes ignored for compute-idle decision: {graphics_count}")
    else:
        print("GPU: unavailable")

    kaggle = report.get("kaggle")
    if kaggle is None:
        return
    print(f"Kaggle credential file: {kaggle['credential_file_present']}")
    print(f"Kaggle Python module: {kaggle['python_module_available']}")
    for kernel in kaggle.get("recent_kernels", []):
        stdout = kernel["status"].get("stdout", "")
        print(f"Kaggle kernel: {kernel['ref']} | {stdout}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle", action="store_true", help="Probe Kaggle CLI/API status.")
    parser.add_argument("--kaggle-limit", type=int, default=3)
    parser.add_argument("--idle-util-percent", type=float, default=35.0)
    parser.add_argument("--idle-memory-mb", type=float, default=2048.0)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": PROJECT_ROOT.name,
        "nvidia_smi_on_path": shutil.which("nvidia-smi") is not None,
        "gpu": gpu_status(args.idle_util_percent, args.idle_memory_mb),
    }
    if args.kaggle:
        report["kaggle"] = kaggle_status(args.kaggle_limit)

    print_summary(report)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
