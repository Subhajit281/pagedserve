import os
import platform
import subprocess
import sys

import torch


def run_command(command):
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        return f"ERROR: {result.stderr.strip()}"

    return result.stdout.strip()


def main():
    print("=" * 60)
    print("              PagedServe Environment")
    print("=" * 60)

    print("\n--- System ---")
    print("Python       :", sys.version.split()[0])
    print("OS           :", platform.platform())
    print("Architecture :", platform.machine())
    print("CPU threads  :", os.cpu_count())

    print("\n--- PyTorch ---")
    print("PyTorch      :", torch.__version__)
    print("CUDA runtime :", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)

        print("\n--- GPU ---")
        print("GPU          :", torch.cuda.get_device_name(0))
        print(
            "VRAM         :",
            round(props.total_memory / (1024**3), 2),
            "GiB",
        )
        print(
            "Compute      :",
            f"{props.major}.{props.minor}",
        )

        free, total = torch.cuda.mem_get_info()

        print(
            "Free VRAM    :",
            round(free / (1024**3), 2),
            "GiB",
        )
        print(
            "Total VRAM   :",
            round(total / (1024**3), 2),
            "GiB",
        )

    print("\n--- NVIDIA-SMI ---")
    print(
        run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,"
                "power.draw,power.limit,temperature.gpu,"
                "clocks.sm,clocks.mem,utilization.gpu,"
                "utilization.memory",
                "--format=csv",
            ]
        )
    )

    print("\n--- Git ---")
    git_commit = run_command(
        ["git", "rev-parse", "--short", "HEAD"]
    )

    if git_commit.startswith("ERROR:"):
        print("No Git commit yet")
    else:
        print("Commit       :", git_commit)

    print("=" * 60)


if __name__ == "__main__":
    main()
