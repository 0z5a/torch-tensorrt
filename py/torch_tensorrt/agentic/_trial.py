# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Worker process: ``python -m torch_tensorrt.agentic._trial REQUEST.json``.

Each job runs in a fresh process, so candidates cannot leak GPU memory,
process-global registrations or crashes into each other or into the session.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.utils._pytree as pytree


def write_json(path: Path, value: Any) -> None:
    """Write ``value`` atomically, so a reader never sees a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def save_reference(tree: Any, path: Path) -> dict[str, Any]:
    """Save reference outputs on the CPU; return a description of them."""
    leaves, _ = pytree.tree_flatten(tree)
    for leaf in leaves:
        if not isinstance(leaf, (torch.Tensor, int, float, bool)):
            raise TypeError(
                "Reference outputs must be tensors, numbers and containers; "
                f"got {type(leaf).__name__}"
            )
    on_cpu = pytree.tree_map(
        lambda leaf: leaf.detach().cpu() if isinstance(leaf, torch.Tensor) else leaf,
        tree,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(on_cpu, path)
    return {
        "leaves": [
            (
                {"shape": list(leaf.shape), "dtype": str(leaf.dtype)}
                if isinstance(leaf, torch.Tensor)
                else {"value": leaf}
            )
            for leaf in leaves
        ]
    }


def load_reference(path: Path) -> Any:
    # weights_only: the file holds tensors and containers, never code.
    return torch.load(path, weights_only=True)


def environment() -> dict[str, Any]:
    """The software and hardware that every measurement in a session depends on."""
    import platform
    import shutil
    import subprocess

    import tensorrt

    import torch_tensorrt

    driver = None
    binary = shutil.which("nvidia-smi")
    if binary is not None:
        try:
            driver = subprocess.run(
                [
                    binary,
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                    "-i",
                    "0",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            driver = None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tensorrt": tensorrt.__version__,
        "torch_tensorrt": torch_tensorrt.__version__,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "driver": driver,
    }


def describe_case(case: Any) -> dict[str, Any]:
    from dataclasses import asdict

    from torch_tensorrt.agentic._tools import apply, catalog

    tools = catalog(case.tools())
    return {
        "name": case.name,
        "tools": [tool.to_json() for tool in tools.values()],
        "baseline": apply(case.baseline(), {}, tools),
        "timing": asdict(case.timing()),
        "accuracy_defaults": {
            name: list(bound) for name, bound in case.accuracy_defaults().items()
        },
    }


def run(request: dict[str, Any]) -> dict[str, Any]:
    from torch_tensorrt.agentic._evaluator import evaluate
    from torch_tensorrt.agentic._spec import Timing, Workload
    from torch_tensorrt.agentic._tools import catalog, resolve

    case = Workload(**request["workload"]).load()
    reference_path = Path(request["reference_path"])
    if request["kind"] in ("describe", "reference"):
        result = {
            "status": "ok",
            "case": describe_case(case),
            "environment": environment(),
        }
        if request["kind"] == "reference":
            result["reference"] = save_reference(case.reference(), reference_path)
        return result
    tools = catalog(case.tools())
    values = resolve(request["recipe"], tools)
    export_dir = request.get("export_dir")
    return evaluate(
        case,
        values,
        load_reference(reference_path),
        Timing(**request["timing"]),
        profile=bool(request.get("profile", True)),
        export_dir=Path(export_dir) if export_dir else None,
    )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(
            "usage: python -m torch_tensorrt.agentic._trial REQUEST.json",
            file=sys.stderr,
        )
        return 2
    request = json.loads(Path(argv[1]).read_text())
    result_path = Path(request["result_path"])
    try:
        result = run(request)
    except Exception as error:  # noqa: BLE001 - report, do not hide, setup failures
        import traceback

        traceback.print_exc()
        result = {
            "status": "failed",
            "stage": "setup",
            "error": "".join(
                traceback.format_exception_only(type(error), error)
            ).strip(),
        }
    write_json(result_path, result)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
