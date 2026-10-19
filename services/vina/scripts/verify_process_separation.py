#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runtime evidence that nvMolKit and the Vina preparation stack run in separate processes.

Run inside a GPU image from the Vina service directory (so ``import app`` is the
service), with the same interpreter the service runs under:

    cd /app && python scripts/verify_process_separation.py --require-gpu --report /tmp/report.json

It drives the real service code in this (parent) process:

1. ``run_auto_dock`` on the 3ert tamoxifen fixture with ``rot_bond_threshold=0``
   in search mode: conformers from the nvMolKit worker, rigid preparation with
   Meeko here, then the docking binary. The request's ``_diagnostics`` must name
   the worker, and the worker's own inventory must show that request id served
   with completed nvMolKit calls and ``gpu_generations`` up by exactly one.
2. Open Babel on purpose, since ``prepare_ligand`` normally returns after
   Meeko: ``_prepare_ligand_openbabel`` on a worker-produced conformer, and
   ``prepare_receptor`` on the raw PDB, whose first branch is Open Babel.

Only then does it capture both processes' PIDs, imported modules and mapped
shared objects, and check them. A second, throwaway worker is started with
``LD_DEBUG=libs`` to record where the dynamic loader found each library.

Exit status is non-zero if any separation check fails, and, with
``--require-gpu``, if the worker is not ready or any step fell back to CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("VINA_GPU", "true")
SERVICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVICE_DIR))

from rdkit import Chem  # noqa: E402

from app import conformer, conformer_client, docking, pipeline, preparation  # noqa: E402

DATA = SERVICE_DIR / "tests" / "data"

PARENT_FORBIDDEN_MODULES = ("nvmolkit", "torch")
PARENT_FORBIDDEN_LIBS = re.compile(
    r"^lib(cuda|cudart|cublas|cublasLt|cudnn\w*|nvinfer\w*|nvrtc\w*|nccl|torch\w*|c10\w*|"
    r"nvToolsExt|cufft|curand|cusparse|cusolver|nvJitLink|nvmolkit\w*)[.-]"
)
WORKER_FORBIDDEN_MODULES = ("meeko", "openbabel", "plip")
WORKER_FORBIDDEN_LIBS = re.compile(r"^lib(openbabel|inchi|plip)\w*[.-]")
SYSTEM_LIB_DIRS = (
    "/lib/x86_64-linux-gnu/",
    "/usr/lib/x86_64-linux-gnu/",
    "/lib/aarch64-linux-gnu/",
    "/usr/lib/aarch64-linux-gnu/",
    "/lib64/",
    "/usr/lib64/",
    # The aarch64 dynamic loader sits directly in /lib, not in a lib dir above.
    "/lib/ld-linux-aarch64.so.1",
)
DRIVER_LIBS = re.compile(r"^lib(cuda|nvidia-\w+|nvcuvid|nvoptix)\.so")


def _shared_objects(pid: str | int = "self") -> list[str]:
    paths = set()
    with open(f"/proc/{pid}/maps") as maps:
        for line in maps:
            parts = line.split(None, 5)
            if len(parts) == 6 and ".so" in os.path.basename(parts[5].strip()):
                paths.add(parts[5].strip())
    return sorted(paths)


def _top_modules(modules) -> list[str]:
    return sorted({m.split(".")[0] for m in modules})


class Checks:
    def __init__(self):
        self.results: list[dict] = []

    def check(self, name: str, ok: bool, detail="") -> bool:
        self.results.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{': ' + str(detail) if detail else ''}", flush=True)
        return bool(ok)

    @property
    def failed(self) -> list[str]:
        return [r["check"] for r in self.results if not r["ok"]]


def _conda_owner_map(prefix: Path) -> dict[str, str]:
    owners = {}
    for record in (prefix / "conda-meta").glob("*.json"):
        data = json.loads(record.read_text())
        for f in data.get("files", []):
            owners[str(prefix / f)] = data["name"]
    return owners


def _ld_debug_resolution(worker_python: str) -> dict:
    """Start a throwaway worker under LD_DEBUG=libs and record library lookups."""
    out_dir = Path(tempfile.mkdtemp(prefix="ld-debug-"))
    saved = {k: os.environ.get(k) for k in ("LD_DEBUG", "LD_DEBUG_OUTPUT")}
    os.environ["LD_DEBUG"] = "libs"
    os.environ["LD_DEBUG_OUTPUT"] = str(out_dir / "ld")
    client = conformer_client.ConformerWorkerClient(conformer_client.WorkerConfig.from_env())
    try:
        ready = client.ensure_ready()
        pid = ready["pid"]
        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
        client.generate(mol, 2, 50)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        client.shutdown()
    log = out_dir / f"ld.{pid}"
    initialised = []
    if log.exists():
        for line in log.read_text(errors="replace").splitlines():
            m = re.search(r"calling init: (\S+)", line)
            if m:
                initialised.append(m.group(1))
    shutil.rmtree(out_dir, ignore_errors=True)
    return {"worker_pid": pid, "initialised": initialised}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--require-gpu", action="store_true", help="fail unless nvMolKit ran on the GPU worker")
    parser.add_argument("--report", default="process_separation_report.json")
    args = parser.parse_args()

    checks = Checks()
    report: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "parent": {}, "worker": {}, "steps": {}}
    client = conformer_client.get_client()
    worker_python = client.config.python
    worker_prefix = Path(worker_python).resolve().parent.parent

    report["parent"]["pid"] = os.getpid()
    report["parent"]["executable"] = sys.executable
    report["parent"]["env"] = {
        k: os.environ.get(k) for k in ("LD_LIBRARY_PATH", "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "VINA_GPU")
    }
    for var in ("LD_LIBRARY_PATH", "PYTHONPATH"):
        checks.check(
            f"parent {var} does not include the worker prefix",
            str(worker_prefix) not in (os.environ.get(var) or ""),
            os.environ.get(var),
        )
    checks.check("docking runs on the GPU binary (VINA_GPU)", docking.VINA_GPU_ENABLED)

    # -- readiness --------------------------------------------------------
    try:
        ready = client.ensure_ready()
    except conformer_client.ConformerWorkerError as exc:
        ready = None
        checks.check("worker readiness handshake", False, str(exc))
    else:
        checks.check("worker readiness handshake", True, ready.get("versions"))
    report["worker"]["ready"] = ready
    if ready is None and args.require_gpu:
        _write(report, checks, args.report)
        return 1

    before = client.inventory()["gpu_generations"] if ready else None

    # -- step 1: ensemble docking through the worker ----------------------
    receptor_pdbqt, _n, err = preparation.prepare_receptor(
        (DATA / "pocket.pdb").read_text(), remove_water=True, remove_heteroatoms=True
    )
    checks.check("receptor prepared", err is None and bool(receptor_pdbqt), err)
    center = tuple(float(v) for v in (DATA / "pocket_center.txt").read_text().split())
    result = pipeline.run_auto_dock(
        receptor_pdb=receptor_pdbqt,
        ligand_sdf=(DATA / "ligand.sdf").read_text(),
        center=center,
        size=(20.0, 20.0, 20.0),
        exhaustiveness=8,
        num_conformers=8,
        rot_bond_threshold=0,
        score_mode="search",
    )
    diag = (result.get("_diagnostics") or {}).get("conformer") or {}
    report["steps"]["ensemble_dock"] = {
        "success": result["success"],
        "error": result["error"],
        "docking_mode": result["docking_mode"],
        "conformers_generated": result["conformers_generated"],
        "best_conformer_idx": result["best_conformer_idx"],
        "best_affinity": result["best_affinity"],
        "gpu_accelerated": result["gpu_accelerated"],
        "diagnostics": diag,
        "docking_binary": docking.UNIDOCK_BINARY if docking.VINA_GPU_ENABLED else docking.VINA_BINARY,
    }
    checks.check("ensemble dock succeeded", result["success"], result["error"])
    checks.check("ensemble path taken", result["docking_mode"] == "rigid_ensemble", result["docking_mode"])
    used_gpu = checks.check(
        "dock request's conformers came from the nvMolKit worker",
        diag.get("backend") == conformer.BACKEND_GPU_WORKER,
        f"backend={diag.get('backend')} reason={diag.get('fallback_reason')}",
    )

    if ready:
        inv = client.inventory()
        served = {s["request_id"]: s for s in inv["served"]}
        entry = served.get(diag.get("worker_request_id"))
        checks.check(
            "worker served that request id with completed nvMolKit calls",
            bool(entry and entry["nvmolkit_calls_completed"] and entry["num_embedded"] > 0),
            entry,
        )
        checks.check(
            "worker gpu_generations increased by exactly one for the dock",
            inv["gpu_generations"] == before + 1,
            f"{before} -> {inv['gpu_generations']}",
        )
        checks.check("dock request's worker pid is the live worker", diag.get("worker_pid") == inv["pid"])

    # -- step 2: Open Babel, deliberately ---------------------------------
    mol = Chem.MolFromMolBlock((DATA / "ligand.sdf").read_text(), removeHs=False)
    conf_mols, conf_report = conformer.generate_conformers_with_report(mol, 2, use_gpu=True)
    checks.check(
        "Open Babel input conformer came from the worker",
        conf_report.backend == conformer.BACKEND_GPU_WORKER,
        conf_report.backend,
    )
    ob_ligand = preparation._prepare_ligand_openbabel(conf_mols[0]) if conf_mols else None
    checks.check(
        "Open Babel ligand preparation produced a PDBQT",
        bool(ob_ligand) and "ROOT" in ob_ligand and "TORSDOF" in ob_ligand,
    )

    calls = {"openbabel": 0, "fallback": 0}
    real_ob, real_fb = preparation._prepare_receptor_openbabel, preparation._prepare_receptor_fallback

    def counted_ob(*a, **k):
        calls["openbabel"] += 1
        return real_ob(*a, **k)

    def counted_fb(*a, **k):
        calls["fallback"] += 1
        return real_fb(*a, **k)

    preparation._prepare_receptor_openbabel, preparation._prepare_receptor_fallback = counted_ob, counted_fb
    try:
        ob_receptor, _n, err = preparation.prepare_receptor(
            (DATA / "pocket.pdb").read_text(), remove_water=True, remove_heteroatoms=True
        )
    finally:
        preparation._prepare_receptor_openbabel, preparation._prepare_receptor_fallback = real_ob, real_fb
    checks.check(
        "Open Babel receptor preparation ran (not the fallback)",
        err is None and bool(ob_receptor) and calls == {"openbabel": 1, "fallback": 0},
        calls,
    )
    report["steps"]["openbabel"] = {"ligand_pdbqt_lines": len((ob_ligand or "").splitlines()), "receptor_calls": calls}

    # -- evidence -----------------------------------------------------------
    parent_modules = sorted(sys.modules)
    parent_libs = _shared_objects()
    report["parent"]["modules"] = parent_modules
    report["parent"]["shared_objects"] = parent_libs
    parent_top = _top_modules(parent_modules)

    checks.check("parent imported meeko", "meeko" in parent_top)
    checks.check("parent imported openbabel", "openbabel" in parent_top)
    checks.check(
        "parent mapped libopenbabel",
        any(os.path.basename(p).startswith("libopenbabel") for p in parent_libs),
    )
    bad_mods = [m for m in parent_top if m in PARENT_FORBIDDEN_MODULES]
    checks.check("parent did not import nvmolkit/torch", not bad_mods, bad_mods)
    bad_libs = [p for p in parent_libs if PARENT_FORBIDDEN_LIBS.match(os.path.basename(p))]
    checks.check("parent mapped no CUDA/cuDNN/TensorRT/torch/nvMolKit libraries", not bad_libs, bad_libs)
    checks.check(
        "parent mapped nothing from the worker prefix",
        not [p for p in parent_libs if p.startswith(str(worker_prefix))],
    )

    if ready:
        inv = client.inventory()
        report["worker"].update(
            {
                "pid": inv["pid"],
                "ppid": inv["ppid"],
                "executable": inv["executable"],
                "cuda_visible_devices": inv["cuda_visible_devices"],
                "gpu_generations": inv["gpu_generations"],
                "served": inv["served"],
                "modules": inv["modules"],
                "shared_objects": inv["shared_objects"],
            }
        )
        worker_top = _top_modules(inv["modules"])
        worker_libs = inv["shared_objects"]
        checks.check("worker is a different process", inv["pid"] != os.getpid(), f"parent={os.getpid()} worker={inv['pid']}")
        checks.check("worker's parent is this process", inv["ppid"] == os.getpid(), inv["ppid"])
        checks.check(
            "worker inherited the parent's CUDA_VISIBLE_DEVICES",
            inv["cuda_visible_devices"] == os.environ.get("CUDA_VISIBLE_DEVICES"),
            inv["cuda_visible_devices"],
        )
        checks.check("worker imported nvmolkit", "nvmolkit" in worker_top)
        checks.check("worker mapped libcuda.so (driver)", any(os.path.basename(p).startswith("libcuda.so") for p in worker_libs))
        checks.check(
            "worker mapped libcudart from the worker prefix",
            any(os.path.basename(p).startswith("libcudart") and p.startswith(str(worker_prefix)) for p in worker_libs),
        )
        bad = [m for m in worker_top if m in WORKER_FORBIDDEN_MODULES]
        checks.check("worker did not import meeko/openbabel/plip", not bad, bad)
        bad = [p for p in worker_libs if WORKER_FORBIDDEN_LIBS.match(os.path.basename(p))]
        checks.check("worker mapped no Open Babel/PLIP libraries", not bad, bad)
        outside = [
            p
            for p in worker_libs
            if not p.startswith(str(worker_prefix))
            and not (p.startswith(SYSTEM_LIB_DIRS) and (DRIVER_LIBS.match(os.path.basename(p)) or _is_glibc(p)))
        ]
        checks.check("worker libraries come only from its prefix, glibc and the driver", not outside, outside)

        info_dir = worker_prefix / "share" / "aevom-conformer-worker"
        try:
            build_info = json.loads((info_dir / "build-info.json").read_text())
            licenses = json.loads((info_dir / "licenses.json").read_text())["packages"]
        except (OSError, ValueError) as exc:
            build_info, licenses = None, []
            checks.check("worker build records present", False, str(exc))
        report["worker"]["build_info"] = build_info
        owners = _conda_owner_map(worker_prefix)
        loaded_pkgs = sorted({owners[p] for p in worker_libs if p in owners})
        flagged = {e["name"]: e["license"] for e in licenses if e["copyleft_flag"]}
        report["worker"]["loaded_conda_packages"] = loaded_pkgs
        report["worker"]["copyleft_flagged_packages"] = flagged
        report["worker"]["copyleft_flagged_and_loaded"] = {n: flagged[n] for n in loaded_pkgs if n in flagged}
        print(f"[INFO] worker copyleft-flagged packages loaded at runtime: {report['worker']['copyleft_flagged_and_loaded']}")

        report["worker"]["ld_debug"] = _ld_debug_resolution(worker_python)
        stray = [
            p
            for p in report["worker"]["ld_debug"]["initialised"]
            if not p.startswith(str(worker_prefix))
            and not (p.startswith(SYSTEM_LIB_DIRS) and (DRIVER_LIBS.match(os.path.basename(p)) or _is_glibc(p)))
        ]
        checks.check(
            "LD_DEBUG: worker initialised libraries only from its prefix, glibc and the driver",
            bool(report["worker"]["ld_debug"]["initialised"]) and not stray,
            stray or f"{len(report['worker']['ld_debug']['initialised'])} libraries",
        )

    conformer_client.shutdown()
    report["gpu_required"] = args.require_gpu
    report["gpu_used"] = used_gpu
    _write(report, checks, args.report)
    if checks.failed:
        print(f"FAILED: {checks.failed}", file=sys.stderr)
        return 1
    return 0


_GLIBC = re.compile(r"^(libc|libm|libdl|libpthread|librt|libutil|libresolv|libnss_\w+|ld-linux[\w-]*|libgcc_s)\.so")


def _is_glibc(path: str) -> bool:
    return bool(_GLIBC.match(os.path.basename(path)))


def _write(report: dict, checks: Checks, path: str) -> None:
    report["checks"] = checks.results
    report["passed"] = not checks.failed
    Path(path).write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"report written to {path}")


if __name__ == "__main__":
    sys.exit(main())
