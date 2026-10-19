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

"""GPU conformer worker: runs nvMolKit in its own interpreter.

The Vina service never imports nvMolKit or loads CUDA. It execs this script
with the dedicated worker interpreter (``/opt/aevom-conformer-worker/bin/python
-I``) and exchanges length-prefixed JSON frames over stdin/stdout, as defined in
``framing.py`` next to this file.

This file must import nothing from the Vina application, and nothing besides
the standard library, RDKit and nvMolKit. Protocol output goes only to a
private duplicate of the original stdout; fd 1 is pointed at stderr before any
native library loads, so stray prints cannot corrupt the protocol.

Messages (all carry ``protocol`` and ``id``):

- ``generate`` {molblock, num_conformers, max_mmff_iters}
  -> ``result`` {conformers: [{conf_index, energy, molblock}], num_embedded,
  truncated, elapsed_ms, backend}
- ``inventory`` -> ``inventory`` {pid, modules, shared_objects, served,
  gpu_generations}
- ``shutdown`` -> ``bye``, then exit
"""

import collections
import ctypes
import importlib.util
import json
import logging
import math
import os
import signal
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))

MAX_CONFORMERS = 1000
MAX_MMFF_ITERS = 100_000
_FRAME_OVERHEAD = 4096
_PR_SET_PDEATHSIG = 1


def _load_framing():
    spec = importlib.util.spec_from_file_location("aevom_conformer_framing", os.path.join(_HERE, "framing.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


framing = _load_framing()

logger = logging.getLogger("aevom.conformer_worker")


class WorkerFatal(Exception):
    """The worker must exit after answering: GPU state may be unusable."""


def _isolate_stdout() -> int:
    proto_fd = os.dup(1)
    os.dup2(2, 1)
    os.set_inheritable(proto_fd, False)
    return proto_fd


def _die_with_parent() -> None:
    expected = os.environ.get("AEVOM_CONFORMER_PARENT_PID")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    except (OSError, AttributeError) as exc:
        logger.warning("could not set PR_SET_PDEATHSIG: %s", exc)
    if expected and os.getppid() != int(expected):
        logger.error("parent %s already gone (ppid=%d); exiting", expected, os.getppid())
        sys.exit(1)


def _shared_objects() -> list[str]:
    paths = set()
    with open("/proc/self/maps") as maps:
        for line in maps:
            parts = line.split(None, 5)
            if len(parts) == 6:
                path = parts[5].strip()
                if ".so" in os.path.basename(path):
                    paths.add(path)
    return sorted(paths)


class Worker:
    def __init__(self, proto_fd: int):
        self.out_fd = proto_fd
        self.in_fd = 0
        self.max_bytes = int(os.environ.get("CONFORMER_WORKER_MAX_PAYLOAD_BYTES", str(64 * 1024 * 1024)))
        self.frame_timeout = float(os.environ.get("CONFORMER_WORKER_FRAME_TIMEOUT_S", "30"))
        self.served = collections.deque(maxlen=256)
        self.gpu_generations = 0
        os.set_blocking(self.in_fd, False)
        os.set_blocking(self.out_fd, False)

    def send(self, message: dict) -> None:
        message.setdefault("protocol", framing.PROTOCOL_VERSION)
        framing.write_frame(
            self.out_fd,
            message,
            max_bytes=self.max_bytes,
            deadline=framing.deadline_after(self.frame_timeout),
        )

    def load_backend(self) -> dict:
        from rdkit import Chem, rdBase
        from rdkit.Chem import rdDistGeom
        from nvmolkit.embedMolecules import EmbedMolecules
        from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs

        self.Chem = Chem
        self.rdDistGeom = rdDistGeom
        self.embed = EmbedMolecules
        self.mmff = MMFFOptimizeMoleculesConfs

        warm = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        _embedded, energies = self._run_nvmolkit(warm, 1, 50)
        if warm.GetNumConformers() == 0:
            raise RuntimeError("nvMolKit warm-up embedded zero conformers (CUDA unavailable or arch mismatch?)")
        if not all(math.isfinite(e) for e in energies):
            raise RuntimeError("nvMolKit warm-up returned non-finite MMFF energies")

        from importlib import metadata

        def version(dist: str) -> str:
            try:
                return metadata.version(dist)
            except metadata.PackageNotFoundError:
                return "unknown"

        return {
            "python": sys.version.split()[0],
            "rdkit": rdBase.rdkitVersion,
            "nvmolkit": version("nvMolKit"),
            "torch": version("torch"),
        }

    def _run_nvmolkit(self, mol, num_conformers: int, max_iters: int) -> tuple[int, list[float]]:
        params = self.rdDistGeom.ETKDGv3()
        params.useRandomCoords = True  # required by nvMolKit
        self.embed(
            molecules=[mol],
            params=params,
            confsPerMolecule=num_conformers,
            maxIterations=-1,  # automatic
        )
        n_embedded = mol.GetNumConformers()
        if n_embedded == 0:
            return 0, []
        energy_lists = self.mmff([mol], maxIters=max_iters)
        energies = list(energy_lists[0]) if energy_lists else [0.0] * n_embedded
        if len(energies) < n_embedded:
            energies.extend([0.0] * (n_embedded - len(energies)))
        return n_embedded, [float(e) for e in energies[:n_embedded]]

    def generate(self, request: dict) -> dict:
        molblock = request.get("molblock")
        if not isinstance(molblock, str) or not molblock.strip():
            raise ValueError("molblock must be a non-empty string")
        num = request.get("num_conformers")
        iters = request.get("max_mmff_iters")
        if not isinstance(num, int) or not 1 <= num <= MAX_CONFORMERS:
            raise ValueError(f"num_conformers must be an integer in [1, {MAX_CONFORMERS}]")
        if not isinstance(iters, int) or not 1 <= iters <= MAX_MMFF_ITERS:
            raise ValueError(f"max_mmff_iters must be an integer in [1, {MAX_MMFF_ITERS}]")

        mol = self.Chem.MolFromMolBlock(molblock, removeHs=False)
        if mol is None:
            raise ValueError("molblock could not be parsed")
        # Stereo has been perceived from the input coordinates by now. nvMolKit
        # appends to existing conformers, so without this the input pose would
        # come back as one of the "generated" conformers.
        mol.RemoveAllConformers()

        start = time.monotonic()
        try:
            n_embedded, energies = self._run_nvmolkit(mol, num, iters)
        except Exception as exc:
            raise WorkerFatal(f"nvMolKit raised {type(exc).__name__}: {exc}") from exc
        nvmolkit_completed = n_embedded > 0
        if nvmolkit_completed:
            self.gpu_generations += 1
        self.served.append(
            {
                "request_id": request["id"],
                "num_embedded": n_embedded,
                "nvmolkit_calls_completed": nvmolkit_completed,
            }
        )

        conformers = []
        budget = self.max_bytes - _FRAME_OVERHEAD
        used = 0
        truncated = False
        for idx in range(n_embedded):
            energy = energies[idx]
            block = self.Chem.MolToMolBlock(mol, confId=mol.GetConformer(idx).GetId())
            used += len(json.dumps(block)) + 64
            if used > budget:
                truncated = True
                logger.warning("response truncated to %d of %d conformers by payload limit", idx, n_embedded)
                break
            conformers.append(
                {
                    "conf_index": idx,
                    "energy": energy if math.isfinite(energy) else None,
                    "molblock": block,
                }
            )
        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.info("request %s: nvMolKit embedded %d/%d in %dms", request["id"], n_embedded, num, elapsed_ms)
        return {
            "type": "result",
            "conformers": conformers,
            "num_embedded": n_embedded,
            "truncated": truncated,
            "elapsed_ms": elapsed_ms,
            "backend": "nvmolkit",
        }

    def inventory(self) -> dict:
        return {
            "type": "inventory",
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "executable": sys.executable,
            "modules": sorted(sys.modules),
            "shared_objects": _shared_objects(),
            "served": list(self.served),
            "gpu_generations": self.gpu_generations,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }

    def serve(self) -> int:
        while True:
            try:
                request = framing.read_frame(
                    self.in_fd,
                    max_bytes=self.max_bytes,
                    deadline=None,
                    frame_timeout=self.frame_timeout,
                )
            except framing.FrameError as exc:
                logger.error("unreadable request, exiting: %s", exc)
                return 2
            if request is None:
                logger.info("stdin closed; exiting")
                return 0

            request_id = request.get("id")
            if request.get("protocol") != framing.PROTOCOL_VERSION or not isinstance(request_id, str):
                self.send(
                    {
                        "type": "error",
                        "id": request_id if isinstance(request_id, str) else None,
                        "reason": f"unsupported protocol {request.get('protocol')!r} or missing id",
                    }
                )
                continue

            kind = request.get("type")
            fatal = False
            try:
                if kind == "generate":
                    response = self.generate(request)
                elif kind == "inventory":
                    response = self.inventory()
                elif kind == "shutdown":
                    self.send({"type": "bye", "id": request_id})
                    return 0
                else:
                    raise ValueError(f"unknown request type {kind!r}")
            except WorkerFatal as exc:
                logger.error("%s; exiting after reply", exc)
                response = {"type": "error", "reason": str(exc), "fatal": True}
                fatal = True
            except Exception as exc:
                response = {"type": "error", "reason": f"{type(exc).__name__}: {exc}", "fatal": False}

            response["id"] = request_id
            try:
                self.send(response)
            except framing.FrameError as exc:
                logger.error("could not deliver response to %s, exiting: %s", request_id, exc)
                return 2
            if fatal:
                return 3


def main() -> int:
    proto_fd = _isolate_stdout()
    logging.basicConfig(
        stream=sys.stderr,
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format=f"%(asctime)s - conformer-worker[{os.getpid()}] - %(levelname)s - %(message)s",
    )
    _die_with_parent()

    worker = Worker(proto_fd)
    try:
        versions = worker.load_backend()
    except BaseException as exc:
        reason = f"{type(exc).__name__}: {exc}"
        logger.error("not ready: %s", reason)
        try:
            worker.send({"type": "ready", "id": None, "ok": False, "reason": reason, "pid": os.getpid()})
        except framing.FrameError:
            pass
        return 1

    worker.send(
        {
            "type": "ready",
            "id": None,
            "ok": True,
            "pid": os.getpid(),
            "versions": versions,
            "executable": sys.executable,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
    )
    logger.info("ready: %s", versions)
    return worker.serve()


if __name__ == "__main__":
    sys.exit(main())
