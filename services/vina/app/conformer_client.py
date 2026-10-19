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

"""Parent-side client for the out-of-process GPU conformer worker.

nvMolKit and CUDA never load into the Vina service. GPU conformer generation
runs in ``conformer_worker/worker.py``, exec'd with a dedicated interpreter from
its own environment (``/opt/aevom-conformer-worker``) and spoken to over pipes
using the frames defined in ``conformer_worker/framing.py``.

One worker per service process is started lazily on the first GPU request and
reused. Requests are serialized. Every exchange runs against a single deadline;
any timeout, EOF, malformed frame or mismatched reply kills and reaps the worker
and raises :class:`ConformerWorkerError`, and the next request starts a fresh
one. Callers are expected to fall back to RDKit on CPU.

Molecules cross the boundary as molblocks. Returned conformers are checked
against the molecule that was sent (atoms, charges, bonds, specified stereo,
finite coordinates and energies) and rebuilt on a copy of it, so atom order and
properties are the parent's own.
"""

from __future__ import annotations

import atexit
import importlib.util
import logging
import math
import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from rdkit import Chem
from rdkit.Chem import rdCIPLabeler
from rdkit.Geometry import Point3D

logger = logging.getLogger(__name__)

_WORKER_DIR = Path(__file__).resolve().parent.parent / "conformer_worker"


def _load_framing():
    spec = importlib.util.spec_from_file_location("aevom_conformer_framing", _WORKER_DIR / "framing.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


framing = _load_framing()

_SCRUBBED_ENV = {"LD_PRELOAD", "LD_LIBRARY_PATH"}


class ConformerWorkerError(Exception):
    """GPU conformer generation could not be completed by the worker."""


@dataclass(frozen=True)
class WorkerConfig:
    python: str
    script: str
    startup_timeout_s: float
    request_timeout_s: float
    max_payload_bytes: int
    restart_backoff_s: float
    ld_library_path: str
    frame_timeout_s: float

    @classmethod
    def from_env(cls) -> "WorkerConfig":
        env = os.environ
        return cls(
            python=env.get("CONFORMER_WORKER_PYTHON", "/opt/aevom-conformer-worker/bin/python"),
            script=env.get("CONFORMER_WORKER_SCRIPT", str(_WORKER_DIR / "worker.py")),
            startup_timeout_s=float(env.get("CONFORMER_WORKER_STARTUP_TIMEOUT_S", "120")),
            request_timeout_s=float(env.get("CONFORMER_WORKER_REQUEST_TIMEOUT_S", "300")),
            max_payload_bytes=int(env.get("CONFORMER_WORKER_MAX_PAYLOAD_BYTES", str(64 * 1024 * 1024))),
            restart_backoff_s=float(env.get("CONFORMER_WORKER_RESTART_BACKOFF_S", "60")),
            ld_library_path=env.get("CONFORMER_WORKER_LD_LIBRARY_PATH", ""),
            frame_timeout_s=float(env.get("CONFORMER_WORKER_FRAME_TIMEOUT_S", "30")),
        )


@dataclass
class WorkerResult:
    """Validated conformers as (energy, single-conformer Mol), in worker order."""

    conformers: list[tuple[float, Chem.Mol]]
    worker_pid: int
    request_id: str
    num_embedded: int
    truncated: bool = False
    rejected: list[str] = field(default_factory=list)


def _stereo_labels(mol: Chem.Mol) -> tuple[dict[int, str], dict[int, str]]:
    rdCIPLabeler.AssignCIPLabels(mol)
    atoms = {a.GetIdx(): a.GetProp("_CIPCode") for a in mol.GetAtoms() if a.HasProp("_CIPCode")}
    bonds = {b.GetIdx(): b.GetProp("_CIPCode") for b in mol.GetBonds() if b.HasProp("_CIPCode")}
    return atoms, bonds


class _Template:
    """What every returned conformer must agree with."""

    def __init__(self, mol: Chem.Mol):
        self.mol = mol
        self.atoms = [(a.GetAtomicNum(), a.GetFormalCharge(), a.GetIsotope(), a.GetTotalNumHs()) for a in mol.GetAtoms()]
        self.bonds = {
            (min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx())): b.GetBondType()
            for b in mol.GetBonds()
        }
        try:
            self.atom_stereo, self.bond_stereo = _stereo_labels(Chem.Mol(mol))
        except RuntimeError as exc:
            logger.warning(f"CIP labelling failed ({exc}); returned conformers will not be stereo-checked")
            self.atom_stereo, self.bond_stereo = {}, {}

    def check(self, returned: Chem.Mol) -> Optional[str]:
        if returned.GetNumAtoms() != len(self.atoms):
            return f"atom count {returned.GetNumAtoms()} != {len(self.atoms)}"
        for idx, atom in enumerate(returned.GetAtoms()):
            got = (atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetIsotope(), atom.GetTotalNumHs())
            if got != self.atoms[idx]:
                return f"atom {idx} is {got}, expected {self.atoms[idx]}"
        bonds = {
            (min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx())): b.GetBondType()
            for b in returned.GetBonds()
        }
        if bonds != self.bonds:
            return "bond table differs"
        if returned.GetNumConformers() != 1:
            return f"{returned.GetNumConformers()} conformers in one molblock"
        positions = returned.GetConformer().GetPositions()
        if not all(math.isfinite(v) for v in positions.flat):
            return "non-finite coordinates"
        if self.atom_stereo or self.bond_stereo:
            Chem.AssignStereochemistryFrom3D(returned)
            atom_stereo, bond_stereo = _stereo_labels(returned)
            for idx, label in self.atom_stereo.items():
                if atom_stereo.get(idx) != label:
                    return f"atom {idx} stereo {atom_stereo.get(idx)} != {label}"
            for idx, label in self.bond_stereo.items():
                if bond_stereo.get(idx) != label:
                    return f"bond {idx} stereo {bond_stereo.get(idx)} != {label}"
        return None

    def rebuild(self, returned: Chem.Mol) -> Chem.Mol:
        out = Chem.Mol(self.mol)
        out.RemoveAllConformers()
        source = returned.GetConformer()
        conf = Chem.Conformer(out.GetNumAtoms())
        for idx in range(out.GetNumAtoms()):
            p = source.GetAtomPosition(idx)
            conf.SetAtomPosition(idx, Point3D(p.x, p.y, p.z))
        conf.Set3D(True)
        out.AddConformer(conf, assignId=True)
        return out


class ConformerWorkerClient:
    def __init__(self, config: WorkerConfig):
        self.config = config
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._ready: Optional[dict[str, Any]] = None
        self._owner_pid = os.getpid()
        self._unavailable_until = 0.0
        self._last_failure: Optional[str] = None
        self._closed = False

    # -- public -----------------------------------------------------------

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc is not None and self._proc.poll() is None else None

    @property
    def ready_info(self) -> Optional[dict[str, Any]]:
        return self._ready

    def ensure_ready(self) -> dict[str, Any]:
        """Start the worker if needed and return its readiness message."""
        with self._acquire(self.config.startup_timeout_s):
            self._ensure_started()
            return dict(self._ready)

    def generate(self, mol: Chem.Mol, num_conformers: int, max_mmff_iters: int) -> WorkerResult:
        template = _Template(mol)
        request_id = uuid.uuid4().hex
        request = {
            "type": "generate",
            "id": request_id,
            "molblock": Chem.MolToMolBlock(mol),
            "num_conformers": int(num_conformers),
            "max_mmff_iters": int(max_mmff_iters),
        }
        with self._acquire(self.config.request_timeout_s):
            self._ensure_started()
            pid = self._proc.pid
            reply = self._exchange(request, self.config.request_timeout_s)
        if reply.get("type") != "result":
            raise ConformerWorkerError(f"worker {pid} returned {reply.get('type')!r}: {reply.get('reason')}")
        return self._validate(reply, template, pid, request_id, int(num_conformers))

    def inventory(self) -> dict[str, Any]:
        with self._acquire(self.config.request_timeout_s):
            self._ensure_started()
            return self._exchange({"type": "inventory", "id": uuid.uuid4().hex}, self.config.request_timeout_s)

    def shutdown(self) -> None:
        """Ask the worker to exit, then make sure it has, and reap it."""
        acquired = self._lock.acquire(timeout=5)
        try:
            self._closed = True
            proc = self._proc
            if proc is None or os.getpid() != self._owner_pid:
                return
            if proc.poll() is None:
                try:
                    self._exchange({"type": "shutdown", "id": uuid.uuid4().hex}, 5, kill_on_error=False)
                except ConformerWorkerError:
                    pass
            self._stop("service shutdown", graceful=True)
        finally:
            if acquired:
                self._lock.release()

    # -- internals --------------------------------------------------------

    def _acquire(self, timeout: float):
        client = self

        class _Held:
            def __enter__(self_inner):
                if not client._lock.acquire(timeout=timeout):
                    raise ConformerWorkerError(f"worker busy for more than {timeout:.0f}s")
                return self_inner

            def __exit__(self_inner, *exc):
                client._lock.release()

        return _Held()

    def _ensure_started(self) -> None:
        if os.getpid() != self._owner_pid:
            # Forked after the worker started: the pipes belong to the parent.
            self._proc, self._ready, self._owner_pid = None, None, os.getpid()
        if self._closed:
            raise ConformerWorkerError("client has been shut down")
        if self._proc is not None:
            if self._proc.poll() is None:
                return
            self._stop(f"worker exited with code {self._proc.returncode}", graceful=False)

        now = time.monotonic()
        if now < self._unavailable_until:
            raise ConformerWorkerError(
                f"worker unavailable ({self._last_failure}); next attempt in {self._unavailable_until - now:.0f}s"
            )
        try:
            self._start()
        except ConformerWorkerError as exc:
            self._last_failure = str(exc)
            self._unavailable_until = time.monotonic() + self.config.restart_backoff_s
            raise

    def _worker_env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON") and k not in _SCRUBBED_ENV}
        if self.config.ld_library_path:
            env["LD_LIBRARY_PATH"] = self.config.ld_library_path
        env["AEVOM_CONFORMER_PARENT_PID"] = str(os.getpid())
        env["CONFORMER_WORKER_MAX_PAYLOAD_BYTES"] = str(self.config.max_payload_bytes)
        env["CONFORMER_WORKER_FRAME_TIMEOUT_S"] = str(self.config.frame_timeout_s)
        return env

    def _start(self) -> None:
        python, script = self.config.python, self.config.script
        if not os.access(python, os.X_OK):
            raise ConformerWorkerError(f"worker interpreter {python} not found or not executable")
        if not os.path.isfile(script):
            raise ConformerWorkerError(f"worker script {script} not found")

        cmd = [python, "-I", script]
        spawned: dict[str, Any] = {}
        started = threading.Event()

        # PR_SET_PDEATHSIG fires when the *thread* that forked the worker exits,
        # so the fork happens on a thread that lives exactly as long as the
        # worker and reaps it.
        def _spawn_and_reap() -> None:
            try:
                spawned["proc"] = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=None,
                    close_fds=True,
                    start_new_session=True,
                    cwd="/",
                    env=self._worker_env(),
                )
            except OSError as exc:
                spawned["error"] = exc
                return
            finally:
                started.set()
            spawned["proc"].wait()

        threading.Thread(target=_spawn_and_reap, name="conformer-worker-reaper", daemon=True).start()
        started.wait()
        if "error" in spawned:
            raise ConformerWorkerError(f"could not exec worker: {spawned['error']}")

        proc = spawned["proc"]
        os.set_blocking(proc.stdin.fileno(), False)
        os.set_blocking(proc.stdout.fileno(), False)
        self._proc = proc
        logger.info(f"Started conformer worker pid={proc.pid}: {' '.join(cmd)}")

        try:
            ready = framing.read_frame(
                proc.stdout.fileno(),
                max_bytes=self.config.max_payload_bytes,
                deadline=framing.deadline_after(self.config.startup_timeout_s),
            )
        except framing.FrameError as exc:
            self._stop(f"no readiness handshake: {exc}", graceful=False)
            raise ConformerWorkerError(f"worker pid={proc.pid} sent no readiness handshake: {exc}") from exc
        if ready is None:
            self._stop("exited before readiness", graceful=False)
            raise ConformerWorkerError(f"worker pid={proc.pid} exited before readiness (code {proc.poll()})")
        if ready.get("protocol") != framing.PROTOCOL_VERSION or ready.get("type") != "ready":
            self._stop("bad handshake", graceful=False)
            raise ConformerWorkerError(f"worker pid={proc.pid} sent an invalid handshake: {ready!r:.200}")
        if not ready.get("ok"):
            self._stop("not ready", graceful=False)
            raise ConformerWorkerError(f"worker pid={proc.pid} not ready: {ready.get('reason')}")
        self._ready = ready
        self._last_failure = None
        logger.info(f"Conformer worker pid={proc.pid} ready: {ready.get('versions')}")

    def _exchange(self, request: dict[str, Any], timeout: float, kill_on_error: bool = True) -> dict[str, Any]:
        """Send *request* and return the matching reply, all within *timeout*."""
        proc = self._proc
        request = {"protocol": framing.PROTOCOL_VERSION, **request}
        deadline = framing.deadline_after(timeout)
        try:
            framing.write_frame(
                proc.stdin.fileno(), request, max_bytes=self.config.max_payload_bytes, deadline=deadline
            )
            reply = framing.read_frame(
                proc.stdout.fileno(), max_bytes=self.config.max_payload_bytes, deadline=deadline
            )
            if reply is None:
                raise framing.FrameEOF("worker closed stdout")
            if reply.get("protocol") != framing.PROTOCOL_VERSION:
                raise framing.FrameError(f"reply has protocol {reply.get('protocol')!r}")
            if reply.get("id") != request["id"]:
                raise framing.FrameError(f"reply id {reply.get('id')!r} does not match request {request['id']}")
        except framing.FrameError as exc:
            reason = f"{type(exc).__name__}: {exc}"
            if kill_on_error:
                self._stop(reason, graceful=False)
            raise ConformerWorkerError(f"worker pid={proc.pid} {request['type']} failed: {reason}") from exc
        if reply.get("type") == "error" and reply.get("fatal"):
            self._stop(f"fatal worker error: {reply.get('reason')}", graceful=True)
        return reply

    def _stop(self, reason: str, graceful: bool) -> None:
        proc, self._proc, self._ready = self._proc, None, None
        if proc is None:
            return
        if proc.poll() is None:
            log = logger.info if graceful else logger.warning
            log(f"Stopping conformer worker pid={proc.pid}: {reason}")
            if graceful:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.terminate()
            if proc.poll() is None:
                proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.error(f"Conformer worker pid={proc.pid} did not exit after SIGKILL")
        for pipe in (proc.stdin, proc.stdout):
            try:
                pipe.close()
            except OSError:
                pass

    def _validate(
        self,
        reply: dict[str, Any],
        template: _Template,
        pid: int,
        request_id: str,
        requested: int,
    ) -> WorkerResult:
        entries = reply.get("conformers")
        num_embedded = reply.get("num_embedded")
        if not isinstance(entries, list) or not isinstance(num_embedded, int):
            raise ConformerWorkerError(f"worker {pid} returned a malformed result")
        if len(entries) > min(num_embedded, requested):
            raise ConformerWorkerError(
                f"worker {pid} returned {len(entries)} conformers for {requested} requested / {num_embedded} embedded"
            )

        result = WorkerResult(
            conformers=[],
            worker_pid=pid,
            request_id=request_id,
            num_embedded=num_embedded,
            truncated=bool(reply.get("truncated")),
        )
        seen: set[int] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ConformerWorkerError(f"worker {pid} returned a malformed conformer entry")
            conf_index, energy, block = entry.get("conf_index"), entry.get("energy"), entry.get("molblock")
            if not isinstance(conf_index, int) or conf_index in seen:
                raise ConformerWorkerError(f"worker {pid} returned a bad or duplicate conf_index {conf_index!r}")
            seen.add(conf_index)
            if isinstance(energy, bool) or not isinstance(energy, (int, float)) or not math.isfinite(energy):
                result.rejected.append(f"conformer {conf_index}: energy {energy!r}")
                continue
            if not isinstance(block, str):
                result.rejected.append(f"conformer {conf_index}: no molblock")
                continue
            returned = Chem.MolFromMolBlock(block, removeHs=False)
            if returned is None:
                result.rejected.append(f"conformer {conf_index}: unparseable molblock")
                continue
            problem = template.check(returned)
            if problem:
                result.rejected.append(f"conformer {conf_index}: {problem}")
                continue
            result.conformers.append((float(energy), template.rebuild(returned)))

        if result.rejected:
            logger.warning(
                f"Rejected {len(result.rejected)} conformer(s) from worker {pid}: {'; '.join(result.rejected[:5])}"
            )
        return result


_client: Optional[ConformerWorkerClient] = None
_client_lock = threading.Lock()


def get_client() -> ConformerWorkerClient:
    global _client
    with _client_lock:
        if _client is None:
            _client = ConformerWorkerClient(WorkerConfig.from_env())
        return _client


def peek_client() -> Optional[ConformerWorkerClient]:
    return _client


def reset_client() -> None:
    """Shut down the current worker and forget the client (tests, reconfiguration)."""
    global _client
    with _client_lock:
        client, _client = _client, None
    if client is not None:
        client.shutdown()


def shutdown() -> None:
    client = _client
    if client is not None:
        client.shutdown()


atexit.register(shutdown)
