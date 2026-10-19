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

"""Tests for out-of-process GPU conformer generation.

Three kinds of worker are used:

- the real ``conformer_worker/worker.py`` with a CPU stand-in for nvMolKit
  (``worker_stubs/``), for the round trip, ordering, concurrency, lifecycle
  and import-isolation tests;
- ``fake_conformer_worker.py``, a protocol-speaking worker that misbehaves on
  demand, for every timeout, EOF, malformed-reply and validation path;
- the real worker on a real GPU, in the ``gpu_worker`` tests at the bottom,
  which skip without one unless ``AEVOM_REQUIRE_GPU_WORKER=1``.

Run with: pytest services/vina/tests/test_conformer_worker.py
"""

import ast
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdCIPLabeler

from app import conformer, conformer_client
from app.conformer import (
    BACKEND_CPU,
    BACKEND_CPU_FALLBACK,
    BACKEND_GPU_WORKER,
    generate_conformers,
    generate_conformers_with_report,
)
from app.conformer_client import ConformerWorkerClient, ConformerWorkerError, WorkerConfig, framing

TESTS = Path(__file__).resolve().parent
APP_ROOT = TESTS.parent
STUBS = TESTS / "worker_stubs"
STUB_PYTHON = STUBS / "stub-python"
FAKE_WORKER = TESTS / "fake_conformer_worker.py"
WORKER_DIR = APP_ROOT / "conformer_worker"
WORKER_SCRIPT = WORKER_DIR / "worker.py"
DATA = TESTS / "data"
HAS_VINA = shutil.which("vina") is not None

IMATINIB = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
ZWITTERION = "C[C@H]([NH3+])C(=O)[O-]"
ALKENE = "C/C=C/C[C@@H](O)c1ccccc1"
ISOTOPE = "[2H]C([2H])([2H])c1ccc(O)cc1"
ROUND_TRIP = {"imatinib": IMATINIB, "zwitterion": ZWITTERION, "alkene": ALKENE, "isotope": ISOTOPE}


# ---------------------------------------------------------------------------
# helpers and fixtures
# ---------------------------------------------------------------------------


def embedded(smiles: str, seed: int = 11) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    return mol


def atom_table(mol):
    return [(a.GetAtomicNum(), a.GetFormalCharge(), a.GetIsotope(), a.GetTotalNumHs()) for a in mol.GetAtoms()]


def bond_table(mol):
    return sorted(
        (min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), str(b.GetBondType()))
        for b in mol.GetBonds()
    )


def cip_labels(mol, from_3d: bool) -> dict:
    probe = Chem.Mol(mol)
    if from_3d:
        Chem.AssignStereochemistryFrom3D(probe)
    rdCIPLabeler.AssignCIPLabels(probe)
    atoms = {("a", a.GetIdx()): a.GetProp("_CIPCode") for a in probe.GetAtoms() if a.HasProp("_CIPCode")}
    bonds = {("b", b.GetIdx()): b.GetProp("_CIPCode") for b in probe.GetBonds() if b.HasProp("_CIPCode")}
    return {**atoms, **bonds}


def mmff_energy(mol) -> float:
    props = AllChem.MMFFGetMoleculeProperties(mol)
    return AllChem.MMFFGetMoleculeForceField(mol, props).CalcEnergy()


def process_gone(pid: int, wait_s: float = 10.0) -> bool:
    """True once *pid* no longer runs (gone, or a zombie of an unrelated reaper)."""
    end = time.monotonic() + wait_s
    while time.monotonic() < end:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            return True
        if state in ("Z", "X"):
            return True
        time.sleep(0.05)
    return False


def make_config(python, script, **overrides) -> WorkerConfig:
    values = dict(
        python=str(python),
        script=str(script),
        startup_timeout_s=60.0,
        request_timeout_s=30.0,
        max_payload_bytes=64 * 1024 * 1024,
        restart_backoff_s=0.0,
        ld_library_path="",
        frame_timeout_s=10.0,
    )
    values.update(overrides)
    return WorkerConfig(**values)


@pytest.fixture
def install(monkeypatch):
    """Make a client the process-wide one that ``conformer.py`` uses."""
    created = []

    def _install(config: WorkerConfig) -> ConformerWorkerClient:
        client = ConformerWorkerClient(config)
        monkeypatch.setattr(conformer_client, "_client", client)
        created.append(client)
        return client

    yield _install
    for client in created:
        client.shutdown()


@pytest.fixture
def stub_worker(monkeypatch, install):
    """The real worker.py, with the CPU stand-in for nvMolKit."""
    monkeypatch.setenv("STUB_REAL_PYTHON", sys.executable)

    def _make(mode: str = "normal", **overrides) -> ConformerWorkerClient:
        monkeypatch.setenv("STUB_NVMOLKIT_MODE", mode)
        return install(make_config(STUB_PYTHON, WORKER_SCRIPT, **overrides))

    return _make


@pytest.fixture
def fake_worker(monkeypatch, install, tmp_path):
    """fake_conformer_worker.py in a given mode; ``client.spawns()`` lists started pids."""
    log = tmp_path / "spawns"
    monkeypatch.setenv("FAKE_WORKER_SPAWN_LOG", str(log))

    def _make(mode: str, **overrides) -> ConformerWorkerClient:
        monkeypatch.setenv("FAKE_WORKER_MODE", mode)
        client = install(make_config(sys.executable, FAKE_WORKER, **overrides))
        client.spawns = lambda: [int(p) for p in log.read_text().split()] if log.exists() else []
        return client

    return _make


@pytest.fixture
def cpu_spy(monkeypatch):
    """Record how many conformers each CPU-fallback input already carried."""
    seen = []
    original = conformer._generate_conformers_cpu

    def spy(mol, num_conformers):
        seen.append(mol.GetNumConformers())
        return original(mol, num_conformers)

    monkeypatch.setattr(conformer, "_generate_conformers_cpu", spy)
    return seen


def test_stub_python_is_executable():
    assert os.access(STUB_PYTHON, os.X_OK), "chmod +x services/vina/tests/worker_stubs/stub-python"


# ---------------------------------------------------------------------------
# framing
# ---------------------------------------------------------------------------


@pytest.fixture
def pipe():
    r, w = os.pipe()
    os.set_blocking(r, False)
    os.set_blocking(w, False)
    fds = {"r": r, "w": w}
    yield fds
    for fd in fds.values():
        try:
            os.close(fd)
        except OSError:
            pass


def close(fds, end):
    os.close(fds[end])
    fds[end] = os.open("/dev/null", os.O_RDONLY)


class TestFraming:
    LIMIT = 1 << 20

    def test_round_trip(self, pipe):
        message = {"protocol": 1, "id": "x", "text": "Ångström ✓", "n": [1, 2.5, None]}
        framing.write_frame(pipe["w"], message, max_bytes=self.LIMIT, deadline=framing.deadline_after(1))
        assert framing.read_frame(pipe["r"], max_bytes=self.LIMIT, deadline=framing.deadline_after(1)) == message

    def test_clean_eof_before_a_frame_is_none(self, pipe):
        close(pipe, "w")
        assert framing.read_frame(pipe["r"], max_bytes=self.LIMIT, deadline=framing.deadline_after(1)) is None

    @pytest.mark.parametrize("partial", [b"\x00\x00", b"\x00\x00\x00\x10{\"a\":"])
    def test_eof_inside_a_frame(self, pipe, partial):
        os.write(pipe["w"], partial)
        close(pipe, "w")
        with pytest.raises(framing.FrameEOF):
            framing.read_frame(pipe["r"], max_bytes=self.LIMIT, deadline=framing.deadline_after(1))

    @pytest.mark.parametrize("partial", [b"\x00", b"\x00\x00\x00\x10{\"a\":"])
    def test_stalled_frame_times_out_at_the_deadline(self, pipe, partial):
        os.write(pipe["w"], partial)
        start = time.monotonic()
        with pytest.raises(framing.FrameTimeout):
            framing.read_frame(pipe["r"], max_bytes=self.LIMIT, deadline=framing.deadline_after(0.3))
        assert 0.25 < time.monotonic() - start < 1.5

    def test_frame_timeout_bounds_a_started_frame_without_a_deadline(self, pipe):
        os.write(pipe["w"], b"\x00\x00")
        start = time.monotonic()
        with pytest.raises(framing.FrameTimeout):
            framing.read_frame(pipe["r"], max_bytes=self.LIMIT, deadline=None, frame_timeout=0.3)
        assert time.monotonic() - start < 1.5

    def test_announced_oversize_is_refused_before_the_payload(self, pipe):
        os.write(pipe["w"], (self.LIMIT + 1).to_bytes(4, "big"))
        with pytest.raises(framing.FrameTooLarge):
            framing.read_frame(pipe["r"], max_bytes=self.LIMIT, deadline=framing.deadline_after(1))

    def test_oversize_is_refused_before_sending(self, pipe):
        with pytest.raises(framing.FrameTooLarge):
            framing.write_frame(pipe["w"], {"pad": "x" * 2000}, max_bytes=1000, deadline=None)
        with pytest.raises(BlockingIOError):
            os.read(pipe["r"], 1)

    @pytest.mark.parametrize(
        "payload", [b"not json", b"[1, 2]", b'{"e": NaN}', b'{"e": Infinity}', b"\xff\xfe"]
    )
    def test_malformed_payloads(self, pipe, payload):
        os.write(pipe["w"], len(payload).to_bytes(4, "big") + payload)
        with pytest.raises(framing.FrameError):
            framing.read_frame(pipe["r"], max_bytes=self.LIMIT, deadline=framing.deadline_after(1))

    def test_non_finite_numbers_are_not_encoded(self):
        with pytest.raises(framing.FrameError):
            framing.encode_frame({"e": float("nan")}, self.LIMIT)

    def test_write_to_a_reader_that_never_drains_times_out(self, pipe):
        start = time.monotonic()
        with pytest.raises(framing.FrameTimeout):
            framing.write_frame(pipe["w"], {"pad": "x" * (1 << 20)}, max_bytes=2 << 20, deadline=framing.deadline_after(0.3))
        assert time.monotonic() - start < 1.5

    def test_write_after_the_reader_closed(self, pipe):
        close(pipe, "r")
        with pytest.raises(framing.FrameEOF):
            framing.write_frame(pipe["w"], {"a": 1}, max_bytes=self.LIMIT, deadline=framing.deadline_after(1))

    def test_one_byte_chunks(self, pipe):
        frame = framing.encode_frame({"id": "slow", "pad": "y" * 300}, self.LIMIT)

        def trickle():
            for byte in frame:
                os.write(pipe["w"], bytes([byte]))
                time.sleep(0.001)

        thread = threading.Thread(target=trickle)
        thread.start()
        got = framing.read_frame(pipe["r"], max_bytes=self.LIMIT, deadline=framing.deadline_after(10))
        thread.join()
        assert got["id"] == "slow"


# ---------------------------------------------------------------------------
# round trips through the real worker
# ---------------------------------------------------------------------------


class TestRoundTrip:
    @pytest.mark.parametrize("name", sorted(ROUND_TRIP))
    def test_conformers_keep_the_molecule(self, stub_worker, name):
        stub_worker()
        mol = embedded(ROUND_TRIP[name])
        reference = Chem.AddHs(mol, addCoords=True)
        input_xyz = reference.GetConformer().GetPositions()
        expected_cip = cip_labels(reference, from_3d=False)

        confs, report = generate_conformers_with_report(mol, num_conformers=6, use_gpu=True)

        assert report.backend == BACKEND_GPU_WORKER, report.fallback_reason
        assert 1 <= len(confs) <= 6
        for conf in confs:
            assert conf.GetNumConformers() == 1
            assert atom_table(conf) == atom_table(reference)
            assert bond_table(conf) == bond_table(reference)
            xyz = conf.GetConformer().GetPositions()
            assert np.isfinite(xyz).all()
            assert not np.allclose(xyz, input_xyz, atol=1e-3), "the input pose came back as a conformer"
            assert cip_labels(conf, from_3d=True) == expected_cip
            assert Chem.MolToSmiles(Chem.RemoveHs(conf)) == Chem.MolToSmiles(Chem.RemoveHs(reference))
        assert mol.GetNumConformers() == 1, "the caller's molecule was modified"

    def test_input_pose_is_not_counted_although_the_backend_appends(self, stub_worker):
        client = stub_worker()
        mol = Chem.AddHs(embedded(ALKENE), addCoords=True)
        result = client.generate(mol, 5, 200)
        assert result.num_embedded == 5
        assert len(result.conformers) == 5

    def test_energies_belong_to_the_returned_coordinates(self, stub_worker):
        client = stub_worker()
        mol = Chem.AddHs(embedded(IMATINIB), addCoords=True)
        result = client.generate(mol, 8, 200)
        assert result.conformers
        for energy, conf in result.conformers:
            assert mmff_energy(conf) == pytest.approx(energy, abs=0.1)

    def test_api_orders_by_energy(self, stub_worker):
        stub_worker()
        confs = generate_conformers(embedded(IMATINIB), num_conformers=8, use_gpu=True)
        energies = [mmff_energy(c) for c in confs]
        assert all(b >= a - 0.1 for a, b in zip(energies, energies[1:])), energies

    def test_api_follows_worker_energies_not_worker_order(self, fake_worker):
        """The fake gives conformer i energy -i, so the worker's last is the lowest."""
        client = fake_worker("reverse")
        mol = embedded(ALKENE)
        direct = client.generate(Chem.AddHs(mol, addCoords=True), 5, 200)
        confs = generate_conformers(mol, num_conformers=5, use_gpu=True)
        expected = [c.GetConformer().GetPositions() for _e, c in sorted(direct.conformers, key=lambda ec: ec[0])]
        assert len(confs) == len(expected) == 5
        for got, want in zip(confs, expected):
            assert np.allclose(got.GetConformer().GetPositions(), want, atol=1e-3)

    def test_partial_embedding_is_a_partial_success(self, stub_worker):
        stub_worker("partial")
        confs, report = generate_conformers_with_report(embedded(ALKENE), num_conformers=6, use_gpu=True)
        assert report.backend == BACKEND_GPU_WORKER
        assert len(confs) == 3

    def test_non_finite_energy_rejects_only_that_conformer(self, stub_worker):
        client = stub_worker("nan_energy")
        result = client.generate(Chem.AddHs(embedded(ALKENE), addCoords=True), 4, 200)
        assert len(result.rejected) == 1 and "energy" in result.rejected[0]
        assert len(result.conformers) == 3

    def test_payload_limit_truncates_instead_of_failing(self, stub_worker):
        client = stub_worker(max_payload_bytes=4096 + 20_000)
        mol = Chem.AddHs(embedded(IMATINIB), addCoords=True)
        result = client.generate(mol, 10, 50)
        assert result.truncated
        assert 1 <= len(result.conformers) < 10

    def test_stray_stdout_from_the_backend_does_not_corrupt_the_protocol(self, stub_worker):
        stub_worker("print")
        confs, report = generate_conformers_with_report(embedded(ALKENE), num_conformers=3, use_gpu=True)
        assert report.backend == BACKEND_GPU_WORKER, report.fallback_reason
        assert confs


# ---------------------------------------------------------------------------
# concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    THREADS = 16

    @pytest.mark.parametrize("make", ["stub", "fake_one_byte"])
    def test_concurrent_requests_are_serialized(self, stub_worker, fake_worker, make):
        client = stub_worker("slow") if make == "stub" else fake_worker("one_byte")
        mol = embedded(ALKENE)
        reports, errors = [], []
        barrier = threading.Barrier(self.THREADS)

        def run():
            try:
                barrier.wait()
                confs, report = generate_conformers_with_report(mol, num_conformers=2, use_gpu=True)
                assert confs
                reports.append(report)
            except BaseException as exc:  # noqa: BLE001 -- surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=run) for _ in range(self.THREADS)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(120)

        assert not errors, errors
        assert len(reports) == self.THREADS
        assert {r.backend for r in reports} == {BACKEND_GPU_WORKER}, [r.fallback_reason for r in reports]
        assert len({r.worker_pid for r in reports}) == 1
        request_ids = {r.worker_request_id for r in reports}
        assert len(request_ids) == self.THREADS
        if make == "stub":
            inventory = client.inventory()
            assert request_ids <= {s["request_id"] for s in inventory["served"]}
            assert inventory["gpu_generations"] == self.THREADS


# ---------------------------------------------------------------------------
# failures: every one falls back to a clean CPU run and the worker is replaced
# ---------------------------------------------------------------------------

TRANSPORT_FAILURES = [
    "hang",
    "partial_header",
    "partial_payload",
    "crash",
    "eof_mid_payload",
    "garbage",
    "non_object",
    "wrong_id",
    "wrong_protocol",
    "oversized",
]
REJECTED_EVERY_CONFORMER = [
    "atom_mismatch",
    "charge_mismatch",
    "bond_mismatch",
    "flipped_stereo",
    "null_energy",
    "string_energy",
    "zero",
]
REJECTED_REPLY = ["too_many", "duplicate_index", "error_reply"]
STARTUP_FAILURES = ["never_ready", "not_ready", "bad_handshake", "exit_before_ready"]


class TestFailures:
    @pytest.mark.parametrize("mode", TRANSPORT_FAILURES)
    def test_transport_failure_kills_the_worker_and_the_next_request_restarts_it(
        self, fake_worker, monkeypatch, cpu_spy, mode
    ):
        client = fake_worker(mode, request_timeout_s=2.0, frame_timeout_s=1.0)
        mol = embedded(ZWITTERION)

        start = time.monotonic()
        confs, report = generate_conformers_with_report(mol, num_conformers=3, use_gpu=True)
        assert time.monotonic() - start < 2.0 + 15

        assert report.backend == BACKEND_CPU_FALLBACK
        assert report.fallback_reason
        assert confs and cpu_spy == [0]
        (first,) = client.spawns()
        assert process_gone(first)
        assert client.pid is None

        monkeypatch.setenv("FAKE_WORKER_MODE", "normal")
        confs, report = generate_conformers_with_report(mol, num_conformers=3, use_gpu=True)
        assert report.backend == BACKEND_GPU_WORKER, report.fallback_reason
        assert report.worker_pid != first
        assert client.spawns() == [first, report.worker_pid]

    @pytest.mark.parametrize("mode", REJECTED_EVERY_CONFORMER)
    def test_rejected_conformers_fall_back_but_keep_the_worker(self, fake_worker, cpu_spy, mode):
        client = fake_worker(mode)
        confs, report = generate_conformers_with_report(embedded(ZWITTERION), num_conformers=3, use_gpu=True)
        assert report.backend == BACKEND_CPU_FALLBACK
        assert "zero usable conformers" in report.fallback_reason
        assert confs and cpu_spy == [0]
        assert client.pid == client.spawns()[0]

    @pytest.mark.parametrize("mode", REJECTED_REPLY)
    def test_rejected_reply_falls_back(self, fake_worker, cpu_spy, mode):
        fake_worker(mode)
        confs, report = generate_conformers_with_report(embedded(ZWITTERION), num_conformers=3, use_gpu=True)
        assert report.backend == BACKEND_CPU_FALLBACK
        assert "ConformerWorkerError" in report.fallback_reason
        assert confs and cpu_spy == [0]

    def test_one_bad_conformer_is_dropped(self, fake_worker):
        client = fake_worker("one_flipped")
        result = client.generate(Chem.AddHs(embedded(ZWITTERION), addCoords=True), 4, 200)
        assert len(result.rejected) == 1 and "stereo" in result.rejected[0]
        assert len(result.conformers) == 3

    def test_mirror_image_is_caught(self, fake_worker):
        client = fake_worker("flipped_stereo")
        result = client.generate(Chem.AddHs(embedded(ALKENE), addCoords=True), 3, 200)
        assert not result.conformers
        assert all("stereo" in reason for reason in result.rejected)

    @pytest.mark.parametrize("mode", STARTUP_FAILURES)
    def test_startup_failure_falls_back(self, fake_worker, cpu_spy, mode):
        client = fake_worker(mode, startup_timeout_s=2.0)
        start = time.monotonic()
        confs, report = generate_conformers_with_report(embedded(ZWITTERION), num_conformers=3, use_gpu=True)
        assert time.monotonic() - start < 2.0 + 15
        assert report.backend == BACKEND_CPU_FALLBACK and report.fallback_reason
        assert confs and cpu_spy == [0]
        assert all(process_gone(pid) for pid in client.spawns())

    def test_startup_failure_backs_off(self, fake_worker):
        client = fake_worker("not_ready", restart_backoff_s=60.0)
        _confs, first = generate_conformers_with_report(embedded(ZWITTERION), num_conformers=2, use_gpu=True)
        _confs, second = generate_conformers_with_report(embedded(ZWITTERION), num_conformers=2, use_gpu=True)
        assert "not ready" in first.fallback_reason
        assert "unavailable" in second.fallback_reason
        assert len(client.spawns()) == 1

    def test_missing_interpreter_falls_back(self, install, cpu_spy):
        install(make_config("/nonexistent/aevom-conformer-worker/bin/python", WORKER_SCRIPT))
        confs, report = generate_conformers_with_report(embedded(ZWITTERION), num_conformers=2, use_gpu=True)
        assert report.backend == BACKEND_CPU_FALLBACK
        assert "not found" in report.fallback_reason
        assert confs and cpu_spy == [0]

    def test_worker_that_never_reads_cannot_block_the_write(self, fake_worker):
        client = fake_worker("never_read")
        client.ensure_ready()
        start = time.monotonic()
        with pytest.raises(ConformerWorkerError, match="writing frame"):
            with client._acquire(5):
                client._exchange({"type": "generate", "id": "x" * 32, "pad": "x" * (1 << 20)}, 1.0)
        assert time.monotonic() - start < 5
        assert process_gone(client.spawns()[0])

    def test_backend_exception_is_fatal_and_the_worker_is_replaced(self, stub_worker, monkeypatch):
        client = stub_worker("raise")
        confs, report = generate_conformers_with_report(embedded(ALKENE), num_conformers=2, use_gpu=True)
        assert report.backend == BACKEND_CPU_FALLBACK
        assert "simulated CUDA failure" in report.fallback_reason
        assert confs
        assert client.pid is None

        monkeypatch.setenv("STUB_NVMOLKIT_MODE", "normal")
        _confs, report = generate_conformers_with_report(embedded(ALKENE), num_conformers=2, use_gpu=True)
        assert report.backend == BACKEND_GPU_WORKER

    def test_backend_that_embeds_nothing_falls_back(self, stub_worker, cpu_spy):
        stub_worker("zero")
        confs, report = generate_conformers_with_report(embedded(ALKENE), num_conformers=3, use_gpu=True)
        assert report.backend == BACKEND_CPU_FALLBACK
        assert "0 embedded" in report.fallback_reason
        assert confs and cpu_spy == [0]

    def test_failed_warm_up_means_not_ready(self, stub_worker):
        stub_worker("warmup_fail")
        _confs, report = generate_conformers_with_report(embedded(ALKENE), num_conformers=2, use_gpu=True)
        assert report.backend == BACKEND_CPU_FALLBACK
        assert "warm-up" in report.fallback_reason


class TestCpuPath:
    def test_cpu_request_never_starts_a_worker(self, fake_worker, cpu_spy):
        client = fake_worker("normal")
        mol = embedded(ALKENE)
        confs, report = generate_conformers_with_report(mol, num_conformers=3, use_gpu=False)
        assert report.backend == BACKEND_CPU and report.fallback_reason is None
        assert confs and cpu_spy == [0]
        assert client.spawns() == []
        assert mol.GetNumConformers() == 1


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_shutdown_reaps_the_worker(self, stub_worker):
        client = stub_worker()
        client.ensure_ready()
        pid = client.pid
        client.shutdown()
        assert client.pid is None
        assert process_gone(pid)
        with pytest.raises(ConformerWorkerError, match="shut down"):
            client.generate(Chem.AddHs(embedded(ALKENE), addCoords=True), 2, 50)

    def test_worker_exits_cleanly_when_stdin_closes(self, stub_worker):
        client = stub_worker()
        client.ensure_ready()
        proc = client._proc
        proc.stdin.close()
        assert proc.wait(10) == 0

    def test_worker_dies_with_the_service_process(self, tmp_path):
        script = tmp_path / "parent.py"
        script.write_text(
            "import os, signal, sys\n"
            f"sys.path.insert(0, {str(APP_ROOT)!r})\n"
            "from app.conformer_client import ConformerWorkerClient, WorkerConfig\n"
            "client = ConformerWorkerClient(WorkerConfig.from_env())\n"
            "client.ensure_ready()\n"
            "print(client.pid, flush=True)\n"
            "os.kill(os.getpid(), signal.SIGKILL)\n"
        )
        env = {
            **os.environ,
            "CONFORMER_WORKER_PYTHON": str(STUB_PYTHON),
            "CONFORMER_WORKER_SCRIPT": str(WORKER_SCRIPT),
            "STUB_REAL_PYTHON": sys.executable,
            "STUB_NVMOLKIT_MODE": "normal",
        }
        out = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=120)
        worker_pid = int(out.stdout.strip())
        assert process_gone(worker_pid), "worker outlived a SIGKILLed parent"

    def test_worker_refuses_to_start_for_a_parent_that_is_gone(self):
        env = {**os.environ, "STUB_REAL_PYTHON": sys.executable, "AEVOM_CONFORMER_PARENT_PID": "999999999"}
        out = subprocess.run(
            [str(STUB_PYTHON), "-I", str(WORKER_SCRIPT)], env=env, capture_output=True, timeout=60, stdin=subprocess.PIPE
        )
        assert out.returncode == 1
        assert out.stdout == b""

    def test_fork_does_not_share_the_parents_worker(self, stub_worker):
        client = stub_worker()
        client.ensure_ready()
        parent_worker = client.pid
        read, write = os.pipe()
        child = os.fork()
        if child == 0:
            os.close(read)
            try:
                client.ensure_ready()
                os.write(write, str(client.pid).encode())
                client.shutdown()
            finally:
                os._exit(0)
        os.close(write)
        child_worker = int(os.read(read, 64) or b"0")
        os.waitpid(child, 0)
        assert child_worker not in (0, parent_worker)
        assert client.pid == parent_worker


# ---------------------------------------------------------------------------
# isolation
# ---------------------------------------------------------------------------

FORBIDDEN_IN_WORKER = ("meeko", "openbabel", "plip", "app", "vina", "pdbfixer")
FORBIDDEN_IN_SERVICE = ("nvmolkit", "torch")


def top_level(modules) -> set[str]:
    return {m.split(".")[0] for m in modules}


class TestIsolation:
    def test_worker_ignores_the_services_python_environment(self, stub_worker, monkeypatch, tmp_path):
        marker = tmp_path / "poisoned"
        poison = f"open({str(marker)!r}, 'a').write(__name__ + '\\n')\n"
        path_dir = tmp_path / "pythonpath"
        home = tmp_path / "home"
        user_site = home / ".local" / "lib" / f"python{sys.version_info[0]}.{sys.version_info[1]}" / "site-packages"
        for root in (path_dir, user_site):
            for name in ("meeko", "openbabel", "plip", "app"):
                (root / name).mkdir(parents=True)
                (root / name / "__init__.py").write_text(poison)
            (root / "sitecustomize.py").write_text(poison)
            (root / "usercustomize.py").write_text(poison)
        monkeypatch.setenv("PYTHONPATH", str(path_dir))
        monkeypatch.setenv("PYTHONSTARTUP", str(path_dir / "sitecustomize.py"))
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("LD_PRELOAD", "")

        client = stub_worker()
        inventory = client.inventory()

        assert not marker.exists(), marker.read_text()
        loaded = top_level(inventory["modules"])
        assert "nvmolkit" in loaded and "rdkit" in loaded
        assert not loaded & set(FORBIDDEN_IN_WORKER), loaded & set(FORBIDDEN_IN_WORKER)
        assert inventory["ppid"] == os.getpid()
        assert inventory["pid"] != os.getpid()

    def test_worker_inherits_cuda_visible_devices(self, stub_worker, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
        client = stub_worker()
        assert client.ensure_ready()["cuda_visible_devices"] == "3"
        assert client.inventory()["cuda_visible_devices"] == "3"

    def test_worker_gets_only_the_configured_library_path(self, stub_worker, monkeypatch):
        monkeypatch.setenv("LD_LIBRARY_PATH", "/should/not/reach/the/worker")
        client = stub_worker(ld_library_path="/only/this")
        env = client._worker_env()
        assert env["LD_LIBRARY_PATH"] == "/only/this"
        assert not any(k.startswith("PYTHON") for k in env)

    def test_service_never_imports_nvmolkit_or_torch(self):
        code = (
            "import sys\n"
            "import app.main, app.pipeline, app.conformer, app.conformer_client, app.preparation\n"
            f"bad = sorted({{m.split('.')[0] for m in sys.modules}} & {set(FORBIDDEN_IN_SERVICE)!r})\n"
            "assert not bad, bad\n"
        )
        out = subprocess.run([sys.executable, "-c", code], cwd=APP_ROOT, capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr

    def test_service_source_has_no_nvmolkit_import(self):
        for path in (APP_ROOT / "app").glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                assert not top_level(names) & set(FORBIDDEN_IN_SERVICE), f"{path.name} imports {names}"

    def test_worker_files_import_nothing_from_the_service(self):
        allowed = {
            "framing.py": set(sys.stdlib_module_names),
            "worker.py": set(sys.stdlib_module_names) | {"rdkit", "nvmolkit"},
        }
        for name, ok in allowed.items():
            for node in ast.walk(ast.parse((WORKER_DIR / name).read_text())):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    assert node.level == 0, f"{name} uses a relative import"
                    modules = [node.module]
                else:
                    continue
                assert top_level(modules) <= ok, f"{name} imports {modules}"


# ---------------------------------------------------------------------------
# the pipeline and the API
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pocket():
    from app.preparation import prepare_receptor

    receptor, _atoms, error = prepare_receptor(
        (DATA / "pocket.pdb").read_text(), remove_water=True, remove_heteroatoms=True
    )
    assert error is None, error
    center = tuple(float(v) for v in (DATA / "pocket_center.txt").read_text().split())
    return receptor, center


@pytest.mark.skipif(not HAS_VINA, reason="vina binary not installed")
class TestPipelineDiagnostics:
    def test_ensemble_result_reports_its_backend(self, pocket):
        from app.pipeline import run_auto_dock

        receptor, center = pocket
        result = run_auto_dock(
            receptor_pdb=receptor,
            ligand_sdf=(DATA / "ligand.sdf").read_text(),
            center=center,
            size=(20.0, 20.0, 20.0),
            exhaustiveness=1,
            num_modes=1,
            num_conformers=2,
            rot_bond_threshold=0,
        )
        assert result["success"], result["error"]
        assert result["docking_mode"] == "rigid_ensemble"
        assert result["_diagnostics"]["conformer"]["backend"] in {
            BACKEND_CPU,
            BACKEND_CPU_FALLBACK,
            BACKEND_GPU_WORKER,
        }

    def test_diagnostics_stay_out_of_the_api_response(self, pocket):
        from fastapi.testclient import TestClient

        from app.main import app

        receptor, center = pocket
        response = TestClient(app).post(
            "/dock-auto",
            json={
                "receptor_pdb": receptor,
                "ligand_sdf": (DATA / "ligand.sdf").read_text(),
                "center_x": center[0],
                "center_y": center[1],
                "center_z": center[2],
                "exhaustiveness": 1,
                "num_modes": 1,
                "num_conformers": 2,
                "rot_bond_threshold": 0,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["success"], body["error"]
        assert body["docking_mode"] == "rigid_ensemble"
        assert "_diagnostics" not in body
        assert set(body) >= {"gpu_accelerated", "conformers_generated", "best_conformer_idx", "docking_mode"}


# ---------------------------------------------------------------------------
# the real worker on a GPU
# ---------------------------------------------------------------------------


@pytest.mark.gpu_worker
class TestRealGpuWorker:
    def test_ready_handshake_reports_versions(self, gpu_worker):
        info = gpu_worker.ready_info
        assert info["ok"] and info["pid"] == gpu_worker.pid
        assert info["executable"].startswith("/opt/aevom-conformer-worker/")
        assert info["versions"]["nvmolkit"] != "unknown"

    @pytest.mark.parametrize("name", sorted(ROUND_TRIP))
    def test_gpu_round_trip(self, gpu_worker, name):
        mol = embedded(ROUND_TRIP[name])
        reference = Chem.AddHs(mol, addCoords=True)
        before = gpu_worker.inventory()["gpu_generations"]

        confs, report = generate_conformers_with_report(mol, num_conformers=8, use_gpu=True)

        assert report.backend == BACKEND_GPU_WORKER, f"fell back to CPU: {report.fallback_reason}"
        inventory = gpu_worker.inventory()
        assert inventory["gpu_generations"] == before + 1
        assert report.worker_request_id in {s["request_id"] for s in inventory["served"]}
        assert report.num_embedded <= 8 and 1 <= len(confs) <= 8
        energies = [mmff_energy(c) for c in confs]
        assert all(b >= a - 0.1 for a, b in zip(energies, energies[1:])), energies
        for conf in confs:
            assert conf.GetNumConformers() == 1
            assert atom_table(conf) == atom_table(reference)
            assert bond_table(conf) == bond_table(reference)
            assert cip_labels(conf, from_3d=True) == cip_labels(reference, from_3d=False)

    def test_gpu_energies_belong_to_the_returned_coordinates(self, gpu_worker):
        result = gpu_worker.generate(Chem.AddHs(embedded(IMATINIB), addCoords=True), 10, 200)
        assert result.num_embedded == 10
        assert not result.rejected, result.rejected
        for energy, conf in result.conformers:
            assert mmff_energy(conf) == pytest.approx(energy, abs=0.1)

    def test_gpu_concurrent_requests(self, gpu_worker):
        mol = embedded(IMATINIB)
        reports = []

        def run():
            reports.append(generate_conformers_with_report(mol, num_conformers=4, use_gpu=True)[1])

        threads = [threading.Thread(target=run) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(300)
        assert len(reports) == 8
        assert {r.backend for r in reports} == {BACKEND_GPU_WORKER}, [r.fallback_reason for r in reports]
        assert len({r.worker_pid for r in reports}) == 1

    def test_gpu_worker_holds_no_service_packages(self, gpu_worker):
        inventory = gpu_worker.inventory()
        loaded = top_level(inventory["modules"])
        assert "nvmolkit" in loaded
        assert not loaded & set(FORBIDDEN_IN_WORKER)
        assert any("libcuda.so" in so for so in inventory["shared_objects"])
