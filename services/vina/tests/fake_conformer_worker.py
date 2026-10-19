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

"""A misbehaving conformer worker, for testing the client's failure handling.

Speaks the worker protocol (via the real ``framing.py``) and generates real
conformers with RDKit on CPU, then breaks things according to
``FAKE_WORKER_MODE``. Every start appends this process's pid to
``FAKE_WORKER_SPAWN_LOG`` if that is set.

Startup modes: never_ready, not_ready, bad_handshake, exit_before_ready.
Request modes: normal, reverse, slow, one_byte, hang, partial_header,
partial_payload, crash, eof_mid_payload, garbage, non_object, wrong_id,
wrong_protocol, oversized, error_reply, atom_mismatch, charge_mismatch,
bond_mismatch, flipped_stereo, one_flipped, null_energy, string_energy,
too_many, duplicate_index, zero, never_read.
"""

import importlib.util
import os
import struct
import sys
import time

_FRAMING = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "conformer_worker", "framing.py")
spec = importlib.util.spec_from_file_location("fake_framing", _FRAMING)
framing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(framing)

MODE = os.environ.get("FAKE_WORKER_MODE", "normal")
MAX = int(os.environ.get("CONFORMER_WORKER_MAX_PAYLOAD_BYTES", str(64 * 1024 * 1024)))

OUT = os.dup(1)
os.dup2(2, 1)
os.set_blocking(0, False)
os.set_blocking(OUT, False)


def raw(data: bytes) -> None:
    os.set_blocking(OUT, True)
    os.write(OUT, data)
    os.set_blocking(OUT, False)


def send(obj: dict) -> None:
    obj.setdefault("protocol", framing.PROTOCOL_VERSION)
    framing.write_frame(OUT, obj, max_bytes=MAX, deadline=framing.deadline_after(30))


def sleep_forever() -> None:
    while True:
        time.sleep(3600)


def conformers_for(request: dict) -> tuple[list[dict], int]:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromMolBlock(request["molblock"], removeHs=False)
    mol.RemoveAllConformers()
    params = AllChem.ETKDGv3()
    params.useRandomCoords = True
    params.randomSeed = 7
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=request["num_conformers"], params=params))
    results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=request["max_mmff_iters"])
    entries = []
    for idx, cid in enumerate(ids):
        entries.append(
            {"conf_index": idx, "energy": results[idx][1], "molblock": Chem.MolToMolBlock(mol, confId=cid)}
        )
    return entries, len(ids)


def corrupt(entries: list[dict]) -> list[dict]:
    from rdkit import Chem

    def edit(entry: dict, fn) -> dict:
        mol = Chem.RWMol(Chem.MolFromMolBlock(entry["molblock"], removeHs=False))
        fn(mol)
        return {**entry, "molblock": Chem.MolToMolBlock(mol)}

    def mirror(mol):
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, (-p.x, p.y, p.z))

    def first_heavy(mol, symbol):
        return next(a for a in mol.GetAtoms() if a.GetSymbol() == symbol)

    if MODE == "atom_mismatch":
        return [edit(e, lambda m: first_heavy(m, "O").SetAtomicNum(16)) for e in entries]
    if MODE == "charge_mismatch":
        return [edit(e, lambda m: first_heavy(m, "N").SetFormalCharge(0)) for e in entries]
    if MODE == "bond_mismatch":

        def flip_bond(mol):
            bond = next(b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
            bond.SetBondType(Chem.BondType.SINGLE)

        return [edit(e, flip_bond) for e in entries]
    if MODE == "flipped_stereo":
        return [edit(e, mirror) for e in entries]
    if MODE == "one_flipped":
        return [edit(entries[0], mirror)] + entries[1:]
    if MODE == "null_energy":
        return [{**e, "energy": None} for e in entries]
    if MODE == "string_energy":
        return [{**e, "energy": "1.0"} for e in entries]
    if MODE == "duplicate_index":
        return [{**e, "conf_index": 0} for e in entries]
    if MODE == "reverse":
        return [{**e, "energy": -float(i)} for i, e in enumerate(entries)]
    return entries


def handle(request: dict) -> None:
    rid = request["id"]
    if request["type"] == "shutdown":
        send({"type": "bye", "id": rid})
        sys.exit(0)
    if request["type"] == "inventory":
        send({"type": "inventory", "id": rid, "pid": os.getpid(), "ppid": os.getppid()})
        return

    if MODE == "hang":
        sleep_forever()
    if MODE == "crash":
        os._exit(7)
    if MODE == "partial_header":
        raw(b"\x00\x00")
        sleep_forever()
    if MODE == "garbage":
        raw(struct.pack(">I", 9) + b"not json!")
        sleep_forever()
    if MODE == "non_object":
        raw(struct.pack(">I", 2) + b"[]")
        sleep_forever()
    if MODE == "oversized":
        raw(struct.pack(">I", MAX + 1))
        sleep_forever()

    entries, embedded = conformers_for(request)
    reply = {"type": "result", "id": rid, "conformers": corrupt(entries), "num_embedded": embedded, "truncated": False}
    if MODE == "wrong_id":
        reply["id"] = "0" * 32
    elif MODE == "wrong_protocol":
        reply["protocol"] = framing.PROTOCOL_VERSION + 1
    elif MODE == "error_reply":
        reply = {"type": "error", "id": rid, "reason": "simulated", "fatal": False}
    elif MODE == "too_many":
        reply["conformers"] = entries + entries[:1]
        reply["conformers"][-1] = {**entries[0], "conf_index": len(entries)}
    elif MODE == "zero":
        reply["conformers"], reply["num_embedded"] = [], 0

    if MODE in ("partial_payload", "eof_mid_payload"):
        frame = framing.encode_frame({"protocol": framing.PROTOCOL_VERSION, **reply}, MAX)
        raw(frame[: len(frame) // 2])
        if MODE == "eof_mid_payload":
            os._exit(0)
        sleep_forever()
    if MODE == "one_byte":
        for byte in framing.encode_frame({"protocol": framing.PROTOCOL_VERSION, **reply}, MAX):
            raw(bytes([byte]))
        return
    if MODE == "slow":
        time.sleep(0.05)
    send(reply)


def main() -> int:
    log = os.environ.get("FAKE_WORKER_SPAWN_LOG")
    if log:
        with open(log, "a") as fh:
            fh.write(f"{os.getpid()}\n")
    if MODE == "never_ready":
        sleep_forever()
    if MODE == "exit_before_ready":
        return 5
    if MODE == "not_ready":
        send({"type": "ready", "id": None, "ok": False, "reason": "simulated: no CUDA device"})
        return 1
    if MODE == "bad_handshake":
        send({"type": "ready", "id": None, "ok": True, "protocol": framing.PROTOCOL_VERSION + 1})
        sleep_forever()
    send({"type": "ready", "id": None, "ok": True, "pid": os.getpid(), "versions": {"fake": MODE}})
    if MODE == "never_read":
        sleep_forever()

    while True:
        request = framing.read_frame(0, max_bytes=MAX, deadline=None, frame_timeout=30)
        if request is None:
            return 0
        handle(request)


if __name__ == "__main__":
    sys.exit(main())
