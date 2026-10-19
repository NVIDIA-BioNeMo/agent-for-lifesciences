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

"""Conformer ensemble generation with GPU (nvMolKit) and CPU (RDKit) backends.

Generates multiple 3D conformers for a molecule using ETKDG distance geometry
followed by MMFF force-field optimization. On GPU-equipped nodes, nvMolKit
provides CUDA-accelerated batch embedding and optimization. On CPU-only nodes,
RDKit's built-in implementations are used as the fallback.

nvMolKit never runs in this process: GPU requests go to a separate worker
process with its own interpreter and environment (see ``conformer_client``).
Any worker failure falls back to RDKit on CPU.

All molecular operations delegate to established comp-chem libraries (RDKit,
nvMolKit). No bespoke coordinate manipulation is performed here.
"""

import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem

from . import conformer_client

logger = logging.getLogger(__name__)


DEFAULT_CONFORMER_COUNT = int(os.environ.get("CONFORMER_COUNT", "50"))
DEFAULT_MAX_MMFF_ITERS = 200

BACKEND_GPU_WORKER = "nvmolkit_worker"
BACKEND_CPU = "rdkit_cpu"
BACKEND_CPU_FALLBACK = "rdkit_cpu_fallback"


@dataclass
class ConformerReport:
    """Which backend produced a conformer ensemble, for diagnostics only."""

    backend: str
    fallback_reason: Optional[str] = None
    worker_pid: Optional[int] = None
    worker_request_id: Optional[str] = None
    num_embedded: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def generate_conformers(
    mol: Chem.Mol,
    num_conformers: int = DEFAULT_CONFORMER_COUNT,
    use_gpu: bool = False,
) -> list[Chem.Mol]:
    """Generate an ensemble of 3D conformers for a molecule.

    The molecule is embedded with ETKDG and each conformer is relaxed with
    MMFF94. The backend (nvMolKit worker vs RDKit) is selected based on
    *use_gpu* and whether the GPU worker can serve the request.

    Args:
        mol: RDKit Mol. Hydrogens are added; the input is not modified.
        num_conformers: Number of conformers to generate.
        use_gpu: Request GPU acceleration (requires the nvMolKit worker + CUDA).

    Returns:
        List of RDKit Mol objects, each carrying exactly one conformer.
        Sorted by MMFF energy (lowest first). May be shorter than
        *num_conformers* if some embeddings failed.
    """
    conformers, _report = generate_conformers_with_report(mol, num_conformers, use_gpu)
    return conformers


def generate_conformers_with_report(
    mol: Chem.Mol,
    num_conformers: int = DEFAULT_CONFORMER_COUNT,
    use_gpu: bool = False,
) -> tuple[list[Chem.Mol], ConformerReport]:
    """:func:`generate_conformers`, plus a report of the backend that ran."""
    start = time.time()

    mol = Chem.AddHs(mol, addCoords=True)

    fallback_reason = None
    if use_gpu:
        try:
            result = conformer_client.get_client().generate(mol, num_conformers, DEFAULT_MAX_MMFF_ITERS)
        except Exception as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"
        else:
            if result.conformers:
                ordered = sorted(enumerate(result.conformers), key=lambda item: (item[1][0], item[0]))
                conformers = [conf_mol for _i, (_energy, conf_mol) in ordered]
                report = ConformerReport(
                    backend=BACKEND_GPU_WORKER,
                    worker_pid=result.worker_pid,
                    worker_request_id=result.request_id,
                    num_embedded=result.num_embedded,
                )
                _log_done(conformers, start, f"nvMolKit GPU worker pid={result.worker_pid}")
                return conformers, report
            fallback_reason = (
                f"worker pid={result.worker_pid} produced zero usable conformers "
                f"({result.num_embedded} embedded, {len(result.rejected)} rejected)"
            )
        logger.warning(f"GPU conformer generation failed ({fallback_reason}); falling back to RDKit CPU")

    backend = BACKEND_CPU_FALLBACK if use_gpu else BACKEND_CPU
    cpu_mol = Chem.Mol(mol)
    cpu_mol.RemoveAllConformers()
    energies = _generate_conformers_cpu(cpu_mol, num_conformers)

    report = ConformerReport(backend=backend, fallback_reason=fallback_reason, num_embedded=cpu_mol.GetNumConformers())
    if cpu_mol.GetNumConformers() == 0:
        logger.error("Conformer generation produced zero conformers")
        return [], report

    conformers = _split_conformers(cpu_mol, energies)
    _log_done(conformers, start, "RDKit CPU (GPU fallback)" if use_gpu else "RDKit CPU")
    return conformers, report


def _log_done(conformers: list[Chem.Mol], start: float, backend: str) -> None:
    elapsed_ms = int((time.time() - start) * 1000)
    logger.info(f"Generated {len(conformers)} conformers in {elapsed_ms}ms (backend={backend})")


# ---------------------------------------------------------------------------
# CPU backend (RDKit)
# ---------------------------------------------------------------------------


def _generate_conformers_cpu(
    mol: Chem.Mol,
    num_conformers: int,
) -> list[float]:
    """Embed + MMFF-optimize conformers on CPU via RDKit.

    Returns list of MMFF energies (one per conformer), in conformer-index order.
    """
    params = AllChem.ETKDGv3()
    params.useRandomCoords = True
    params.numThreads = 0  # use all available cores

    conf_ids = AllChem.EmbedMultipleConfs(
        mol,
        numConfs=num_conformers,
        params=params,
    )

    n_embedded = len(conf_ids)
    logger.info(f"RDKit embedded {n_embedded}/{num_conformers} conformers")

    if n_embedded == 0:
        return []

    energies: list[float] = []
    for cid in conf_ids:
        try:
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props is None:
                energies.append(0.0)
                continue
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
            if ff is None:
                energies.append(0.0)
                continue
            ff.Minimize(maxIts=DEFAULT_MAX_MMFF_ITERS)
            energies.append(ff.CalcEnergy())
        except Exception:
            energies.append(0.0)

    return energies


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_conformers(
    mol: Chem.Mol,
    energies: list[float],
) -> list[Chem.Mol]:
    """Split a multi-conformer molecule into individual Mol objects.

    Returns Mols sorted by energy (lowest first). Each returned Mol has
    exactly one conformer.
    """
    n_confs = mol.GetNumConformers()
    if len(energies) < n_confs:
        energies.extend([0.0] * (n_confs - len(energies)))

    indexed = list(enumerate(energies[:n_confs]))
    indexed.sort(key=lambda pair: pair[1])

    result: list[Chem.Mol] = []
    for conf_idx, _energy in indexed:
        conf_mol = Chem.RWMol(mol)
        # Remove all conformers except the one we want
        keep_id = mol.GetConformer(conf_idx).GetId()
        ids_to_remove = [
            conf_mol.GetConformer(i).GetId()
            for i in range(conf_mol.GetNumConformers())
            if conf_mol.GetConformer(i).GetId() != keep_id
        ]
        for cid in ids_to_remove:
            conf_mol.RemoveConformer(cid)
        result.append(conf_mol.GetMol())

    return result
