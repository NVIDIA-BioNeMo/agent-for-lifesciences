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

"""CPU stand-in for nvMolKit, for exercising the real conformer worker in tests.

Behaviour is selected per worker process with ``STUB_NVMOLKIT_MODE``:

- ``normal``: RDKit ETKDG + MMFF.
- ``partial``: embed only half of the requested conformers.
- ``zero``: embed nothing for anything larger than the warm-up molecule.
- ``raise``: raise from EmbedMolecules after the warm-up.
- ``nan_energy``: report NaN for the first MMFF energy.
- ``print``: write junk to fd 1 and ``print()`` during each call.
- ``warmup_fail``: embed nothing at all, so readiness fails.
- ``slow``: sleep 50 ms per call.

Like nvMolKit, embedding appends to any conformers already on the molecule.
"""

import os

__version__ = "0.0.0+stub"

WARMUP_HEAVY_ATOMS = 3


def mode() -> str:
    return os.environ.get("STUB_NVMOLKIT_MODE", "normal")


def is_warmup(mol) -> bool:
    return mol.GetNumHeavyAtoms() <= WARMUP_HEAVY_ATOMS
