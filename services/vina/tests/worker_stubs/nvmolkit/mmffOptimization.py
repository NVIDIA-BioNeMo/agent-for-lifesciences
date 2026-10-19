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

from rdkit.Chem import AllChem

from . import is_warmup, mode


def MMFFOptimizeMoleculesConfs(molecules, maxIters=200):
    out = []
    for mol in molecules:
        results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=maxIters)
        energies = [energy for _not_converged, energy in results]
        if mode() == "nan_energy" and energies and not is_warmup(mol):
            energies[0] = float("nan")
        out.append(energies)
    return out
