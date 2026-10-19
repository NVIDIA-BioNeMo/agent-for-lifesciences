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

import os
import sys
import time

from rdkit.Chem import AllChem

from . import is_warmup, mode


def EmbedMolecules(molecules, params, confsPerMolecule=1, maxIterations=-1):
    current = mode()
    if current == "slow":
        time.sleep(0.05)
    if current == "print":
        os.write(1, b"\x00\x00\x00\x05junk from a native library\n")
        print("stray print from inside the backend")
        sys.stdout.flush()
    for mol in molecules:
        warm = is_warmup(mol)
        if current == "warmup_fail" or (current == "zero" and not warm):
            continue
        if current == "raise" and not warm:
            raise RuntimeError("simulated CUDA failure")
        count = confsPerMolecule
        if current == "partial" and not warm:
            count = max(1, confsPerMolecule // 2)
        params.clearConfs = False
        params.randomSeed = 0xA5
        AllChem.EmbedMultipleConfs(mol, numConfs=count, params=params)
