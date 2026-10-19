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

"""Torch-free replacement for ``safe/__init__.py``.

Upstream's ``__init__`` eagerly imports ``.tokenizer`` (``import torch``),
``.sample`` (transformers), ``.io`` (wandb) and ``.viz`` (matplotlib), so
``import safe`` drags in several gigabytes of deep-learning stack. Aevom only
needs SAFE notation conversion, which lives entirely in ``.converter`` and
``.utils`` and needs nothing beyond datamol, networkx, numpy and RDKit.

Everything heavy stays reachable through PEP 562 lazy attribute access, so a
caller that genuinely wants ``SAFETokenizer`` gets a normal ``ImportError``
naming the missing dependency rather than an ``AttributeError`` implying the
package is broken.

Copied over the installed ``safe/__init__.py`` by ``services/rdkit/Dockerfile``;
the original is preserved next to it as ``_init_full.py``. Pinned against
``safe-mol==0.1.14`` -- bumping that pin means re-checking this surface.
"""

import importlib

from ._exception import SAFEDecodeError, SAFEEncodeError, SAFEFragmentationError
from .converter import SAFEConverter, decode, encode

__all__ = [
    "SAFEConverter",
    "SAFEDecodeError",
    "SAFEDesign",
    "SAFEEncodeError",
    "SAFEFragmentationError",
    "SAFETokenizer",
    "decode",
    "encode",
    "split",
    "to_image",
    "trainer",
    "upload_to_wandb",
    "utils",
]

# name -> module that provides it. A value ending in ".{name}" means the
# attribute *is* the module (submodule access), otherwise it is pulled out of
# the module.
_LAZY = {
    "SAFEDesign": "safe.sample",
    "SAFETokenizer": "safe.tokenizer",
    "split": "safe.tokenizer",
    "to_image": "safe.viz",
    "upload_to_wandb": "safe.io",
    "trainer": "safe.trainer",
    "utils": "safe.utils",
}


def __getattr__(name):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'safe' has no attribute '{name}'")
    module = importlib.import_module(target)
    return module if target.endswith(f".{name}") else getattr(module, name)


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
