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

"""SAFE (Sequential Attachment-based Fragment Embedding) conversion.

SAFE rewrites a SMILES so each fragment of a molecular decomposition occupies a
contiguous substring, separated by ``.``. The result is still a valid SMILES, but
the fragment ordering is what generative models such as GenMol and SAFE-GPT are
trained on, so handing them a plain SMILES degrades generation silently rather
than failing.

``safe-mol`` is installed with ``--no-deps`` and a lazy ``__init__`` in this
image (see ``vendor/safe_init_lite.py``); only ``encode``/``decode`` are
available, which is all this module needs.
"""

from typing import Optional, Tuple

import safe as sf
from rdkit import Chem

# Slicers safe-mol's SAFEConverter accepts. BRICS is the default the SAFE paper
# and the GenMol NIM both use.
SUPPORTED_SLICERS = ("brics", "hr", "rotatable", "recap", "mmpa", "attach")
DEFAULT_SLICER = "brics"

# Tried in order when the requested slicer finds no cut points. A scaffold that
# stays in one piece cannot be decorated (see :func:`apply_mask`), so falling
# back to a different decomposition is far more useful than reporting a
# single-fragment encoding. ``attach`` is deliberately absent: on a molecule with
# no explicit ``[*]`` dummy atoms it cuts every C-H bond and emits each hydrogen
# as its own fragment, which GenMol answers with garbage.
FALLBACK_SLICERS = ("recap", "mmpa", "hr", "brics", "rotatable")


def _encode_with(smiles: str, slicer: str) -> Optional[str]:
    """Encode with one slicer, returning None when it yields nothing usable.

    A result carrying ``[H]`` fragments is treated as unusable: that is the
    ``attach`` slicer's degenerate per-hydrogen decomposition, which no
    SAFE-trained model can do anything sensible with.
    """
    try:
        encoded = sf.encode(smiles, slicer=slicer)
    except Exception:
        return None
    if not encoded or "[H]" in encoded:
        return None
    return encoded


def smiles_to_safe(
    smiles: str,
    slicer: str = DEFAULT_SLICER,
) -> Tuple[Optional[str], Optional[str], bool, str]:
    """Encode a SMILES string as SAFE.

    When ``slicer`` finds no cut points, the other slicers in
    :data:`FALLBACK_SLICERS` are tried before giving up. Which one succeeded is
    reported back, because it changes where a generation mask will attach.

    Args:
        smiles: SMILES string of the molecule.
        slicer: Preferred fragmentation algorithm, one of
            :data:`SUPPORTED_SLICERS`.

    Returns:
        Tuple of ``(safe_string, error_message, fragmented, slicer_used)``.
        ``fragmented`` is False when no slicer could decompose the molecule and
        the canonical SMILES was returned as a single fragment. Such a string
        round-trips fine but cannot be decorated -- see :func:`apply_mask`.
    """
    if not smiles or not smiles.strip():
        return None, "SMILES string is empty", False, slicer

    smiles = smiles.strip()

    if slicer not in SUPPORTED_SLICERS:
        return (
            None,
            f"Unsupported slicer '{slicer}'. Choose one of: "
            f"{', '.join(SUPPORTED_SLICERS)}",
            False,
            slicer,
        )

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, f"Invalid SMILES string: '{smiles}'", False, slicer

    for candidate in [slicer] + [s for s in FALLBACK_SLICERS if s != slicer]:
        encoded = _encode_with(smiles, candidate)
        if encoded is not None:
            return encoded, None, True, candidate

    # Small, highly-fused or single-ring molecules genuinely have no cut points
    # under any slicer. The canonical SMILES is a valid one-fragment SAFE string.
    return Chem.MolToSmiles(mol), None, False, slicer


def safe_to_smiles(safe_string: str) -> Tuple[Optional[str], Optional[str]]:
    """Decode a SAFE string back to canonical SMILES.

    Args:
        safe_string: SAFE fragment notation.

    Returns:
        Tuple of ``(smiles, error_message)``.
    """
    if not safe_string or not safe_string.strip():
        return None, "SAFE string is empty"

    safe_string = safe_string.strip()

    try:
        smiles = sf.decode(safe_string, canonical=True, ignore_errors=False)
    except sf.SAFEDecodeError as e:
        return None, f"Invalid SAFE string: {e}"
    except Exception as e:
        return None, f"SAFE decoding failed: {e}"

    if not smiles:
        return None, f"SAFE string decoded to nothing: '{safe_string}'"

    return smiles, None


def apply_mask(
    safe_string: str, frag_min: int, frag_max: int
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Replace a SAFE string's terminal fragment with a GenMol generation mask.

    ``[*{min-max}]`` asks GenMol to invent a fragment of between ``min`` and
    ``max`` tokens. Crucially it must *replace* a fragment rather than be tacked
    onto a complete one: fragments in a SAFE string are bonded to each other
    through shared ring-bond indices, so dropping the last fragment leaves those
    indices dangling and the mask has something to bond to. Appending the mask to
    an already-closed molecule leaves it disconnected, and GenMol responds by
    echoing the input back unchanged.

    Dropping the *last* fragment is the recipe from NVIDIA's own GenMol examples
    (``safe_str.rsplit(".", 1)[0] + ".[*{m-n}]"``). SAFE orders the core first, so
    the fragment that goes is a peripheral substituent.

    Args:
        safe_string: The scaffold in SAFE notation, already fragmented.
        frag_min: Minimum fragment size.
        frag_max: Maximum fragment size.

    Returns:
        Tuple of ``(masked_safe_string, replaced_fragment, error_message)``.
    """
    if frag_min < 1:
        return None, None, f"frag_min must be at least 1, got {frag_min}"
    if frag_max < frag_min:
        return None, None, f"frag_max ({frag_max}) must be >= frag_min ({frag_min})"

    fragments = safe_string.split(".")
    if len(fragments) < 2:
        return (
            None,
            None,
            "Cannot add a generation mask: this molecule has no fragmentation "
            "points, so there is no attachment point for a new fragment to bond "
            "to. Masking a single-fragment scaffold makes GenMol return the input "
            "molecule unchanged. Either pick a larger scaffold that can be "
            "fragmented, or generate de novo from a bare mask such as '[*{20-30}]'.",
        )

    mask = f"[*{{{frag_min}-{frag_max}}}]"
    return ".".join(fragments[:-1] + [mask]), fragments[-1], None
