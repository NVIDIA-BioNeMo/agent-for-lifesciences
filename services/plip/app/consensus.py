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

"""Folding interaction fingerprints across several structures.

A single predicted pose is one sample from a stochastic process. Boltz-2 emits
ten diffusion samples and Vina emits nine modes, and the interactions those
poses agree on are evidence about the binding mode, while the ones that appear
once are evidence about the sampler. Reporting a single pose's fingerprint as
"the interactions" conflates the two.

So the fold is deliberately simple and the discarded half is kept: an interaction
present in 9 of 10 poses and one present in 2 of 10 are both worth seeing, and
which side of the threshold each lands on is the answer.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

# Half the poses. Low enough to keep an interaction that a sampler only finds in
# the better-scoring half, high enough that a single pose cannot assert one.
DEFAULT_THRESHOLD = 0.5

# Cap on the reported below-threshold list. It exists to show what was
# stochastic, not to enumerate every fleeting contact in a large ensemble.
MAX_MINORITY_REPORTED = 40


@dataclass
class ConsensusEntry:
    """One interaction and how many structures showed it."""

    key: str
    chain: str
    residue_number: int
    residue_name: str
    family: str
    count: int
    total: int
    sources: list[str] = field(default_factory=list)

    @property
    def fraction(self) -> float:
        return round(self.count / self.total, 3) if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "chain": self.chain,
            "residue_number": self.residue_number,
            "residue_name": self.residue_name,
            "family": self.family,
            "count": self.count,
            "total": self.total,
            "fraction": self.fraction,
            "sources": self.sources,
        }


@dataclass
class Consensus:
    """The fold over a set of fingerprints."""

    total: int
    threshold: float
    conserved: list[ConsensusEntry] = field(default_factory=list)
    variable: list[ConsensusEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_structures": self.total,
            "threshold": self.threshold,
            "conserved": [entry.to_dict() for entry in self.conserved],
            "variable": [
                entry.to_dict() for entry in self.variable[:MAX_MINORITY_REPORTED]
            ],
            "num_variable": len(self.variable),
        }

    def residues(self) -> list[dict[str, Any]]:
        """Conserved interactions grouped by residue, most frequent first."""
        grouped: dict[tuple[str, int, str], dict[str, Any]] = {}
        for entry in self.conserved:
            key = (entry.chain, entry.residue_number, entry.residue_name)
            record = grouped.setdefault(
                key,
                {
                    "chain": entry.chain,
                    "residue_number": entry.residue_number,
                    "residue_name": entry.residue_name,
                    "families": [],
                    "min_fraction": 1.0,
                },
            )
            record["families"].append(entry.family)
            record["min_fraction"] = min(record["min_fraction"], entry.fraction)
        for record in grouped.values():
            record["families"] = sorted(record["families"])
        return sorted(
            grouped.values(),
            key=lambda record: (-record["min_fraction"], record["residue_number"]),
        )


def _parse(key: str) -> Optional[tuple[str, int, str, str]]:
    """Split a fingerprint key back into its parts."""
    parts = key.split(":")
    if len(parts) != 4:
        return None
    chain, number, name, family = parts
    try:
        return chain, int(number), name, family
    except ValueError:
        return None


def fold(
    fingerprints: dict[str, list[str]],
    threshold: float = DEFAULT_THRESHOLD,
) -> Consensus:
    """Count each interaction across structures and split on the threshold.

    Args:
        fingerprints: Structure name to its list of interaction keys. A
            structure with no interactions still counts toward the total --
            dropping it would inflate every fraction.
        threshold: Minimum fraction of structures for an interaction to be
            called conserved.

    Returns:
        A Consensus with conserved and variable interactions, each ordered by
        frequency and then residue number.
    """
    total = len(fingerprints)
    counts: dict[str, list[str]] = {}
    for name, keys in fingerprints.items():
        # De-duplicate within a structure: two hydrophobic contacts to the same
        # residue are one fact about the pose, not two votes.
        for key in set(keys):
            counts.setdefault(key, []).append(name)

    conserved: list[ConsensusEntry] = []
    variable: list[ConsensusEntry] = []

    for key, sources in counts.items():
        parsed = _parse(key)
        if parsed is None:
            continue
        chain, number, name, family = parsed
        entry = ConsensusEntry(
            key=key,
            chain=chain,
            residue_number=number,
            residue_name=name,
            family=family,
            count=len(sources),
            total=total,
            sources=sorted(sources),
        )
        if total and entry.fraction >= threshold:
            conserved.append(entry)
        else:
            variable.append(entry)

    def order(entry: ConsensusEntry) -> tuple[float, int, str]:
        return (-entry.fraction, entry.residue_number, entry.family)

    conserved.sort(key=order)
    variable.sort(key=order)

    return Consensus(
        total=total,
        threshold=threshold,
        conserved=conserved,
        variable=variable,
    )
