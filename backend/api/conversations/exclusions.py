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

"""Rules derived from candidates that already failed.

The waste this removes is concrete. An agent enumerates twenty analogues, eight
fail because they all carry a catechol, and the next round contains six more
catechols -- because "catechol is disqualifying" was never written down anywhere
the agent still reads. The feature gets rediscovered instead of excluded.

Derived rather than stored. Every rule here is computed from the candidate ledger
on read, so there is no second table to keep in step with the first, and a rule
cannot outlive the evidence for it. Delete the candidate and the rule goes with
it, which is the correct behaviour and the kind that is hard to get wrong when
the derivation is the only definition.

**Only structural alerts name a feature.** A PAINS or Brenk hit *is* a
substructure match, so "this molecule failed" and "this substructure is
disqualifying" are the same statement. A property violation is not: molecular
weight 720 is a fact about the whole molecule, and nothing in the gate result
says which fragment carried the mass. Pretending otherwise would produce
confident, wrong exclusions -- so property rules are stated as the bound and the
margin already missed, which is true, rather than as a substructure, which would
not be.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# A rule needs more than one offender to be worth stating as a pattern. One
# failure is a fact about that molecule and is already in the ledger; the rule
# earns its place in the prompt by predicting the next failure.
MIN_OFFENDERS = 1

# Alert families whose hits name a substructure. Kept explicit so a new gate that
# happens to end in _alerts does not silently start producing exclusions.
ALERT_GATES = frozenset({"pains_alerts", "brenk_alerts", "nih_alerts"})


@dataclass
class Exclusion:
    """One reason later candidates are likely to fail, with its evidence.

    Attributes:
        kind: ``"alert"`` for a named substructure, ``"property"`` for a bound.
        feature: The alert description, or the gate name for a property.
        bound: The threshold that was violated, for property rules.
        offenders: Canonical SMILES of the candidates this disqualified.
        worst: The furthest-out value observed, for property rules.
    """

    kind: str
    feature: str
    bound: Optional[Any] = None
    offenders: list[str] = field(default_factory=list)
    worst: Optional[float] = None

    @property
    def count(self) -> int:
        return len(self.offenders)

    def describe(self) -> str:
        """One line, phrased as an instruction rather than a statistic."""
        molecules = "candidate" if self.count == 1 else "candidates"
        if self.kind == "alert":
            return (
                f"Avoid the {self.feature} substructure "
                f"({self.count} {molecules} disqualified)"
            )
        bound = f" {self.bound}" if self.bound is not None else ""
        worst = f", worst {_number(self.worst)}" if self.worst is not None else ""
        return (
            f"Keep {self.feature} within{bound} "
            f"({self.count} {molecules} outside it{worst})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "feature": self.feature,
            "bound": self.bound,
            "offenders": list(self.offenders),
            "worst": self.worst,
            "count": self.count,
        }


def _number(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, int):
        return str(value)
    return f"{value:.2f}".rstrip("0").rstrip(".")


def derive(candidates: list[dict[str, Any]]) -> list[Exclusion]:
    """Work out what has been disqualifying candidates, most prolific first.

    Args:
        candidates: Ledger rows. Only those that failed a gate contribute.

    Returns:
        Exclusions ordered by how many candidates each accounts for, so the
        feature costing the most enumerations is the one read first.
    """
    alerts: dict[str, Exclusion] = {}
    properties: dict[str, Exclusion] = {}

    for row in candidates:
        if row.get("passed") is not False:
            continue
        smiles = row.get("smiles") or ""
        for gate in row.get("gates") or []:
            if not isinstance(gate, dict) or gate.get("passed"):
                continue
            name = gate.get("name")
            if not name:
                continue

            if name in ALERT_GATES:
                for hit in gate.get("hits") or []:
                    rule = alerts.setdefault(hit, Exclusion("alert", str(hit)))
                    if smiles not in rule.offenders:
                        rule.offenders.append(smiles)
                continue

            rule = properties.setdefault(
                str(name), Exclusion("property", str(name), bound=gate.get("bound"))
            )
            if smiles not in rule.offenders:
                rule.offenders.append(smiles)
            rule.worst = _further_out(rule.worst, gate.get("value"), gate.get("bound"))

    rules = [
        rule
        for rule in (*alerts.values(), *properties.values())
        if rule.count >= MIN_OFFENDERS
    ]
    # Alerts before properties at equal count: an alert is actionable (remove the
    # group) where a property bound only says the molecule was too something.
    return sorted(
        rules,
        key=lambda rule: (
            -rule.count,
            0 if rule.kind == "alert" else 1,
            rule.feature,
        ),
    )


def _further_out(
    current: Optional[float], value: Any, bound: Any
) -> Optional[float]:
    """Keep whichever observed value misses the bound by more.

    Which direction counts as worse depends on the bound the gate reported, so it
    is read off the bound's own text (``<= 600`` versus ``>= 0.3``) rather than
    assumed. An unparseable bound keeps the largest magnitude, which is wrong for
    a lower bound but is only ever used to illustrate a rule, never to enforce
    one.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return current
    value = float(value)
    if current is None:
        return value

    text = str(bound or "")
    if ">" in text and "<" not in text:
        return min(current, value)
    return max(current, value, key=abs)


def screen(
    candidates: list[dict[str, Any]], smiles: str
) -> list[Exclusion]:
    """Which derived alert rules a molecule already known to trip.

    Only answers for molecules already in the ledger, because deciding whether an
    unseen SMILES contains a substructure needs RDKit, which the API image does
    not have. That is the honest boundary: this reports what was measured rather
    than guessing what would be.

    Args:
        candidates: Ledger rows.
        smiles: Canonical SMILES to look up.

    Returns:
        The exclusions this molecule is an offender for.
    """
    return [rule for rule in derive(candidates) if smiles in rule.offenders]
