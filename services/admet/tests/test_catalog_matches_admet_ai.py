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

"""Pin the catalog to ADMET-AI's own property table.

``properties.py`` is a hand-written overlay: it adds direction, aliases, short
labels and notes on top of names and units that belong to ADMET-AI. Two ways that
drifts, both silent. A property renamed or added by an ADMET-AI upgrade simply
stops being requestable, because ``properties=['all']`` enumerates our catalog
rather than the model's columns. And a units string that disagrees -- ``Da``
against ``Dalton`` -- puts the wrong unit next to a number in a chemist's report.

So this compares the two directly against ``admet_ai/resources/data/admet.csv``,
which is the table the package's own web UI renders from. It only runs inside the
service container; the catalog suite next to it is what runs everywhere else.
"""

import csv

import pytest

from app.properties import BY_NAME, PROPERTIES, TaskType

admet_ai = pytest.importorskip(
    "admet_ai", reason="only runs where the ADMET-AI package is installed"
)


def package_catalog() -> dict[str, dict[str, str]]:
    """ADMET-AI's shipped property table, keyed by property id."""
    from admet_ai.constants import DEFAULT_ADMET_PATH

    with open(str(DEFAULT_ADMET_PATH), newline="") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def normalize_units(units: str) -> str:
    """The package writes ``-`` where a value is dimensionless; we write ``''``."""
    return "" if units.strip() in ("-", "") else units.strip()


@pytest.fixture(scope="module")
def package():
    return package_catalog()


def test_the_catalog_covers_every_property_the_package_defines(package):
    missing = sorted(set(package) - set(BY_NAME))
    assert not missing, (
        f"ADMET-AI defines properties absent from our catalog: {missing}. "
        "Add them, or they cannot be requested by name and describe() will "
        "report them as unknown."
    )


def test_the_catalog_invents_nothing(package):
    extra = sorted(set(BY_NAME) - set(package))
    assert not extra, (
        f"our catalog lists properties ADMET-AI does not produce: {extra}"
    )


def test_units_agree_with_the_package(package):
    problems = [
        f"{prop.name}: ours={prop.units!r} package="
        f"{normalize_units(package[prop.name]['units'])!r}"
        for prop in PROPERTIES
        if prop.name in package
        and prop.units != normalize_units(package[prop.name]["units"])
    ]
    assert not problems, "units drift: " + "; ".join(problems)


def test_task_types_agree_with_the_package(package):
    problems = [
        f"{prop.name}: ours={prop.task_type.value} "
        f"package={package[prop.name]['task_type']}"
        for prop in PROPERTIES
        if prop.name in package
        and prop.task_type.value != package[prop.name]["task_type"]
    ]
    assert not problems, "task type drift: " + "; ".join(problems)


def test_categories_agree_with_the_package(package):
    problems = [
        f"{prop.name}: ours={prop.category} package={package[prop.name]['category']}"
        for prop in PROPERTIES
        if prop.name in package and prop.category != package[prop.name]["category"]
    ]
    assert not problems, "category drift: " + "; ".join(problems)


def test_classification_bounds_are_probabilities(package):
    """A property we call a classification must be one the package bounds 0..1.

    This is what makes a threshold like "hERG below 0.3 is clean" meaningful.
    """
    for prop in PROPERTIES:
        if prop.task_type is not TaskType.CLASSIFICATION:
            continue
        row = package[prop.name]
        assert (row["minimum"], row["maximum"]) == ("0.0", "1.0"), prop.name
