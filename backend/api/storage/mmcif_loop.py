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

"""Just enough mmCIF to read the ``_atom_site`` loop.

Aevom has no mmCIF library in the backend image, and two things here need the
same small slice of one: recognising an annotated CIF by its condition
columns, and mapping author residue numbering onto label numbering. Both want
the ``_atom_site`` header block and its rows, and nothing else.

This is deliberately not a parser. It does not handle semicolon-delimited
multi-line values, save frames, or any category but ``_atom_site`` -- none of
which occur in the rows we read. Anything beyond that belongs in a real
library, in the RDKit service, not here.
"""

from __future__ import annotations

from typing import Iterator


def as_text(content: str | bytes) -> str:
    """Decode to text, returning empty string for anything undecodable."""
    if isinstance(content, bytes):
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return ""
    return content or ""


def tokenize_row(line: str) -> list[str]:
    """Split one mmCIF loop row into values, honouring quoted strings.

    mmCIF quotes any value containing whitespace with single or double quotes,
    which is how an atom name like ``'O5''`` survives. Values are otherwise
    whitespace-separated with no fixed columns.
    """
    tokens: list[str] = []
    rest = line.strip()
    while rest:
        if rest[0] in "'\"":
            quote = rest[0]
            end = rest.find(quote, 1)
            if end == -1:
                tokens.append(rest[1:])
                break
            tokens.append(rest[1:end])
            rest = rest[end + 1 :].lstrip()
        else:
            token, _, rest = rest.partition(" ")
            tokens.append(token)
            rest = rest.lstrip()
    return tokens


def atom_site_columns(text: str) -> list[str]:
    """Return the ``_atom_site`` column names, in file order.

    Names come back without the category prefix, so ``_atom_site.label_seq_id``
    is reported as ``label_seq_id``.
    """
    columns: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("_atom_site."):
            columns.append(stripped.split(".", 1)[1].split()[0])
        elif columns and not stripped.startswith("_"):
            # The header block is contiguous; the first non-tag line ends it.
            break
    return columns


def iter_atom_site_rows(text: str) -> Iterator[list[str]]:
    """Yield each ``_atom_site`` row as a list of values.

    Only rows of the ``_atom_site`` loop itself are read: from the line after
    its header block to the first terminator (``#``, ``loop_``, ``data_`` or
    another tag). A full PDB entry has other loops whose rows can have exactly
    as many fields as ``_atom_site`` has columns -- ``_struct_conn`` in 5O45
    does -- and reading those as atoms is worse than reading nothing.

    Rows whose field count does not match the header are still skipped: a
    wrapped line announces itself that way.
    """
    columns: list[str] = []
    in_rows = False
    width = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not in_rows:
            if stripped.startswith("_atom_site."):
                columns.append(stripped.split(".", 1)[1].split()[0])
            elif columns and stripped and not stripped.startswith("_"):
                # First non-tag line after the header is the first row.
                in_rows = True
                width = len(columns)
            else:
                continue
            if not in_rows:
                continue
        if not stripped:
            continue
        if stripped.startswith(("_", "#", "loop_", "data_")):
            return
        values = tokenize_row(stripped)
        if len(values) == width:
            yield values
