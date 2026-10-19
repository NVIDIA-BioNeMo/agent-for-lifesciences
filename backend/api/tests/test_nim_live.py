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

"""Checks against a real RFD4-Proteina NIM, skipped when none is running.

The mocked tests elsewhere pin what Aevom *believes* about this NIM. These
pin what it actually does, and exist because the published documentation was
wrong about the task enum -- it lists a task named ``generation`` that the
live service does not have.

What they deliberately do not do is run a design. The model needs an A100,
H100 or H200; on anything older every request dies with
``cudaErrorNoKernelImageForDevice`` after the payload has been fully
accepted. So the strongest assertion available without that hardware is that
a request gets *past* validation: schema, contig parsing, selection
resolution and condition application all succeed, and the failure that
remains is CUDA's. A 422 would mean Aevom built the request wrongly; a 500
naming an accelerator error means it built it correctly.

Point AEVOM_TEST_NIM_URL at a capable host to run these somewhere the model
can actually execute.
"""

import json
import os

import httpx
import pytest

from api.nim.client import NIMClient
from api.storage.mmcif_numbering import build_numbering_map
from api.tools.rfd4_proteina import (
    build_condition_spec,
    build_contig,
    build_hotspot_spec,
)

NIM_URL = os.environ.get("AEVOM_TEST_NIM_URL", "http://localhost:8000")


def _nim_is_up() -> bool:
    try:
        response = httpx.get(f"{NIM_URL}/v1/models", timeout=2.0)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


requires_nim = pytest.mark.skipif(
    not _nim_is_up(), reason=f"no NIM reachable at {NIM_URL}"
)

# A two-residue target is enough to exercise parsing and conditioning without
# shipping a real structure into the test suite. Author numbering runs 16
# ahead of label numbering, as it does in 5O45.
TARGET_CIF = """\
data_TINY
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM 1 N N  . TYR A 1 1 ? 0.000 0.000 0.000 1.00 0.00 17 TYR A N  1
ATOM 2 C CA . TYR A 1 1 ? 1.458 0.000 0.000 1.00 0.00 17 TYR A CA 1
ATOM 3 C C  . TYR A 1 1 ? 2.009 1.420 0.000 1.00 0.00 17 TYR A C  1
ATOM 4 O O  . TYR A 1 1 ? 1.251 2.390 0.000 1.00 0.00 17 TYR A O  1
ATOM 5 C CB . TYR A 1 1 ? 2.000 -0.780 1.200 1.00 0.00 17 TYR A CB 1
ATOM 6 N N  . ALA A 1 2 ? 3.332 1.550 0.000 1.00 0.00 18 ALA A N  1
ATOM 7 C CA . ALA A 1 2 ? 3.993 2.850 0.000 1.00 0.00 18 ALA A CA 1
ATOM 8 C C  . ALA A 1 2 ? 5.500 2.700 0.000 1.00 0.00 18 ALA A C  1
ATOM 9 O O  . ALA A 1 2 ? 6.000 1.580 0.000 1.00 0.00 18 ALA A O  1
#
"""


@requires_nim
@pytest.mark.asyncio
class TestLiveDiscovery:
    async def test_probe_identifies_the_model(self):
        info = await NIMClient(NIM_URL).probe()
        assert info.model_id
        assert info.ready is True

    async def test_task_enum_matches_the_service_not_the_docs(self):
        # The docs name a "generation" task. The service does not have one,
        # which is why the adapter reads /v1/tasks rather than hardcoding.
        info = await NIMClient(NIM_URL).probe()
        if info.tasks:
            assert "generation" not in info.tasks
            assert "unconditional" in info.tasks

    async def test_unavailable_metric_tags_are_reported_as_such(self):
        info = await NIMClient(NIM_URL).probe()
        if info.metric_tags:
            # Something must be available, or the tool would request nothing.
            assert info.available_metric_tags


@requires_nim
class TestLiveRequestAcceptance:
    """The request Aevom builds is one this NIM accepts."""

    @staticmethod
    def _request(**overrides):
        numbering = build_numbering_map(TARGET_CIF)
        selection = {
            "atoms": [
                {
                    "name": "CB",
                    "element": "C",
                    "id": 5,
                    "coords": [2.0, -0.78, 1.2],
                    "residue": {
                        "name": "TYR",
                        "seqNumber": 17,
                        "chainId": "A",
                        "insCode": "",
                        "entityType": "polymer",
                    },
                }
            ]
        }
        hotspots = build_hotspot_spec(selection, numbering)
        request = build_condition_spec(
            example_id="aevom_live_test",
            contig=build_contig(numbering, hotspots.chains, 20, 20),
            target_chains=hotspots.chains,
            hotspot_selectors=hotspots.selectors,
            target_cif=TARGET_CIF,
            num_designs=1,
            # The checkpoint refuses flat sampling below 20 steps; this is the
            # cheapest request it will actually accept.
            num_steps=20,
        )
        request.update(overrides)
        return request

    @staticmethod
    def _post(request):
        return httpx.post(
            f"{NIM_URL}/biology/ipd/rfd4-proteina/generative",
            json=request,
            timeout=300.0,
        )

    @staticmethod
    def _error_code(response) -> str:
        try:
            return str((response.json().get("error") or {}).get("code") or "")
        except ValueError:
            return ""

    def test_the_spec_passes_validation(self):
        response = self._post(self._request())

        if response.status_code == 200:
            # A capable GPU: the design actually ran.
            assert response.json()["designs"]
            return

        body = response.text.lower()
        assert response.status_code != 422, (
            f"the NIM rejected Aevom's request as malformed: {response.text[:500]}"
        )
        # Everything above CUDA succeeded; only the GPU is wrong.
        assert "cuda" in body or "accelerator" in body, response.text[:500]

    def test_a_bad_task_is_still_rejected(self):
        # Proves the check above is meaningful: this request *is* validated,
        # so reaching CUDA was not the service skipping validation.
        response = self._post(self._request(task="not_a_real_task"))
        assert response.status_code == 422

    def test_a_bad_contig_is_rejected(self):
        request = self._request()
        request["input"]["spec"]["contig"] = "this is not a contig"
        response = self._post(request)
        assert response.status_code >= 400
        assert self._error_code(response) != "internal_error" or "cuda" not in (
            response.text.lower()
        ), "a malformed contig should fail before the GPU is reached"

    def test_pdb_input_is_refused(self):
        # The NIM takes mmCIF only, which is why the tool checks before
        # calling rather than letting the user discover it here.
        request = self._request()
        request["input"]["files"][0]["content"] = (
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n"
        )
        response = self._post(request)
        assert response.status_code >= 400

    def test_error_bodies_carry_a_branchable_code(self):
        # The tool's actionable advice is keyed on this code, so its presence
        # is part of the contract rather than a nicety.
        response = self._post(self._request(task="not_a_real_task"))
        payload = response.json()
        assert "error" in payload
        assert payload["error"].get("code")
        assert json.dumps(payload)


def _viewer_atoms(seq: int, comp: str, names: tuple[str, ...]) -> list[dict]:
    """Atoms as the viewer posts them: label ids plus author ids."""
    return [
        {
            "name": name,
            "element": name[0],
            "id": seq * 10 + i,
            "coords": [0.0, 0.0, 0.0],
            "residue": {
                "name": comp,
                "seqNumber": seq,
                "chainId": "A",
                "insCode": "",
                "entityType": "polymer",
                "authSeqNumber": seq + 16,
                "authChainId": "A",
            },
        }
        for i, name in enumerate(names)
    ]


_BACKBONE = ("N", "CA", "C", "O", "CB")
_TYR = {"atoms": _viewer_atoms(1, "TYR", _BACKBONE)}
_ALA = {"atoms": _viewer_atoms(2, "ALA", _BACKBONE)}
_BOTH = {"atoms": _TYR["atoms"] + _ALA["atoms"]}
_TYR_CB = {"atoms": _viewer_atoms(1, "TYR", ("CB",))}
_TYR_N = {"atoms": _viewer_atoms(1, "TYR", ("N",))}


def _plan(mode, *layers, layout=None):
    from api.selections.design_spec import Layout, SpecLayer
    from api.selections.role_compile import LayerInput, compile_rfd4

    inputs = [
        LayerInput(layer=SpecLayer(cond, f"selection_{i}", value), selection=sel)
        for i, (cond, sel, value) in enumerate(layers)
    ]
    plan = compile_rfd4(
        inputs, build_numbering_map(TARGET_CIF), mode=mode, binder_min=20, binder_max=20,
        layout=layout or Layout(),
    )
    assert plan.ok, plan.errors
    return plan


@requires_nim
class TestLiveCompiledRoles:
    """What the role compiler builds is accepted by this NIM.

    The same strongest-available check as above: past validation (not a 422,
    no enum or condition error), stopping only at the GPU when it is too old.
    """

    @staticmethod
    def _send(plan, task):
        request = build_condition_spec(
            example_id="aevom_live_roles",
            contig=plan.contig,
            target_cif=TARGET_CIF,
            num_designs=1,
            num_steps=20,
            conditions=plan.conditions,
            task=task,
        )
        return httpx.post(
            f"{NIM_URL}/biology/ipd/rfd4-proteina/generative", json=request, timeout=300.0
        )

    @staticmethod
    def _assert_accepted(response, plan):
        if response.status_code == 200:
            assert response.json()["designs"]
            return
        body = response.text.lower()
        sent = f"contig={plan.contig!r} conditions={json.dumps(plan.conditions)}"
        assert response.status_code != 422, f"rejected as malformed: {response.text[:500]}\n{sent}"
        assert "has no member" not in body and "valueerror" not in body, (
            f"a condition value was not understood: {response.text[:500]}\n{sent}"
        )
        assert "cuda" in body or "accelerator" in body, f"{response.text[:500]}\n{sent}"

    def test_a_binder_with_keep_hotspot_and_fixed_sequence(self):
        plan = _plan(
            "binder",
            ("hotspot", _TYR_CB, None),
            ("keep", _BOTH, None),
            ("fix_sequence", _TYR, None),
        )
        self._assert_accepted(self._send(plan, "unconditional"), plan)

    def test_a_motif_with_a_layout_and_a_float_group(self):
        from api.selections.design_spec import Layout

        plan = _plan(
            "motif",
            ("keep", _BOTH, None),
            ("fix_sequence", _BOTH, None),
            ("float", _BOTH, "motif"),
            layout=Layout(n_term=(10, 12), c_term=(10, 12), linkers=((8, 10),)),
        )
        self._assert_accepted(self._send(plan, "motif_scaffolding"), plan)

    def test_a_motif_that_holds_its_position(self):
        plan = _plan(
            "motif",
            ("keep", _BOTH, None),
            ("fix_position", _BOTH, None),
            ("fix_sequence", _BOTH, None),
        )
        self._assert_accepted(self._send(plan, "motif_scaffolding"), plan)

    def test_chemistry_and_secondary_structure_conditions(self):
        plan = _plan(
            "binder",
            ("hotspot", _TYR_CB, None),
            ("keep", _BOTH, None),
            ("hbond", _TYR_N, "DONOR"),
            ("rasa", _TYR, "BURIED"),
            ("secondary_structure", _ALA, "HELIX"),
        )
        for condition in ("C_HB", "C_RAS", "C_SEC"):
            assert condition in plan.conditions
        self._assert_accepted(self._send(plan, "unconditional"), plan)

    def test_a_bad_condition_value_is_still_caught(self):
        # Proves the checks above are meaningful: the NIM does validate these,
        # so reaching CUDA with the real names was not it skipping validation.
        plan = _plan("binder", ("hotspot", _TYR_CB, None), ("keep", _BOTH, None))
        plan.conditions["C_SEC"] = {"STRAND": [{"select": "sel('A/*/2')"}]}
        response = self._send(plan, "unconditional")
        body = response.text.lower()
        assert response.status_code >= 400
        assert response.status_code == 422 or "member" in body or "strand" in body


# TYR-ALA are bonded; GLY is 8 A further on, as if residues between were
# unmodelled, while the label ids stay consecutive (what a PDB converted to
# mmCIF looks like).
_GLY_ROWS = (
    "ATOM 10 N N  . GLY A 1 3 ? 14.000 2.000 0.000 1.00 0.00 19 GLY A N  1\n"
    "ATOM 11 C CA . GLY A 1 3 ? 15.000 2.000 0.000 1.00 0.00 19 GLY A CA 1\n"
    "ATOM 12 C C  . GLY A 1 3 ? 16.000 2.000 0.000 1.00 0.00 19 GLY A C  1\n"
    "ATOM 13 O O  . GLY A 1 3 ? 17.000 2.000 0.000 1.00 0.00 19 GLY A O  1\n"
)
BROKEN_CIF = TARGET_CIF[: -len("#\n")] + _GLY_ROWS + "#\n"


@requires_nim
class TestLiveBrokenTarget:
    """A chain with unmodelled residues is not bonded across, which the NIM checks."""

    @staticmethod
    def _post(contig, conditions):
        request = build_condition_spec(
            example_id="aevom_live_broken",
            contig=contig,
            target_cif=BROKEN_CIF,
            num_designs=1,
            num_steps=20,
            conditions=conditions,
        )
        return httpx.post(
            f"{NIM_URL}/biology/ipd/rfd4-proteina/generative", json=request, timeout=300.0
        )

    def test_bonding_across_the_break_is_what_the_nim_rejects(self):
        plan = _plan_on(BROKEN_CIF, ("hotspot", _TYR_CB, None))
        response = self._post("20, /0, A1-3", plan.conditions)
        assert response.status_code == 500
        assert "bond_length" in response.text, response.text[:400]

    def test_the_compiled_target_stops_at_the_break_and_is_accepted(self):
        plan = _plan_on(BROKEN_CIF, ("hotspot", _TYR_CB, None))
        assert plan.contig == "20, /0, A1-2"
        assert any("broken" in w for w in plan.warnings)
        response = self._post(plan.contig, plan.conditions)
        if response.status_code == 200:
            return
        body = response.text.lower()
        assert "bond_length" not in body and response.status_code != 422, response.text[:400]
        assert "cuda" in body or "accelerator" in body, response.text[:400]


def _plan_on(cif, *layers):
    from api.selections.design_spec import SpecLayer
    from api.selections.role_compile import LayerInput, compile_rfd4

    inputs = [
        LayerInput(layer=SpecLayer(cond, f"selection_{i}", value), selection=sel)
        for i, (cond, sel, value) in enumerate(layers)
    ]
    plan = compile_rfd4(
        inputs, build_numbering_map(cif), mode="binder", binder_min=20, binder_max=20
    )
    assert plan.ok, plan.errors
    return plan
