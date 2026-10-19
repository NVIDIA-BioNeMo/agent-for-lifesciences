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

"""Tests for the search_rcsb tool.

The tool is a proxy over the RCSB Search API, so most of what is worth testing
is the boundary: what it sends, what it does with each response shape, and what
it refuses to do. Both RCSB endpoints are served by an httpx.MockTransport, so
nothing here touches the network.

The behaviour these tests pin down is the reason the tool was rewritten: it must
add no filters of its own, must not drop entries that have no resolution, and
must report the experimental-method breakdown so a narrow query is visible as
narrow.
"""

import json

import httpx
import pytest

from api.tools import rcsb
from api.tools.base import ContentType
from api.tools.rcsb import DEFAULT_ROWS, MAX_ROWS, SearchRcsbTool

GENE_QUERY = {
    "type": "terminal",
    "service": "text",
    "parameters": {
        "attribute": "rcsb_entity_source_organism.rcsb_gene_name.value",
        "operator": "exact_match",
        "value": "TRPV1",
    },
}


def _entry(
    pdb_id: str,
    *,
    method: str = "EM",
    resolution: list[float] | None = None,
    title: str = "A structure",
    ligands: tuple[tuple[str, str, float], ...] = (),
    affinity: list[dict] | None = None,
    methodology: str = "experimental",
) -> dict:
    """Build a Data API entry in the shape the GraphQL endpoint returns."""
    return {
        "rcsb_id": pdb_id,
        "struct": {"title": title},
        "rcsb_entry_info": {
            "resolution_combined": resolution,
            "experimental_method": method,
            "structure_determination_methodology": methodology,
        },
        "rcsb_accession_info": {"initial_release_date": "2024-05-08T00:00:00Z"},
        "nonpolymer_entities": [
            {"nonpolymer_comp": {"chem_comp": {"id": lid, "name": name, "formula_weight": mw}}}
            for lid, name, mw in ligands
        ],
        "rcsb_binding_affinity": affinity,
    }


class _Recorder:
    """Serves both RCSB endpoints and records the payloads sent to them."""

    def __init__(self, search_response, entries=None):
        self.search_response = search_response
        self.entries = entries if entries is not None else []
        self.search_payloads: list[dict] = []
        self.graphql_payloads: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content) if request.content else {}

        if "rcsbsearch" in str(request.url):
            self.search_payloads.append(payload)
            if isinstance(self.search_response, httpx.Response):
                return self.search_response
            return httpx.Response(200, json=self.search_response)

        self.graphql_payloads.append(payload)
        return httpx.Response(200, json={"data": {"entries": self.entries}})

    @property
    def search_payload(self) -> dict:
        assert self.search_payloads, "no search request was made"
        return self.search_payloads[0]

    @property
    def request_options(self) -> dict:
        return self.search_payload["request_options"]


@pytest.fixture
def serve_handler(monkeypatch):
    """Route every client the tool builds through a MockTransport.

    The real class is captured before patching: the factory replaces
    `rcsb.httpx.AsyncClient`, so constructing one by that name would recurse.
    """
    real_client = httpx.AsyncClient

    def install(handler):
        def factory(**kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            return real_client(**kwargs)

        monkeypatch.setattr(rcsb.httpx, "AsyncClient", factory)

    return install


@pytest.fixture
def serve(serve_handler):
    """Install a mock transport and hand back the recorder for assertions."""

    def install(search_response, entries=None) -> _Recorder:
        recorder = _Recorder(search_response, entries)
        serve_handler(recorder)
        return recorder

    return install


@pytest.fixture
def tool() -> SearchRcsbTool:
    return SearchRcsbTool()


def _hits(result) -> list[dict]:
    """Pull the hit list back out of the model-facing text.

    Hits are rendered as one compact JSON object per line, so this also acts as
    a check that each line really is parseable on its own.
    """
    body = result.data.split("One JSON object per line:\n", 1)[1]
    return [json.loads(line) for line in body.split("\n\n", 1)[0].splitlines() if line.strip()]


class TestSchema:
    def test_schema_shape(self, tool):
        schema = tool.get_schema()
        assert schema["type"] == "function"

        func = schema["function"]
        assert func["name"] == "search_rcsb"

        params = func["parameters"]
        assert set(params["properties"]) == {
            "query",
            "return_type",
            "request_options",
            "find_attributes",
        }
        # Neither query nor find_attributes is required on its own; execute()
        # enforces that one of the two is present.
        assert params["required"] == []

    def test_description_warns_against_filtering_by_method(self, tool):
        """The cryo-EM exclusion was the bug; the description has to say so."""
        description = tool.get_schema()["function"]["description"].lower()
        assert "cryo-em" in description
        assert "rcsb_entry_info.experimental_method" in description

    def test_description_does_not_promise_quality_filters(self, tool):
        """The old tool advertised 'resolution <= 3.0, X-ray only'."""
        description = tool.get_schema()["function"]["description"]
        assert "X-RAY DIFFRACTION" in description, "still listed as a *choice*"
        assert "best_holo" not in description
        assert "best_apo" not in description

    def test_summary_names_the_search_target(self, tool):
        assert tool.format_summary({"query": GENE_QUERY}) == "Searching RCSB PDB for TRPV1..."

    def test_summary_survives_an_unrecognizable_query(self, tool):
        """format_summary must never leak a literal {placeholder} to the UI."""
        for arguments in ({}, {"query": {"type": "group", "nodes": []}}):
            summary = tool.format_summary(arguments)
            assert summary == "Searching RCSB PDB..."
            assert "{" not in summary


class TestQueryPassthrough:
    @pytest.mark.asyncio
    async def test_query_is_sent_verbatim(self, tool, serve):
        """No filters may be added: that was the whole defect."""
        recorder = serve({"total_count": 0, "result_set": []})

        await tool.execute({"query": GENE_QUERY})

        assert recorder.search_payload["query"] == GENE_QUERY
        assert recorder.search_payload["return_type"] == "entry"

    @pytest.mark.asyncio
    async def test_sort_and_content_type_pass_through(self, tool, serve):
        options = {
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
            "results_content_type": ["experimental", "computational"],
        }
        recorder = serve({"total_count": 0, "result_set": []})

        await tool.execute({"query": GENE_QUERY, "request_options": options})

        assert recorder.request_options["sort"] == options["sort"]
        assert recorder.request_options["results_content_type"] == options["results_content_type"]

    @pytest.mark.asyncio
    async def test_caller_arguments_are_not_mutated(self, tool, serve):
        """The guards deep-copy, so the caller's dict is left alone."""
        options = {"paginate": {"rows": 500}, "return_all_hits": True}
        serve({"total_count": 0, "result_set": []})

        await tool.execute({"query": GENE_QUERY, "request_options": options})

        assert options == {"paginate": {"rows": 500}, "return_all_hits": True}

    @pytest.mark.asyncio
    async def test_query_is_required(self, tool, serve):
        recorder = serve({"total_count": 0, "result_set": []})

        result = await tool.execute({})

        assert result.content_type == ContentType.ERROR
        assert "query" in result.data
        assert not recorder.search_payloads

    @pytest.mark.asyncio
    async def test_unknown_return_type_is_rejected_locally(self, tool, serve):
        recorder = serve({"total_count": 0, "result_set": []})

        result = await tool.execute({"query": GENE_QUERY, "return_type": "galaxy"})

        assert result.content_type == ContentType.ERROR
        assert "galaxy" in result.data
        assert not recorder.search_payloads


class TestRequestGuards:
    @pytest.mark.parametrize(
        "requested,expected",
        [(500, MAX_ROWS), (10_000, MAX_ROWS), (5, 5), (0, DEFAULT_ROWS), ("junk", DEFAULT_ROWS)],
    )
    @pytest.mark.asyncio
    async def test_rows_are_clamped(self, tool, serve, requested, expected):
        recorder = serve({"total_count": 0, "result_set": []})

        await tool.execute(
            {"query": GENE_QUERY, "request_options": {"paginate": {"rows": requested}}}
        )

        assert recorder.request_options["paginate"]["rows"] == expected

    @pytest.mark.asyncio
    async def test_rows_default_when_unspecified(self, tool, serve):
        recorder = serve({"total_count": 0, "result_set": []})

        await tool.execute({"query": GENE_QUERY})

        assert recorder.request_options["paginate"]["rows"] == DEFAULT_ROWS

    @pytest.mark.asyncio
    async def test_return_all_hits_is_stripped(self, tool, serve):
        recorder = serve({"total_count": 0, "result_set": []})

        await tool.execute({"query": GENE_QUERY, "request_options": {"return_all_hits": True}})

        assert "return_all_hits" not in recorder.request_options

    @pytest.mark.asyncio
    async def test_start_is_preserved_so_the_agent_can_page(self, tool, serve):
        recorder = serve({"total_count": 0, "result_set": []})

        await tool.execute(
            {"query": GENE_QUERY, "request_options": {"paginate": {"start": 40, "rows": 5}}}
        )

        assert recorder.request_options["paginate"]["start"] == 40

    @pytest.mark.asyncio
    async def test_method_facet_is_injected_when_absent(self, tool, serve):
        """Without this the agent cannot see what its filters excluded."""
        recorder = serve({"total_count": 0, "result_set": []})

        await tool.execute({"query": GENE_QUERY})

        assert recorder.request_options["facets"] == [
            {
                "name": "experimental_method",
                "aggregation_type": "terms",
                "attribute": "rcsb_entry_info.experimental_method",
            }
        ]

    @pytest.mark.asyncio
    async def test_agent_supplied_facets_are_not_overridden(self, tool, serve):
        facets = [
            {"name": "year", "aggregation_type": "terms", "attribute": "rcsb_accession_info.initial_release_date"}
        ]
        recorder = serve({"total_count": 0, "result_set": []})

        await tool.execute({"query": GENE_QUERY, "request_options": {"facets": facets}})

        assert recorder.request_options["facets"] == facets

    @pytest.mark.asyncio
    async def test_verbosity_forced_compact_for_flat_results(self, tool, serve):
        recorder = serve({"total_count": 0, "result_set": []})

        await tool.execute(
            {"query": GENE_QUERY, "request_options": {"results_verbosity": "verbose"}}
        )

        assert recorder.request_options["results_verbosity"] == "compact"

    @pytest.mark.asyncio
    async def test_verbosity_left_alone_for_grouped_results(self, tool, serve):
        """Compact verbosity would discard the group members that were asked for."""
        recorder = serve({"total_count": 0, "group_set": [], "group_by_count": 0})

        await tool.execute(
            {
                "query": GENE_QUERY,
                "request_options": {
                    "group_by": {"aggregation_method": "sequence_identity", "similarity_cutoff": 90},
                    "group_by_return_type": "groups",
                },
            }
        )

        assert "results_verbosity" not in recorder.request_options


class TestResponseHandling:
    @pytest.mark.asyncio
    async def test_no_hits_is_text_not_an_error(self, tool, serve):
        """HTTP 204 means the query ran and matched nothing."""
        serve(httpx.Response(204))

        result = await tool.execute({"query": GENE_QUERY})

        assert result.content_type == ContentType.TEXT
        assert "total_count: 0" in result.data

    @pytest.mark.asyncio
    async def test_no_hits_names_the_likely_culprit(self, tool, serve):
        serve(httpx.Response(204))

        result = await tool.execute({"query": GENE_QUERY})

        assert "resolution" in result.data
        assert "method" in result.data

    @pytest.mark.asyncio
    async def test_rcsb_rejection_message_is_relayed_verbatim(self, tool, serve):
        """RCSB names the bad attribute, which is what lets the agent retry."""
        message = "search is not enabled on [ rcsb_entry_info.bogus ] attribute"
        serve(httpx.Response(400, json={"status": 400, "message": message}))

        result = await tool.execute({"query": GENE_QUERY})

        assert result.content_type == ContentType.ERROR
        assert message in result.data

    @pytest.mark.asyncio
    async def test_transport_failure_is_reported_as_unreachable(self, tool, serve_handler):
        def explode(request):
            raise httpx.ConnectError("no route to host")

        serve_handler(explode)

        result = await tool.execute({"query": GENE_QUERY})

        assert result.content_type == ContentType.ERROR
        assert "Search API" in result.data

    @pytest.mark.asyncio
    async def test_return_counts_reports_only_the_total(self, tool, serve):
        recorder = serve({"total_count": 84})

        result = await tool.execute(
            {"query": GENE_QUERY, "request_options": {"return_counts": True}}
        )

        assert "total_count: 84" in result.data
        assert result.metadata["total_count"] == 84
        # Counts-only responses carry no hits, so no enrichment is attempted.
        assert not recorder.graphql_payloads

    @pytest.mark.asyncio
    async def test_empty_page_past_the_end_explains_itself(self, tool, serve):
        serve({"total_count": 84, "result_set": []})

        result = await tool.execute(
            {"query": GENE_QUERY, "request_options": {"paginate": {"start": 900}}}
        )

        assert result.content_type == ContentType.TEXT
        assert "paginate.start" in result.data


class TestEnrichment:
    @pytest.mark.asyncio
    async def test_hits_carry_per_entry_facts(self, tool, serve):
        serve(
            {"total_count": 1, "result_set": ["8U4D"]},
            entries=[
                _entry(
                    "8U4D",
                    method="EM",
                    resolution=[2.2],
                    title="TRPV1 in nanodisc",
                    ligands=(("VPN", "a phospholipid", 1182.7),),
                )
            ],
        )

        result = await tool.execute({"query": GENE_QUERY})

        (hit,) = _hits(result)
        assert hit["pdb_id"] == "8U4D"
        assert hit["experimental_method"] == "EM"
        assert hit["resolution"] == 2.2
        assert hit["released"] == "2024-05-08"
        assert hit["is_holo"] is True
        assert hit["ligands"] == [
            {"id": "VPN", "name": "a phospholipid", "formula_weight": 1182.7}
        ]

    @pytest.mark.asyncio
    async def test_entries_without_resolution_are_kept(self, tool, serve):
        """The old code did `if resolution is None: continue`, losing all NMR."""
        serve(
            {"total_count": 1, "result_set": ["2LMT"]},
            entries=[_entry("2LMT", method="NMR", resolution=None)],
        )

        result = await tool.execute({"query": GENE_QUERY})

        (hit,) = _hits(result)
        assert hit["pdb_id"] == "2LMT"
        assert hit["experimental_method"] == "NMR"
        assert hit["resolution"] is None

    @pytest.mark.asyncio
    async def test_search_order_survives_the_data_api(self, tool, serve):
        """GraphQL answers in its own order, so hits are re-keyed by ID."""
        serve(
            {"total_count": 3, "result_set": ["3SUI", "11CL", "8U4D"]},
            entries=[_entry("8U4D"), _entry("3SUI"), _entry("11CL")],
        )

        result = await tool.execute({"query": GENE_QUERY})

        assert [hit["pdb_id"] for hit in _hits(result)] == ["3SUI", "11CL", "8U4D"]

    @pytest.mark.asyncio
    async def test_ions_and_buffers_do_not_make_a_structure_holo(self, tool, serve):
        serve(
            {"total_count": 1, "result_set": ["3SUI"]},
            entries=[
                _entry(
                    "3SUI",
                    method="X-ray",
                    resolution=[1.95],
                    ligands=(("CA", "CALCIUM ION", 40.078), ("SO4", "SULFATE ION", 96.063)),
                )
            ],
        )

        result = await tool.execute({"query": GENE_QUERY})

        (hit,) = _hits(result)
        assert hit["is_holo"] is False
        assert hit["ligands"] == []

    @pytest.mark.asyncio
    async def test_binding_affinity_is_included_when_present(self, tool, serve):
        serve(
            {"total_count": 1, "result_set": ["8GFA"]},
            entries=[
                _entry(
                    "8GFA",
                    ligands=(("ZEI", "an antagonist", 287.7),),
                    affinity=[{"comp_id": "ZEI", "type": "IC50", "value": 270.0, "unit": "nM"}],
                )
            ],
        )

        result = await tool.execute({"query": GENE_QUERY})

        (hit,) = _hits(result)
        assert hit["binding_affinity"] == [
            {"comp_id": "ZEI", "type": "IC50", "value": 270.0, "unit": "nM"}
        ]

    @pytest.mark.asyncio
    async def test_computed_models_are_flagged(self, tool, serve):
        """A predicted model must not be mistaken for an experimental one."""
        serve(
            {"total_count": 1, "result_set": ["AF_AFP00533F1"]},
            entries=[_entry("AF_AFP00533F1", method="Other", methodology="computational")],
        )

        result = await tool.execute({"query": GENE_QUERY})

        assert _hits(result)[0].get("methodology") == "computational"

    @pytest.mark.asyncio
    async def test_only_this_page_is_enriched(self, tool, serve):
        recorder = serve(
            {"total_count": 84, "result_set": ["8U4D", "3SUI"]},
            entries=[_entry("8U4D"), _entry("3SUI")],
        )

        await tool.execute({"query": GENE_QUERY})

        assert len(recorder.graphql_payloads) == 1
        assert recorder.graphql_payloads[0]["variables"]["ids"] == ["8U4D", "3SUI"]

    @pytest.mark.asyncio
    async def test_data_api_failure_still_returns_the_identifiers(self, tool, serve_handler):
        """The search succeeded; losing the details must not lose the result."""

        def handler(request):
            if "rcsbsearch" in str(request.url):
                return httpx.Response(200, json={"total_count": 1, "result_set": ["8U4D"]})
            return httpx.Response(500, text="upstream is down")

        serve_handler(handler)

        result = await tool.execute({"query": GENE_QUERY})

        assert result.content_type == ContentType.TEXT
        assert "8U4D" in result.data
        assert "unavailable" in result.data

    @pytest.mark.asyncio
    async def test_non_entry_return_type_skips_enrichment(self, tool, serve):
        """PDBID_N is not an entry ID, so there is nothing to look up."""
        recorder = serve({"total_count": 84, "result_set": ["11CJ_1", "11CK_1"]})

        result = await tool.execute({"query": GENE_QUERY, "return_type": "polymer_entity"})

        assert not recorder.graphql_payloads
        assert _hits(result) == [{"identifier": "11CJ_1"}, {"identifier": "11CK_1"}]
        assert "return_type='entry'" in result.data


class TestOutputFraming:
    @pytest.mark.asyncio
    async def test_total_count_is_always_reported(self, tool, serve):
        """A narrow query has to be visibly narrow."""
        serve({"total_count": 84, "result_set": ["8U4D"]}, entries=[_entry("8U4D")])

        result = await tool.execute({"query": GENE_QUERY})

        assert "total_count: 84" in result.data
        assert result.metadata["total_count"] == 84
        assert result.metadata["identifiers"] == ["8U4D"]

    @pytest.mark.asyncio
    async def test_output_names_the_search_subject(self, tool, serve):
        """The findings ledger drops the query object for being too long, so the
        tool's own text has to say what was searched for."""
        serve({"total_count": 84, "result_set": ["8U4D"]}, entries=[_entry("8U4D")])

        result = await tool.execute({"query": GENE_QUERY})

        assert result.data.startswith("Searched for 'TRPV1'.")

    @pytest.mark.asyncio
    async def test_output_omits_the_subject_when_unidentifiable(self, tool, serve):
        serve({"total_count": 1, "result_set": ["8U4D"]}, entries=[_entry("8U4D")])

        numeric_only = {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "less",
                "value": 2.0,
            },
        }
        result = await tool.execute({"query": numeric_only})

        assert result.data.startswith("total_count:")

    @pytest.mark.asyncio
    async def test_method_breakdown_is_surfaced(self, tool, serve):
        serve(
            {
                "total_count": 84,
                "result_set": ["8U4D"],
                "facets": [
                    {
                        "name": "experimental_method",
                        "buckets": [
                            {"label": "EM", "population": 80},
                            {"label": "X-ray", "population": 4},
                        ],
                    }
                ],
            },
            entries=[_entry("8U4D")],
        )

        result = await tool.execute({"query": GENE_QUERY})

        assert "EM: 80" in result.data
        assert "X-ray: 4" in result.data

    @pytest.mark.asyncio
    async def test_unsorted_results_advertise_sorting(self, tool, serve):
        serve({"total_count": 84, "result_set": ["8U4D"]}, entries=[_entry("8U4D")])

        result = await tool.execute({"query": GENE_QUERY})

        assert "relevance score" in result.data
        assert "request_options.sort" in result.data

    @pytest.mark.asyncio
    async def test_sorted_results_do_not_advertise_sorting(self, tool, serve):
        serve({"total_count": 84, "result_set": ["8U4D"]}, entries=[_entry("8U4D")])

        result = await tool.execute(
            {
                "query": GENE_QUERY,
                "request_options": {
                    "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}]
                },
            }
        )

        assert "relevance score" not in result.data
        assert "paginate.start=1" in result.data

    @pytest.mark.asyncio
    async def test_a_full_page_fits_the_response_budget(self, tool, serve):
        """Overflow truncation keeps the head and tail, dropping the middle of
        the very candidate list the agent is choosing from."""
        ligands = tuple((f"L{i:02d}", "a long ligand name " * 3, 493.6) for i in range(8))
        ids = [f"{i:03d}X" for i in range(MAX_ROWS)]
        serve(
            {"total_count": 9999, "result_set": ids},
            entries=[
                _entry(
                    pdb_id,
                    title="a deliberately long structure title " * 4,
                    resolution=[2.2],
                    ligands=ligands,
                    affinity=[
                        {"comp_id": "L00", "type": "IC50", "value": 270.0, "unit": "nM"}
                    ] * 4,
                )
                for pdb_id in ids
            ],
        )

        result = await tool.execute(
            {"query": GENE_QUERY, "request_options": {"paginate": {"rows": MAX_ROWS}}}
        )

        # The budget is in tokens; four characters per token is the conservative
        # approximation the budget itself uses.
        assert len(result.data) / 4 < tool.max_response_tokens
        assert len(_hits(result)) == MAX_ROWS

    @pytest.mark.asyncio
    async def test_grouped_results_preserve_membership(self, tool, serve):
        serve(
            {
                "total_count": 84,
                "group_by_count": 2,
                "group_by": {"aggregation_method": "sequence_identity"},
                "group_set": [
                    {
                        "identifier": "34156_90",
                        "count": 2,
                        "representative": {"identifier": "11CJ_1"},
                        "result_set": [{"identifier": "11CJ_1"}, {"identifier": "11CK_1"}],
                    }
                ],
            }
        )

        result = await tool.execute(
            {
                "query": GENE_QUERY,
                "return_type": "polymer_entity",
                "request_options": {
                    "group_by": {"aggregation_method": "sequence_identity", "similarity_cutoff": 90},
                    "group_by_return_type": "groups",
                },
            }
        )

        (group,) = _hits(result)
        assert group["group_id"] == "34156_90"
        assert group["representative"] == "11CJ_1"
        assert group["members"] == ["11CJ_1", "11CK_1"]


class TestAttributeDiscovery:
    SCHEMA = {
        "properties": {
            "rcsb_entry_info": {
                "properties": {
                    "experimental_method": {
                        "type": "string",
                        "description": "The category of experimental method.",
                        "enum": ["EM", "NMR", "X-ray"],
                    },
                    "resolution_combined": {
                        "type": "array",
                        "items": {"type": "number", "description": "Combined resolution estimate."},
                    },
                }
            },
            "rcsb_membrane_lineage": {
                "type": "array",
                "items": {
                    "properties": {
                        "name": {"type": "string", "description": "Membrane protein class."}
                    }
                },
            },
        }
    }

    @pytest.fixture(autouse=True)
    def _clear_cache(self, monkeypatch):
        monkeypatch.setattr(rcsb, "_ATTRIBUTE_INDEX", None)
        monkeypatch.setattr(rcsb, "_ATTRIBUTE_INDEX_RETRY_AFTER", 0.0)

    @pytest.fixture
    def serve_schema(self, serve_handler):
        def install(response: httpx.Response) -> list[int]:
            calls: list[int] = []

            def handler(request):
                calls.append(1)
                return response

            serve_handler(handler)
            return calls

        return install

    def test_flatten_addresses_array_members_by_dotted_path(self):
        index = {item["attribute"]: item for item in rcsb._flatten_schema(self.SCHEMA)}

        # resolution_combined is an array of numbers; membrane_lineage an array
        # of objects. Both are addressed as if they were plain nested fields.
        assert index["rcsb_entry_info.resolution_combined"]["type"] == "number"
        assert index["rcsb_membrane_lineage.name"]["type"] == "string"

    def test_flatten_captures_allowed_values(self):
        index = {item["attribute"]: item for item in rcsb._flatten_schema(self.SCHEMA)}

        assert index["rcsb_entry_info.experimental_method"]["allowed_values"] == [
            "EM",
            "NMR",
            "X-ray",
        ]

    def test_matches_on_attribute_name_rank_first(self):
        index = rcsb._flatten_schema(self.SCHEMA)

        matches = rcsb._match_attributes(index, "method")

        assert matches[0]["attribute"] == "rcsb_entry_info.experimental_method"

    def test_matches_on_description_when_the_name_misses(self):
        index = rcsb._flatten_schema(self.SCHEMA)

        matches = rcsb._match_attributes(index, "membrane protein class")

        assert [m["attribute"] for m in matches] == ["rcsb_membrane_lineage.name"]

    @pytest.mark.asyncio
    async def test_find_attributes_returns_matches(self, tool, serve_schema):
        serve_schema(httpx.Response(200, json=self.SCHEMA))

        result = await tool.execute({"find_attributes": "resolution"})

        assert result.content_type == ContentType.TEXT
        assert "rcsb_entry_info.resolution_combined" in result.data

    @pytest.mark.asyncio
    async def test_find_attributes_takes_precedence_over_query(self, tool, serve_schema):
        """Documented as either/or; discovery wins so no search is issued."""
        serve_schema(httpx.Response(200, json=self.SCHEMA))

        result = await tool.execute({"find_attributes": "method", "query": GENE_QUERY})

        assert "experimental_method" in result.data

    @pytest.mark.asyncio
    async def test_find_attributes_reports_a_miss_helpfully(self, tool, serve_schema):
        serve_schema(httpx.Response(200, json=self.SCHEMA))

        result = await tool.execute({"find_attributes": "zzznotanattribute"})

        assert result.content_type == ContentType.TEXT
        assert "zzznotanattribute" in result.data

    @pytest.mark.asyncio
    async def test_schema_is_fetched_once_and_cached(self, tool, serve_schema):
        calls = serve_schema(httpx.Response(200, json=self.SCHEMA))

        await tool.execute({"find_attributes": "method"})
        await tool.execute({"find_attributes": "resolution"})

        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_schema_failure_degrades_instead_of_erroring(self, tool, serve_schema):
        serve_schema(httpx.Response(503, text="unavailable"))

        result = await tool.execute({"find_attributes": "method"})

        assert result.content_type == ContentType.TEXT
        assert "unavailable" in result.data.lower()

    @pytest.mark.asyncio
    async def test_schema_failure_is_not_retried_immediately(self, tool, serve_schema):
        """A network outage must not add a 10s timeout to every later call."""
        calls = serve_schema(httpx.Response(503, text="unavailable"))

        await tool.execute({"find_attributes": "method"})
        await tool.execute({"find_attributes": "method"})

        assert len(calls) == 1
