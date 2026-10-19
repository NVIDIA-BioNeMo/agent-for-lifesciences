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

"""Mock tool registry for training.

This module provides a mock registry that can be used during RL training
to avoid hitting real microservices. It supports:

- Cached responses: Replay recorded tool outputs
- Mock responses: Return predefined responses
- Passthrough: Selectively use real tools for some calls

This is critical for training because:
1. Real microservices are too slow for RL training loops
2. Real microservices cost money (NVIDIA API, etc.)
3. Training needs determinism for reproducibility
4. Some tools have rate limits
"""

import base64
import json
import logging
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional

from api.tools.base import BaseTool, ContentType, DataIOCategory, ToolResult, ToolType
from api.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from api.storage import DataStorageService

logger = logging.getLogger(__name__)

# A storage reference is "{data_type}_{hex}" -- see api/storage/datatypes.py.
_REFERENCE_PATTERN = re.compile(r"^([a-z0-9]+)_([0-9a-f]{8,32})$")

# The same shape found anywhere inside a longer string. Tool messages embed
# references in prose ("PDB data ... has been stored as 'pdb_d75154ac'."), and
# that prose is what the model reads, so replay has to rewrite it too. Over-
# matching is harmless: _capture_artifacts keeps only references that actually
# resolve in storage.
_EMBEDDED_REFERENCE_PATTERN = re.compile(r"\b([a-z0-9]+_[0-9a-f]{8,32})\b")


@dataclass
class CachedArtifact:
    """An artifact a cached tool stored, captured so replay can re-store it.

    Tools in the OUTPUT_STORES and BOTH categories write to the data store and
    return only a reference. Replaying such a response verbatim hands back a
    reference to a row that does not exist in a fresh database, so every
    downstream INPUT_RETRIEVES tool fails and the storage strip comes up empty.
    Capturing the payload alongside the response is what makes replay usable
    for anything past the first tool call.

    Attributes:
        reference: The reference the tool returned at record time. Used only as
            the lookup key for rewriting; the replayed run gets a fresh one.
        data_type: Storage data type, needed to re-store under the same type.
        content_base64: The artifact bytes. Base64 because artifacts may be
            binary (``array`` is an ``.npz``) and the cache file is JSON.
        metadata: The artifact's stored metadata.
        description: The artifact's stored description.
    """

    reference: str
    data_type: str
    content_base64: str
    metadata: dict = field(default_factory=dict)
    description: str | None = None

    def to_dict(self) -> dict:
        return {
            "reference": self.reference,
            "data_type": self.data_type,
            "content_base64": self.content_base64,
            "metadata": self.metadata,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CachedArtifact":
        return cls(
            reference=data["reference"],
            data_type=data["data_type"],
            content_base64=data["content_base64"],
            metadata=data.get("metadata") or {},
            description=data.get("description"),
        )

    @property
    def content(self) -> bytes:
        return base64.b64decode(self.content_base64)


@dataclass
class CachedResponse:
    """A cached tool response for replay.

    Attributes:
        tool_name: Name of the tool.
        args_hash: Hash of the arguments (for lookup).
        result: The cached ToolResult.
        metadata: Additional metadata about the cache entry.
        artifacts: Artifacts the tool stored, keyed by their record-time
            reference, so replay can re-materialize them.
    """

    tool_name: str
    args_hash: str
    result: ToolResult
    metadata: dict = field(default_factory=dict)
    artifacts: dict[str, CachedArtifact] = field(default_factory=dict)

    # Arguments the model writes as free prose, which label the stored artifact
    # and have no bearing on what the tool returns. Hashing them makes a cache
    # useless against a live model: a recorded fetch_and_clean_pdb of 1CRN missed
    # on replay purely because the model described it as "1CRN crambin structure"
    # instead of "1CRN structure for analysis", and the resulting CACHE MISS sent
    # the model off to predict the structure with OpenFold instead.
    #
    # The cost is that a replayed artifact carries the description from the
    # recording rather than the one the model just wrote. That shows up in
    # artifact listings and nowhere else.
    COSMETIC_ARGS: frozenset[str] = frozenset({"description", "label"})

    @staticmethod
    def hash_args(arguments: dict) -> str:
        """Hash the arguments that determine the response, ignoring the rest."""
        significant = {
            key: value
            for key, value in arguments.items()
            if key not in CachedResponse.COSMETIC_ARGS
        }
        normalized = json.dumps(significant, sort_keys=True)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def _collect_references(value: Any, found: set[str]) -> None:
    """Walk a payload gathering every storage reference it mentions.

    Tools do not restrict references to ``result.data``: ``vina_dock`` reports
    the docked complex in metadata, skill endpoints report ``array`` refs for
    auxiliary matrices, and ``data_reference`` results repeat the ref inside a
    prose ``message``. Missing any of those leaves dangling references in an
    otherwise-rehydrated response.
    """
    if isinstance(value, str):
        found.update(_EMBEDDED_REFERENCE_PATTERN.findall(value))
    elif isinstance(value, dict):
        for item in value.values():
            _collect_references(item, found)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_references(item, found)


def _rewrite_references(value: Any, mapping: dict[str, str]) -> Any:
    """Return ``value`` with every record-time reference swapped for its new one.

    Substitution is by whole token, so a reference embedded in a sentence is
    rewritten while a longer identifier that merely starts with one is left
    alone.
    """
    if not mapping:
        return value
    if isinstance(value, str):
        return _EMBEDDED_REFERENCE_PATTERN.sub(
            lambda match: mapping.get(match.group(1), match.group(1)), value
        )
    if isinstance(value, dict):
        return {key: _rewrite_references(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_references(item, mapping) for item in value]
    if isinstance(value, tuple):
        return tuple(_rewrite_references(item, mapping) for item in value)
    return value


class MockTool(BaseTool):
    """A mock tool that returns predefined responses.

    This is used to replace real tools during training. It can:
    - Return a fixed response
    - Call a response function
    - Look up responses from a cache

    Example:
        mock_tool = MockTool(
            name="fetch_pdb",
            description="Fetch PDB structure",
            response_fn=lambda args: ToolResult.text(f"Mocked PDB for {args['pdb_id']}"),
        )
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict | None = None,
        response: ToolResult | None = None,
        response_fn: Callable[[dict], ToolResult] | None = None,
    ):
        """Initialize mock tool.

        Args:
            name: Tool name.
            description: Tool description.
            parameters: JSON schema for parameters.
            response: Fixed response to return.
            response_fn: Function to compute response from args.
        """
        self._name = name
        self._description = description
        self._parameters = parameters or {"type": "object", "properties": {}}
        self._response = response
        self._response_fn = response_fn

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def tool_type(self) -> ToolType:
        return ToolType.LOCAL

    @property
    def data_io_category(self) -> DataIOCategory:
        return DataIOCategory.TEXT_ONLY

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._parameters,
            },
        }

    async def execute(self, arguments: dict) -> ToolResult:
        """Execute and return mock response."""
        if self._response_fn:
            return self._response_fn(arguments)
        elif self._response:
            return self._response
        else:
            return ToolResult.text(f"[MOCK] {self.name} called with {arguments}")


class CachingRegistry(ToolRegistry):
    """Registry that caches tool responses for deterministic replay.

    This registry wraps a real registry and caches responses to disk.
    During training, it can replay cached responses instead of making
    real tool calls.

    Modes:
    - "record": Call real tools and cache responses
    - "replay": Return cached responses only
    - "mixed": Use cache if available, call real tool otherwise

    Example:
        # Record mode: Build cache
        registry = CachingRegistry(real_registry, cache_dir="/cache", mode="record")
        result = await registry.execute("fetch_pdb", {"pdb_id": "1AKE"})

        # Replay mode: Use cache for training
        registry = CachingRegistry(real_registry, cache_dir="/cache", mode="replay")
        result = await registry.execute("fetch_pdb", {"pdb_id": "1AKE"})  # Returns cached
    """

    # Tools whose result is a function of live session state rather than of an
    # external service. Caching buys no determinism for them -- they make no
    # network call -- and replaying them actively lies:
    #
    #   structure_query emits an SSE event and blocks on the client POSTing a
    #   selection back. Served from cache, the round-trip never happens, so the
    #   eval that exists to prove the browser bridge works would pass with the
    #   bridge unplugged.
    #
    # The viewer and sandbox tools have the same shape: their output describes
    # the state of this session, and a recording describes some other one.
    NEVER_CACHED: frozenset[str] = frozenset(
        {
            "structure_query",
            "get_viewer_state",
            "visualize_pdb",
            "visualize_sdf",
            "python_execute",
            "search_session_history",
        }
    )

    def __init__(
        self,
        base_registry: ToolRegistry | None = None,
        cache_dir: str | Path = ".tool_cache",
        mode: str = "replay",
        never_cached: Iterable[str] | None = None,
    ):
        """Initialize caching registry.

        Args:
            base_registry: The real registry to wrap (for record/mixed mode).
            cache_dir: Directory to store cached responses.
            mode: "record", "replay", or "mixed".
            never_cached: Tool names to always execute live, overriding
                NEVER_CACHED. These bypass the cache even in "replay" mode.
        """
        super().__init__()
        self._base_registry = base_registry
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._mode = mode
        self._never_cached = (
            self.NEVER_CACHED if never_cached is None else frozenset(never_cached)
        )
        self._memory_cache: dict[str, CachedResponse] = {}

        # Load existing cache to memory for fast lookup
        self._load_cache()

        logger.info(f"CachingRegistry initialized: mode={mode}, cache_dir={cache_dir}")

    def _load_cache(self) -> None:
        """Load cache entries from disk to memory."""
        for cache_file in self._cache_dir.glob("*.json"):
            try:
                with open(cache_file) as f:
                    data = json.load(f)

                # Reconstruct ToolResult from cached data
                result = ToolResult(
                    content_type=ContentType(data["content_type"]),
                    data=data["data"],
                    metadata=data.get("metadata", {}),
                )

                entry = CachedResponse(
                    tool_name=data["tool_name"],
                    args_hash=data["args_hash"],
                    result=result,
                    metadata=data.get("cache_metadata", {}),
                    artifacts={
                        payload["reference"]: CachedArtifact.from_dict(payload)
                        for payload in data.get("artifacts", [])
                    },
                )

                cache_key = f"{entry.tool_name}:{entry.args_hash}"
                self._memory_cache[cache_key] = entry

            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}")

        logger.info(f"Loaded {len(self._memory_cache)} cached tool responses")

    def _save_cache_entry(self, entry: CachedResponse) -> None:
        """Save a cache entry to disk."""
        cache_key = f"{entry.tool_name}:{entry.args_hash}"
        # The filename separator is not the lookup separator: _load_cache reads
        # tool_name out of the JSON body, so the name is cosmetic, and a colon in
        # it makes `git checkout` fail outright on Windows. The eval fixtures
        # under evals/fixtures/tool_cache/ are committed, so that matters.
        cache_file = self._cache_dir / f"{entry.tool_name}__{entry.args_hash}.json"

        data = {
            "tool_name": entry.tool_name,
            "args_hash": entry.args_hash,
            "content_type": entry.result.content_type.value,
            "data": entry.result.data,
            "metadata": entry.result.metadata,
            "cache_metadata": entry.metadata,
            "artifacts": [
                artifact.to_dict() for artifact in entry.artifacts.values()
            ],
        }

        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

        self._memory_cache[cache_key] = entry

    def _lookup_cache(self, tool_name: str, arguments: dict) -> CachedResponse | None:
        """Look up a cached entry."""
        args_hash = CachedResponse.hash_args(arguments)
        cache_key = f"{tool_name}:{args_hash}"

        entry = self._memory_cache.get(cache_key)
        if entry:
            logger.debug(f"Cache hit: {cache_key}")
            return entry

        logger.debug(f"Cache miss: {cache_key}")
        return None

    @staticmethod
    async def _artifact_description(
        storage: "DataStorageService", reference: str
    ) -> str | None:
        """The artifact's description, which ``retrieve_with_metadata`` omits.

        Description is its own column, and it is the label the storage strip and
        the eval verifier read, so a replayed artifact without it looks different
        from a recorded one.
        """
        try:
            matches = await storage.search_artifacts(query=reference, limit=5)
        except Exception as e:
            logger.debug(f"Could not read description for {reference}: {e}")
            return None
        for match in matches:
            if match.get("reference") == reference:
                return match.get("description")
        return None

    async def _capture_artifacts(
        self,
        result: ToolResult,
        arguments: dict[str, Any],
        storage: Optional["DataStorageService"],
    ) -> dict[str, CachedArtifact]:
        """Read back every artifact the tool just created, for later replay."""
        if storage is None:
            return {}

        references: set[str] = set()
        _collect_references(result.data, references)
        _collect_references(result.metadata, references)

        # References the caller supplied are inputs that already exist in the
        # replaying run's store. Re-storing them would duplicate rows and make
        # the artifact listing disagree with a live run.
        inputs: set[str] = set()
        _collect_references(arguments, inputs)
        references -= inputs

        if not references:
            return {}

        captured: dict[str, CachedArtifact] = {}
        for reference in sorted(references):
            try:
                content, metadata = await storage.retrieve_with_metadata(reference)
            except Exception as e:
                # Reference-shaped strings that are not live artifacts land here
                # -- the embedded-reference pattern is deliberately permissive
                # and this lookup is what filters it.
                logger.debug(f"Not capturing {reference} for cache: {e}")
                continue

            captured[reference] = CachedArtifact(
                reference=reference,
                data_type=_REFERENCE_PATTERN.match(reference).group(1),
                content_base64=base64.b64encode(content).decode("ascii"),
                metadata={
                    key: value
                    for key, value in (metadata or {}).items()
                    # session_id is re-applied by SessionScopedStorage on replay;
                    # carrying the recording session's id would misattribute the
                    # artifact and break per-session artifact listing.
                    if key != "session_id"
                },
                description=await self._artifact_description(storage, reference),
            )

        return captured

    async def _rehydrate(
        self,
        entry: CachedResponse,
        storage: Optional["DataStorageService"],
    ) -> ToolResult:
        """Re-store the entry's artifacts and point the result at the new refs.

        Without this, a replayed OUTPUT_STORES response names an artifact that
        does not exist in this run's database.
        """
        if not entry.artifacts:
            return entry.result
        if storage is None:
            logger.warning(
                f"Replaying {entry.tool_name} with {len(entry.artifacts)} cached "
                "artifact(s) but no storage service; references will dangle"
            )
            return entry.result

        mapping: dict[str, str] = {}
        for reference, artifact in entry.artifacts.items():
            try:
                mapping[reference] = await storage.store(
                    artifact.data_type,
                    artifact.content,
                    dict(artifact.metadata),
                    description=artifact.description,
                )
            except Exception as e:
                logger.error(
                    f"Failed to rehydrate cached artifact {reference} for "
                    f"{entry.tool_name}: {e}"
                )
                return ToolResult.error(
                    f"[CACHE REHYDRATION FAILED] Could not restore cached artifact "
                    f"for {entry.tool_name}: {e}"
                )

        return ToolResult(
            content_type=entry.result.content_type,
            data=_rewrite_references(entry.result.data, mapping),
            metadata=_rewrite_references(entry.result.metadata, mapping),
        )

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        storage: Optional["DataStorageService"] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        job_manager: Optional[Any] = None,
    ) -> ToolResult:
        """Execute tool with caching.

        Behavior depends on mode:
        - "replay": Return cached only, error if not cached
        - "record": Call real tool and cache result
        - "mixed": Return cached if available, else call real tool

        Tools in ``never_cached`` ignore all of that and go straight to the base
        registry, in every mode. See NEVER_CACHED for why.

        The signature must track ToolRegistry.execute() exactly: callers pass
        these by keyword, so an argument missing here is a TypeError at the first
        tool call rather than anything the type checker catches.
        """
        if tool_name in self._never_cached:
            if self._base_registry is None:
                return ToolResult.error(
                    f"{tool_name} is never cached and no base registry is "
                    "configured to execute it live"
                )
            return await self._base_registry.execute(
                tool_name,
                arguments,
                storage,
                session_id,
                request_id=request_id,
                job_manager=job_manager,
            )

        # Check cache first
        cached = self._lookup_cache(tool_name, arguments)

        if self._mode == "replay":
            if cached:
                return await self._rehydrate(cached, storage)
            else:
                logger.warning(f"No cached response for {tool_name}, returning error")
                return ToolResult.error(
                    f"[CACHE MISS] No cached response for {tool_name} with given arguments. "
                    "Run in 'record' mode first to build cache."
                )

        if self._mode == "mixed" and cached:
            return await self._rehydrate(cached, storage)

        # Call real tool (record or mixed mode)
        if self._base_registry is None:
            return ToolResult.error("No base registry configured for record/mixed mode")

        result = await self._base_registry.execute(
            tool_name,
            arguments,
            storage,
            session_id,
            request_id=request_id,
        )

        # Cache the result
        if result.content_type != ContentType.ERROR:
            entry = CachedResponse(
                tool_name=tool_name,
                args_hash=CachedResponse.hash_args(arguments),
                result=result,
                metadata={"arguments": arguments},
                artifacts=await self._capture_artifacts(result, arguments, storage),
            )
            self._save_cache_entry(entry)
            logger.info(
                f"Cached response for {tool_name} "
                f"({len(entry.artifacts)} artifact(s))"
            )

        return result

    async def get_all_schemas(
        self,
        session_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get schemas from base registry if available."""
        if self._base_registry:
            return await self._base_registry.get_all_schemas(session_id)
        return await super().get_all_schemas(session_id)


class MockRegistry(ToolRegistry):
    """Registry with mock tools for fully deterministic training.

    This registry contains only mock tools with predefined responses.
    It's useful when you want fully controlled training without any
    real tool calls or cache dependencies.

    Example:
        registry = MockRegistry()
        registry.register_mock("fetch_pdb", response=ToolResult.text("Mock PDB data"))

        result = await registry.execute("fetch_pdb", {"pdb_id": "1AKE"})
        # Returns "Mock PDB data"
    """

    def __init__(self):
        """Initialize empty mock registry."""
        super().__init__()
        self._mock_responses: dict[str, ToolResult | Callable] = {}

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        storage: Optional["DataStorageService"] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        job_manager: Optional[Any] = None,
    ) -> ToolResult:
        """Execute a mock tool (no telemetry overhead).

        This simplified execute() skips telemetry spans to avoid
        issues in training contexts where telemetry isn't initialized.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Arguments to pass to the tool.
            storage: Ignored for mock tools.
            session_id: Ignored for mock tools.
            request_id: Ignored for mock tools.
            job_manager: Ignored for mock tools -- a mock never runs the real
                async tool, so there is no job to track. Accepted so this
                signature stays interchangeable with the real registry's.

        Returns:
            ToolResult from mock tool execution.
        """
        tool = self.get(tool_name)
        if tool is None:
            logger.warning(f"Mock tool not found: {tool_name}")
            return ToolResult.error(f"Unknown mock tool: {tool_name}")

        try:
            return await tool.execute(arguments)
        except Exception as e:
            logger.exception(f"Mock tool {tool_name} execution failed")
            return ToolResult.error(f"Mock tool execution failed: {str(e)}")

    def register_mock(
        self,
        tool_name: str,
        description: str = "Mock tool",
        parameters: dict | None = None,
        response: ToolResult | None = None,
        response_fn: Callable[[dict], ToolResult] | None = None,
    ) -> None:
        """Register a mock tool.

        Args:
            tool_name: Name of the tool to mock.
            description: Tool description.
            parameters: JSON schema for parameters.
            response: Fixed response to return.
            response_fn: Function to compute response from args.
        """
        mock_tool = MockTool(
            name=tool_name,
            description=description,
            parameters=parameters,
            response=response,
            response_fn=response_fn,
        )
        self.register(mock_tool)
        logger.info(f"Registered mock tool: {tool_name}")

    def register_passthrough(
        self,
        tool_name: str,
        real_registry: ToolRegistry,
    ) -> None:
        """Register a tool that passes through to a real registry.

        Use this when you want to mock most tools but let some through.

        Args:
            tool_name: Tool to pass through.
            real_registry: Registry with the real tool.
        """
        real_tool = real_registry.get(tool_name)
        if real_tool:
            self.register(real_tool)
            logger.info(f"Registered passthrough tool: {tool_name}")
        else:
            logger.warning(f"Tool {tool_name} not found in real registry")


def create_training_registry(
    mode: str = "mock",
    cache_dir: str | Path = ".tool_cache",
    mock_responses: dict[str, ToolResult] | None = None,
) -> ToolRegistry:
    """Factory function to create a training-appropriate registry.

    Args:
        mode: Registry mode:
            - "mock": Fully mocked with predefined responses
            - "cached": Use CachingRegistry in replay mode
            - "mixed": Use cache when available, real tools otherwise
        cache_dir: Directory for cached responses.
        mock_responses: Optional dict of tool_name -> response for mock mode.

    Returns:
        Registry suitable for training.

    Example:
        # Fully mocked training
        registry = create_training_registry(mode="mock")

        # Replay cached responses
        registry = create_training_registry(mode="cached", cache_dir="/data/cache")

        # Mixed mode for building cache
        registry = create_training_registry(mode="mixed")
    """
    if mode == "mock":
        registry = MockRegistry()

        # Register default mock responses with clear [MOCK] prefixes
        # These responses simulate realistic tool outputs for training.
        #
        # Primary training tools (names aligned with production registry):
        #   - vina_dock, python_execute, molecule_info
        # Legacy aliases (kept for backward compatibility with old JSONL data):
        #   - python_exec, load_molecule, compute_metrics, etc.
        default_mocks = {
            # === Primary training tools ===
            "vina_dock": ToolResult.text(
                "[MOCK] Autodock Vina Docking Results:\n"
                "  Ligand: (mock ligand)\n"
                "  Receptor: (mock receptor)\n"
                "  Best binding affinity: -8.5 kcal/mol\n"
                "  Poses found: 9\n"
                "  Interpretation: Good binding\n"
                "\n  Top poses:\n"
                "    Pose 1: -8.500 kcal/mol\n"
                "    Pose 2: -8.100 kcal/mol\n"
                "    Pose 3: -7.800 kcal/mol\n"
                "(This is a MOCK response - no real docking performed)"
            ),
            "python_execute": ToolResult.text(
                "[MOCK] Python Execution Complete:\n"
                "Code executed successfully.\n"
                "Output: Analysis complete. LogP difference: 2.5, MW difference: 150.3\n"
                "(This is a MOCK response - no real Python code executed)"
            ),
            "molecule_info": ToolResult.text(
                "[MOCK] Molecular Information:\n"
                "  SMILES: CC(C)Cc1ccc(cc1)C(C)C(=O)O\n"
                "  Formula: C13H18O2\n"
                "  Molecular Weight: 206.28 g/mol\n"
                "  LogP: 3.50\n"
                "  TPSA: 37.30 Å²\n"
                "  QED: 0.7453\n"
                "  H-Bond Donors: 1\n"
                "  H-Bond Acceptors: 2\n"
                "  Rotatable Bonds: 4\n"
                "  Rings: 1\n"
                "  Atoms: 15\n"
                "(This is a MOCK response - no real RDKit computation performed)"
            ),
            # === Production tools (non-training) ===
            "fetch_protein_sequence": ToolResult.text(
                "[MOCK] UniProt Sequence Retrieved:\n"
                "- Protein: Example Kinase\n"
                "- Length: 300 residues\n"
                "- Sequence: MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH...\n"
                "(This is a MOCK response)"
            ),
            "openfold3": ToolResult.text(
                "[MOCK] Structure Prediction Complete:\n"
                "- Job ID: mock_job_12345\n"
                "- pLDDT Score: 85.0 (high confidence)\n"
                "- Structure stored as: pdb_mock_12345\n"
                "(This is a MOCK response - no real OpenFold prediction performed)"
            ),
            # Shaped like a real search_rcsb response: total_count, the
            # experimental-method breakdown, and flat per-entry facts. A rollout
            # trained against a bare ID list would not learn to read is_holo or
            # to notice that a method filter had excluded most of the archive.
            "search_rcsb": ToolResult.text(
                "Searched for 'MOCK'. total_count: 84 — showing 2 (page on with "
                "request_options.paginate.start=2). return_type: entry.\n\n"
                "Match set breakdown:\n"
                "experimental_method — EM: 80, X-ray: 4\n\n"
                "One JSON object per line:\n"
                '{"pdb_id":"1ABC","title":"[MOCK] Receptor with bound inhibitor",'
                '"experimental_method":"EM","resolution":2.2,"released":"2024-05-08",'
                '"is_holo":true,"ligands":[{"id":"STI","name":"[MOCK] inhibitor",'
                '"formula_weight":493.6}]}\n'
                '{"pdb_id":"2DEF","title":"[MOCK] Apo receptor","experimental_method":"X-ray",'
                '"resolution":1.9,"released":"2021-03-11","is_holo":false,"ligands":[]}\n\n'
                "(This is a MOCK response)"
            ),
            "fetch_and_clean_pdb": ToolResult.text(
                "[MOCK] PDB Structure Fetched:\n"
                "- PDB ID: 1ABC\n"
                "- Chains: A, B\n"
                "- Residues: 245\n"
                "- Stored as: pdb_1abc_clean\n"
                "(This is a MOCK response)"
            ),
            "visualize_pdb": ToolResult.text(
                "[MOCK] Visualization Ready:\n"
                "- Structure displayed in 3D viewer\n"
                "(This is a MOCK response)"
            ),
            "structure_query": ToolResult.text(
                "[MOCK] Structure Query Executed:\n"
                "- Selected 100 atoms matching query\n"
                "(This is a MOCK response)"
            ),
            # === Legacy aliases (backward compat with old data/prompts) ===
            "get_molecule_info": ToolResult.text(
                "[MOCK] Molecular Information:\n"
                "- Formula: C21H30O2\n"
                "- Molecular Weight: 314.47 g/mol\n"
                "- LogP: 3.2\n"
                "- TPSA: 37.3 Å²\n"
                "(This is a MOCK response)"
            ),
            "fetch_uniprot_sequence": ToolResult.text(
                "[MOCK] UniProt Sequence Retrieved (legacy alias):\n"
                "(This is a MOCK response)"
            ),
            "predict_structure_openfold3": ToolResult.text(
                "[MOCK] Structure Prediction (legacy alias):\n(This is a MOCK response)"
            ),
            "compute_metrics": ToolResult.text(
                "[MOCK] Molecular Metrics Computed:\n"
                "- LogP: 3.5\n"
                "- TPSA: 45.2 Å²\n"
                "- QED: 0.65\n"
                "- Molecular Weight: 425.5 g/mol\n"
                "(This is a MOCK response)"
            ),
            "load_molecule": ToolResult.text(
                "[MOCK] Molecule Loaded:\n"
                "- Reference: sdf_mock_12345\n"
                "- SMILES: CC(C)Cc1ccc(cc1)C(C)C(=O)O\n"
                "- Atoms: 25\n"
                "(This is a MOCK response)"
            ),
            "python_exec": ToolResult.text(
                "[MOCK] Python Execution (legacy alias):\n"
                "Code executed successfully.\n"
                "(This is a MOCK response)"
            ),
        }

        # Override with user-provided responses
        if mock_responses:
            default_mocks.update(mock_responses)

        for tool_name, response in default_mocks.items():
            registry.register_mock(
                tool_name,
                description=f"Mock {tool_name}",
                response=response,
            )

        return registry

    elif mode in ("cached", "replay"):
        from api.tools.registry import get_registry

        real_registry = get_registry()
        return CachingRegistry(
            base_registry=real_registry,
            cache_dir=cache_dir,
            mode="replay",
        )

    elif mode == "mixed":
        from api.tools.registry import get_registry

        real_registry = get_registry()
        return CachingRegistry(
            base_registry=real_registry,
            cache_dir=cache_dir,
            mode="mixed",
        )

    elif mode == "record":
        from api.tools.registry import get_registry

        real_registry = get_registry()
        return CachingRegistry(
            base_registry=real_registry,
            cache_dir=cache_dir,
            mode="record",
        )

    else:
        raise ValueError(f"Unknown registry mode: {mode}")
