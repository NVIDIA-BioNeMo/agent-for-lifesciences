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

"""Load the ADMET-AI ensemble once and predict against it.

Three things this file exists to get right.

The model is loaded lazily and exactly once. ADMET-AI ships two multitask
Chemprop ensembles -- one classification, one regression, ten checkpoints between
them -- and constructing ``ADMETModel`` reads all of them off disk, which takes
around five seconds. Doing that per request would dominate the latency of a call
that then runs in a fraction of a second. Loading it at import
time instead would mean the container could not start without the weights, and a
service that cannot start cannot report why.

Invalid SMILES are found before the model runs, not after. ``ADMETModel.predict``
drops what RDKit cannot parse and returns a frame indexed by the survivors, so a
caller scoring thirty candidates would otherwise have to infer which one went
missing -- and an all-invalid batch reaches the trainer with an empty dataloader.
Parsing here makes both cases explicit.

Failure is a value, not an exception. If the weights are missing or torch is
unusable, ``/health`` says so with the reason attached, and predictions fail with
that same sentence. The alternative -- a container that dies on start -- shows up
as a Docker restart loop several layers away from anything actionable.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import Any, Optional

from .properties import CORE_PANEL, resolve

logger = logging.getLogger(__name__)

# Overridable so a deployment can mount the ensembles rather than baking them
# into the image. Empty means "wherever the installed admet_ai package put them".
MODELS_DIR = os.environ.get("ADMET_MODELS_DIR", "").strip()

# DrugBank percentiles are ADMET-AI's context feature: for each property, where
# this molecule sits among approved drugs. Genuinely useful, and free once the
# reference set is loaded, but it doubles the column count.
INCLUDE_DRUGBANK_PERCENTILES = os.environ.get(
    "ADMET_DRUGBANK_PERCENTILES", "true"
).lower() in ("true", "1", "yes")

# Suffix ADMET-AI appends to a property name for its percentile column when no
# ATC filter is applied.
PERCENTILE_SUFFIX = "_drugbank_approved_percentile"


class ModelUnavailableError(RuntimeError):
    """The ensemble could not be loaded, with a caller-facing explanation."""


class AdmetPredictor:
    """A lazily-loaded ADMET-AI ensemble.

    Thread-safe: uvicorn runs the sync prediction path in a worker thread, and
    two concurrent first requests must not both build the ensemble.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._load_error: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def _build(self) -> Any:
        """Construct the ADMETModel, translating any failure into a message."""
        try:
            from admet_ai import ADMETModel
        except ImportError as exc:
            raise ModelUnavailableError(
                f"The admet_ai package is not importable in this container: "
                f"{exc}"
            ) from exc

        kwargs: dict[str, Any] = {}
        if MODELS_DIR:
            kwargs["models_dir"] = MODELS_DIR
        if not INCLUDE_DRUGBANK_PERCENTILES:
            kwargs["drugbank_path"] = None

        try:
            return ADMETModel(**kwargs)
        except FileNotFoundError as exc:
            raise ModelUnavailableError(
                f"ADMET-AI model weights were not found ({exc}). Set "
                "ADMET_MODELS_DIR to a directory of Chemprop ensembles, or "
                "rebuild the image so the admet-ai package data is present."
            ) from exc
        except Exception as exc:  # noqa: BLE001 - the reason is the payload
            raise ModelUnavailableError(
                f"Failed to load the ADMET-AI ensemble: {type(exc).__name__}: "
                f"{exc}"
            ) from exc

    def ensure_loaded(self) -> Any:
        """Return the loaded model, building it on first use.

        Raises:
            ModelUnavailableError: With the reason, on every call after a failed
                load attempt -- retried each time, since a missing mount can be
                fixed without restarting the container.
        """
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            logger.info("Loading ADMET-AI ensemble (this takes a moment)...")
            try:
                model = self._build()
            except ModelUnavailableError as exc:
                self._load_error = str(exc)
                logger.error("ADMET-AI unavailable: %s", exc)
                raise
            self._model = model
            self._load_error = None
            logger.info(
                "ADMET-AI ensemble loaded: %d multitask ensembles on %s",
                getattr(model, "num_ensembles", -1),
                getattr(model, "device", "?"),
            )
            return model

    def available_columns(self) -> list[str]:
        """Property columns this ensemble actually produces.

        Read off the loaded model rather than assumed from the catalog, so a
        deployment with a partial set of ensembles reports what it has.
        """
        model = self.ensure_loaded()
        columns: list[str] = []
        for tasks in getattr(model, "task_lists", []):
            columns.extend(tasks)
        return columns

    @staticmethod
    def _partition(smiles: list[str]) -> tuple[list[str], list[str]]:
        """Split inputs into what RDKit can parse and what it cannot.

        Duplicates are collapsed: every copy would score identically, and a
        repeated index turns ``frame.loc[smiles]`` into a sub-frame rather than
        a row.
        """
        from rdkit import Chem
        from rdkit import RDLogger

        # RDKit writes parse failures to stderr, which is noise here: the caller
        # is told which SMILES failed, in the response.
        RDLogger.DisableLog("rdApp.error")

        valid: list[str] = []
        invalid: list[str] = []
        seen: set[str] = set()
        for entry in smiles:
            if entry in seen:
                continue
            seen.add(entry)
            if Chem.MolFromSmiles(entry) is None:
                invalid.append(entry)
            else:
                valid.append(entry)
        return valid, invalid

    def predict(
        self,
        smiles: list[str],
        properties: Optional[list[str]] = None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Predict properties for a batch of SMILES.

        Args:
            smiles: Molecules to score. Order is preserved in the result.
            properties: Canonical property names to keep, or None for the core
                panel. Names not produced by the ensemble are simply absent
                from each molecule's result rather than reported as an error --
                the ensemble's column set is the authority, and the caller can
                see what it got.

        Returns:
            ``(results, invalid)`` where each result is
            ``{"smiles": ..., "properties": {...}, "percentiles": {...}}`` and
            ``invalid`` lists the inputs RDKit could not parse.

        Raises:
            ModelUnavailableError: If the ensemble cannot be loaded.
        """
        model = self.ensure_loaded()

        valid, invalid = self._partition(smiles)
        if not valid:
            return [], invalid

        frame = model.predict(smiles=valid)

        keep = properties if properties is not None else list(CORE_PANEL)

        results: list[dict[str, Any]] = []
        for entry in valid:
            if entry not in frame.index:
                # Should not happen now that parsing is done up front, but a
                # missing row must not become a KeyError mid-batch.
                invalid.append(entry)
                continue
            results.append(self._row_to_result(entry, frame.loc[entry], keep))

        return results, invalid

    @staticmethod
    def _clean(value: Any) -> Optional[float]:
        """Coerce a prediction to a finite float, or None."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return None if math.isnan(number) or math.isinf(number) else number

    @classmethod
    def _row_to_result(
        cls, smiles: str, row: Any, keep: list[str]
    ) -> dict[str, Any]:
        """Turn one prediction row into a result dict."""
        properties: dict[str, float] = {}
        percentiles: dict[str, float] = {}

        for name in keep:
            prop = resolve(name)
            column = prop.name if prop else name
            if column in row.index:
                value = cls._clean(row[column])
                if value is not None:
                    properties[column] = value

            percentile_column = f"{column}{PERCENTILE_SUFFIX}"
            if percentile_column in row.index:
                percentile = cls._clean(row[percentile_column])
                if percentile is not None:
                    percentiles[column] = percentile

        return {
            "smiles": smiles,
            "properties": properties,
            "percentiles": percentiles,
        }


_predictor: Optional[AdmetPredictor] = None


def get_predictor() -> AdmetPredictor:
    """Return the process-wide predictor."""
    global _predictor
    if _predictor is None:
        _predictor = AdmetPredictor()
    return _predictor
