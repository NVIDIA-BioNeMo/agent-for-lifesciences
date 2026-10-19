/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { useEffect, useState } from "react";
import type { PickerState, ResiduePicker } from "../lib/molstar/residuePicker";
import type { KeepPiece, Layout, RoleOption } from "../lib/molstar/roles";

interface Props {
  state: PickerState;
  picker: ResiduePicker | null;
}

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

/** `A21-28`, or `A21` for a single residue, in author numbering. */
const pieceName = (p: KeepPiece) =>
  p.firstAuth === p.lastAuth
    ? `${p.chain}${p.firstAuth}`
    : `${p.chain}${p.firstAuth}-${p.lastAuth}`;

/** Two number boxes for a ``[min, max]`` length, applied when focus leaves them. */
function RangeInput({
  label,
  value,
  onApply,
}: {
  label: string;
  value: [number, number] | undefined;
  onApply: (next: [number, number] | undefined) => void;
}) {
  const [low, setLow] = useState(value ? String(value[0]) : "");
  const [high, setHigh] = useState(value ? String(value[1]) : "");
  useEffect(() => {
    setLow(value ? String(value[0]) : "");
    setHigh(value ? String(value[1]) : "");
  }, [value?.[0], value?.[1]]);

  const apply = () => {
    if (low === "" && high === "") return onApply(undefined);
    const a = Number(low === "" ? high : low);
    const b = Number(high === "" ? low : high);
    if (!Number.isFinite(a) || !Number.isFinite(b) || a < 0 || b < a || b > 200) return;
    if (value && value[0] === a && value[1] === b) return;
    onApply([a, b]);
  };
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") (e.target as HTMLInputElement).blur();
  };
  return (
    <span className="layout-range" title={`${label}: shortest and longest, in residues`}>
      <input
        aria-label={`${label} minimum`}
        value={low}
        inputMode="numeric"
        placeholder="–"
        onChange={(e) => setLow(e.target.value)}
        onBlur={apply}
        onKeyDown={onKey}
      />
      <span>–</span>
      <input
        aria-label={`${label} maximum`}
        value={high}
        inputMode="numeric"
        placeholder="–"
        onChange={(e) => setHigh(e.target.value)}
        onBlur={apply}
        onKeyDown={onKey}
      />
    </span>
  );
}

/**
 * The kept pieces in order, with the lengths designed around and between them.
 *
 * The layout is a parameter of the design, not a role, so it is edited here
 * and rides on the Keep layer.
 */
function LayoutStrip({
  pieces,
  layout,
  onChange,
}: {
  pieces: KeepPiece[];
  layout: Layout | null;
  onChange: (next: Layout) => void;
}) {
  const current: Layout = layout ?? {};
  const set = (patch: Partial<Layout>) => onChange({ ...current, ...patch });
  const setLinker = (index: number, range: [number, number] | undefined) => {
    const linkers = [...(current.linkers ?? [])];
    while (linkers.length < pieces.length - 1) linkers.push([10, 30]);
    linkers[index] = range ?? [10, 30];
    set({ linkers });
  };
  return (
    <div
      className="layout-strip"
      title="Lengths of what is designed around and between the kept pieces (motif scaffolding)"
    >
      <span className="layout-title">Layout</span>
      <RangeInput label="N-terminal" value={current.n_term} onApply={(v) => set({ n_term: v })} />
      {pieces.map((piece, i) => (
        <span className="layout-run" key={`${piece.chain}${piece.firstLabel}`}>
          <span className="layout-piece" title={`${piece.length} residues kept`}>
            {pieceName(piece)}
          </span>
          {i < pieces.length - 1 && (
            <RangeInput
              label={`Linker ${i + 1}`}
              value={current.linkers?.[i] ?? [10, 30]}
              onApply={(v) => setLinker(i, v)}
            />
          )}
        </span>
      ))}
      <RangeInput label="C-terminal" value={current.c_term} onApply={(v) => set({ c_term: v })} />
    </div>
  );
}

/**
 * Picking, giving picks a role, and the tray of roles assigned so far.
 *
 * The buttons come from the backend's role catalog, filtered by what the
 * current pick can be: the five primary roles are always shown (greyed out
 * with a reason when they cannot apply) and everything else appears under
 * More only when valid.
 */
export default function DesignRolesPanel({ state, picker }: Props) {
  const [moreOpen, setMoreOpen] = useState(false);
  const { pending, layers, options } = state;
  const primary = options.filter((o) => o.group === "primary");
  const more = options.filter((o) => o.group === "more");
  const editingRole = state.editingLabel;

  useEffect(() => {
    if (more.length === 0) setMoreOpen(false);
  }, [more.length]);

  const press = (option: RoleOption) => {
    setMoreOpen(false);
    void picker?.commitRole(option.id);
  };

  const status =
    pending.count === 0
      ? "right-click a residue or atom to pick"
      : `${plural(pending.count, "residue")} · ${plural(pending.atomCount, "atom")}${
          pending.dirty ? " · unsent" : ""
        }`;

  return (
    <div className="pick-toolbar" onContextMenu={(e) => e.preventDefault()}>
      <div className="pick-layer pick-pending" title={pending.description || undefined}>
        <span className="pick-swatch pick-swatch-pending" />
        <span className="pick-label">{editingRole ? `Editing ${editingRole}` : "Pick"}</span>
        <span className="pick-count">{status}</span>
        {pending.count > 0 && (
          <>
            <button
              className="pick-action"
              title="Add residues within 4 Å"
              onClick={() => picker?.expand(4)}
            >
              +4 Å
            </button>
            <button className="pick-action" title="Clear picks" onClick={() => picker?.clear()}>
              Clear
            </button>
          </>
        )}
        {state.editingId && (
          <button
            className="pick-action"
            title="Put the role back as it was"
            onClick={() => picker?.cancelEdit()}
          >
            Cancel
          </button>
        )}
        {state.engines.length > 1 && (
          <select
            className="engine-select"
            value={state.engine}
            title="The design engine. Roles it cannot take are greyed out."
            onChange={(e) => picker?.setEngine(e.target.value)}
          >
            {state.engines.map((engine) => (
              <option key={engine.key} value={engine.key} title={engine.note}>
                {engine.label}
              </option>
            ))}
          </select>
        )}
      </div>

      {state.offered && (
        <div className="pick-layer pick-offered">
          <span className="pick-label">{state.offered.label}</span>
          <span className="pick-count">
            {plural(state.offered.residueCount, "residue")} selected by the assistant
          </span>
          <button
            className="pick-action"
            disabled={state.busy}
            title="Take this selection as the pick, then choose a role for it"
            onClick={() => picker?.useOffered()}
          >
            Use as…
          </button>
        </div>
      )}

      {primary.length > 0 && (
        <div className="role-buttons">
          {primary.map((option) => (
            <button
              key={option.id}
              className="role-button"
              style={{ borderLeftColor: option.colour }}
              disabled={!option.enabled || state.busy}
              title={option.enabled ? option.tooltip : `${option.reason ?? ""}\n${option.tooltip}`}
              onClick={() => press(option)}
            >
              {option.label}
            </button>
          ))}
          {more.length > 0 && (
            <span className="role-more">
              <button
                className="role-button role-button-more"
                aria-expanded={moreOpen}
                onClick={() => setMoreOpen((open) => !open)}
                title="Other roles that are valid for this pick"
              >
                More ▾
              </button>
              {moreOpen && (
                <div className="role-more-menu">
                  {more.map((option) => (
                    <button
                      key={option.id}
                      className="role-button"
                      style={{ borderLeftColor: option.colour }}
                      disabled={state.busy}
                      title={option.tooltip}
                      onClick={() => press(option)}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              )}
            </span>
          )}
        </div>
      )}

      {layers.length > 0 && (
        <div className="role-tray">
          {layers.map((layer) => (
            <span
              key={layer.id}
              className="role-chip"
              style={{ borderColor: layer.colour }}
              title={`${layer.description}\nClick to edit`}
              onClick={() => picker?.editLayer(layer.id)}
            >
              <span className="pick-swatch" style={{ backgroundColor: layer.colour }} />
              <span className="pick-label">{layer.label}</span>
              <span className="pick-count">{plural(layer.count, "res")}</span>
              <button
                className="chip-remove"
                aria-label={`Remove ${layer.label}`}
                title="Remove this role"
                onClick={(e) => {
                  e.stopPropagation();
                  void picker?.removeLayer(layer.id);
                }}
              >
                ×
              </button>
            </span>
          ))}
        </div>
      )}

      {state.pieces.length > 0 && (
        <LayoutStrip
          pieces={state.pieces}
          layout={state.layout}
          onChange={(next) => void picker?.setLayout(next)}
        />
      )}

      {state.notice && <div className="pick-notice">{state.notice}</div>}
      <div className="pick-hint">
        Left-click a residue to show its atoms, right-click an atom to pick it · Right-click the
        ribbon for the whole residue · Shift: extend along chain · Ctrl: whole chain
      </div>
    </div>
  );
}
