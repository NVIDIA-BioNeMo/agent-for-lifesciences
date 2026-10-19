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

import {
  useState,
  useRef,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  type ReactNode,
} from "react";

/**
 * Artifact data from the backend storage API.
 */
export interface Artifact {
  reference: string;
  data_type: string;
  description: string | null;
  metadata: Record<string, unknown>;
  created_at: string | null;
}

interface StorageStripProps {
  /** Callback when user clicks an artifact to inject its reference */
  onInjectRef: (reference: string) => void;
  /** Artifacts to display (if provided, component won't fetch its own) */
  artifacts?: Artifact[];
  /** Loading state (used when artifacts are provided externally) */
  isLoading?: boolean;
}

/**
 * Artifacts of a single data type, shown as one chip in the strip.
 */
interface ArtifactGroup {
  /** Normalized (lowercase) data type shared by every entry. */
  dataType: string;
  /** Entries newest first, preserving the order the backend returned. */
  entries: Artifact[];
}

interface PopoverState {
  group: ArtifactGroup;
  /** Anchor rect of the chip the popover belongs to. */
  anchor: DOMRect;
  /** Pinned popovers ignore the hover close timer. */
  pinned: boolean;
}

/**
 * 4x4 checkerboard SVG icon evoking an MSA visualization.
 */
const MSA_ICON = (
  <svg
    width="16"
    height="16"
    viewBox="0 0 16 16"
    style={{ display: "inline-block", verticalAlign: "middle" }}
  >
    <rect x="0" y="0" width="4" height="4" fill="#5B9BD5" />
    <rect x="4" y="0" width="4" height="4" fill="#BE4BDB" />
    <rect x="8" y="0" width="4" height="4" fill="#20C997" />
    <rect x="12" y="0" width="4" height="4" fill="#7B5EA7" />
    <rect x="0" y="4" width="4" height="4" fill="#A29BFE" />
    <rect x="4" y="4" width="4" height="4" fill="#51CF66" />
    <rect x="8" y="4" width="4" height="4" fill="#5B9BD5" />
    <rect x="12" y="4" width="4" height="4" fill="#D170DB" />
    <rect x="0" y="8" width="4" height="4" fill="#38D9A9" />
    <rect x="4" y="8" width="4" height="4" fill="#8E5FAD" />
    <rect x="8" y="8" width="4" height="4" fill="#A29BFE" />
    <rect x="12" y="8" width="4" height="4" fill="#51CF66" />
    <rect x="0" y="12" width="4" height="4" fill="#BE4BDB" />
    <rect x="4" y="12" width="4" height="4" fill="#20C997" />
    <rect x="8" y="12" width="4" height="4" fill="#D170DB" />
    <rect x="12" y="12" width="4" height="4" fill="#5B9BD5" />
  </svg>
);

/**
 * Get the icon for a given data type.
 */
function getArtifactIcon(dataType: string): ReactNode {
  switch (dataType.toLowerCase()) {
    case "pdb":
    case "mmcif":
      return "🔬";
    case "annotated_cif":
      // A structure carrying design conditions — the hotspots it marks are
      // what distinguishes it from the plain mmCIF beside it in the strip.
      return "🎯";
    case "sequence":
      return "🧬";
    case "sdf":
    case "mol2":
      return "⬡";
    case "safe":
      return "🧩";
    case "selection":
      return "✦";
    case "design_spec":
      return "🧭";
    case "msa":
      return MSA_ICON;
    case "skill":
      return "📖";
    default:
      return "📦";
  }
}

/**
 * Collapse skill artifacts into a single entry per skill.
 *
 * Installing a skill creates multiple `skill_file` artifacts (the SKILL.md
 * body plus every bundled resource, however deeply nested). We show one
 * "SKILL" chip per skill rather than one box per file, grouping by the
 * skill_id recorded in each artifact's metadata.
 */
function collapseSkillArtifacts(artifacts: Artifact[]): Artifact[] {
  const result: Artifact[] = [];
  const skillGroups = new Map<string, Artifact[]>();

  for (const artifact of artifacts) {
    if (artifact.data_type.toLowerCase() === "skill_file") {
      const key = String(artifact.metadata?.skill_id ?? "skill");
      const group = skillGroups.get(key) ?? [];
      group.push(artifact);
      skillGroups.set(key, group);
    } else {
      result.push(artifact);
    }
  }

  for (const group of skillGroups.values()) {
    // Prefer the body artifact as the representative of the skill.
    const representative =
      group.find((a) => a.metadata?.role === "body") ?? group[0];
    const skillName = representative.metadata?.skill_name;
    result.push({
      ...representative,
      data_type: "skill",
      description: skillName ? String(skillName) : "Skill",
    });
  }

  return result;
}

/**
 * Bucket artifacts by data type so each type occupies a single chip.
 *
 * A Map preserves insertion order, so a group sits where its newest member
 * would have: the backend returns artifacts newest first, and entries keep
 * that order within their group.
 */
function groupArtifacts(artifacts: Artifact[]): ArtifactGroup[] {
  const groups = new Map<string, ArtifactGroup>();

  for (const artifact of artifacts) {
    const dataType = artifact.data_type.toLowerCase();
    const existing = groups.get(dataType);
    if (existing) {
      existing.entries.push(artifact);
    } else {
      groups.set(dataType, { dataType, entries: [artifact] });
    }
  }

  return [...groups.values()];
}

/**
 * Get a short label for a given data type.
 */
function getArtifactLabel(dataType: string): string {
  switch (dataType.toLowerCase()) {
    case "pdb":
      return "PDB";
    case "mmcif":
      return "CIF";
    case "annotated_cif":
      return "CIF+";
    case "sequence":
      return "SEQ";
    case "sdf":
      return "SDF";
    case "safe":
      return "SAFE";
    case "mol2":
      return "MOL2";
    case "selection":
      return "SEL";
    case "design_spec":
      return "SPEC";
    case "msa":
      return "MSA";
    case "skill":
      return "SKILL";
    default:
      return dataType.toUpperCase().slice(0, 4);
  }
}

/**
 * Extract a description from an artifact.
 *
 * Prefers the LLM-provided description, then falls back to metadata or type
 * defaults. Returned untruncated — the popover clamps it to two lines in CSS.
 */
function getArtifactDescription(artifact: Artifact): string {
  const { description, metadata, data_type } = artifact;

  // Prefer the LLM-provided description
  if (description) {
    return description;
  }

  // SAFE is handled before the metadata sweep below: a SAFE artifact carries
  // the SMILES it was encoded from under `metadata.smiles`, and showing that
  // would label a SAFE chip with a plain SMILES string. Its fragment count is
  // the summary that actually distinguishes it.
  if (data_type.toLowerCase() === "safe") {
    const fragments = Number(metadata.num_fragments);
    const masked = Boolean(metadata.masked);
    const parts: string[] = [];
    if (fragments) {
      parts.push(`${fragments} fragment${fragments === 1 ? "" : "s"}`);
    }
    if (masked) {
      parts.push("masked");
    }
    return parts.length ? `SAFE: ${parts.join(", ")}` : "SAFE fragment notation";
  }

  // Try common metadata keys for a description
  for (const key of [
    "name",
    "label",
    "gene_name",
    "molecule_name",
    "smiles",
    "summary",
  ]) {
    if (metadata[key]) {
      return String(metadata[key]);
    }
  }

  // Fallback to type-specific defaults
  switch (data_type.toLowerCase()) {
    case "pdb":
    case "mmcif":
      return "Protein structure";
    case "annotated_cif":
      return "Structure with design conditions";
    case "sequence":
      return "Protein sequence";
    case "sdf":
    case "mol2":
      return "Molecule structure";
    case "selection": {
      const atomCount = Number(metadata.atom_count);
      const residueCount = Number(metadata.residue_count);
      if (atomCount && residueCount) {
        return (
          `${atomCount} atom${atomCount === 1 ? "" : "s"}, ` +
          `${residueCount} residue${residueCount === 1 ? "" : "s"}`
        );
      }
      return "Structure selection";
    }
    case "design_spec": {
      const layerCount = Number(metadata.layer_count);
      return layerCount
        ? `${layerCount} condition layer${layerCount === 1 ? "" : "s"}`
        : "Design specification";
    }
    case "msa":
      const numSeqs = metadata.num_sequences;
      if (numSeqs) return `${numSeqs} sequences`;
      return "Multiple sequence alignment";
    default:
      return `${data_type.toUpperCase()} data`;
  }
}

/** Gap between a chip and its popover, and the minimum viewport margin. */
const POPOVER_GAP = 8;
const VIEWPORT_MARGIN = 8;

/** Grace period that lets the pointer travel from a chip into its popover. */
const POPOVER_CLOSE_DELAY_MS = 150;

/**
 * Individual chip representing one group of same-typed artifacts.
 *
 * A group holding a single entry injects its reference on click. A group
 * holding several only opens the popover, where each entry is injected
 * individually.
 */
function ArtifactItem({
  group,
  isOpen,
  onInjectRef,
  onHoverStart,
  onHoverEnd,
  onActivate,
}: {
  group: ArtifactGroup;
  isOpen: boolean;
  onInjectRef: (ref: string) => void;
  onHoverStart: (group: ArtifactGroup, element: HTMLElement) => void;
  onHoverEnd: () => void;
  onActivate: (group: ArtifactGroup, element: HTMLElement) => void;
}) {
  const itemRef = useRef<HTMLDivElement>(null);
  const count = group.entries.length;
  const isGrouped = count > 1;
  const icon = getArtifactIcon(group.dataType);
  const label = getArtifactLabel(group.dataType);

  const activate = () => {
    if (isGrouped) {
      if (itemRef.current) {
        onActivate(group, itemRef.current);
      }
    } else {
      onInjectRef(group.entries[0].reference);
    }
  };

  return (
    <div
      ref={itemRef}
      className="artifact-item"
      onMouseEnter={() => {
        if (itemRef.current) {
          onHoverStart(group, itemRef.current);
        }
      }}
      onMouseLeave={onHoverEnd}
      onClick={activate}
      role="button"
      tabIndex={0}
      aria-label={`${label} — ${count} stored artifact${count === 1 ? "" : "s"}`}
      aria-haspopup={isGrouped ? "menu" : undefined}
      aria-expanded={isGrouped ? isOpen : undefined}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          activate();
        }
      }}
    >
      <span className="artifact-icon">{icon}</span>
      <span className="artifact-label">{label}</span>
      {isGrouped && <span className="artifact-count-pill">{count}</span>}
    </div>
  );
}

/**
 * Floating panel listing every entry in a group.
 *
 * Positioned against the chip's rect, clamped into the viewport and flipped
 * above the chip when there is no room below.
 */
function ArtifactPopover({
  state,
  onInjectRef,
  onHoverStart,
  onHoverEnd,
}: {
  state: PopoverState;
  onInjectRef: (ref: string) => void;
  onHoverStart: () => void;
  onHoverEnd: () => void;
}) {
  const panelRef = useRef<HTMLDivElement>(null);
  const [placement, setPlacement] = useState<{
    left: number;
    top: number;
    above: boolean;
    arrowLeft: number;
  } | null>(null);

  const { group, anchor } = state;

  useLayoutEffect(() => {
    const panel = panelRef.current;
    if (!panel) return;

    const rect = panel.getBoundingClientRect();
    const anchorCenter = anchor.left + anchor.width / 2;

    const maxLeft = window.innerWidth - rect.width - VIEWPORT_MARGIN;
    const left = Math.max(
      VIEWPORT_MARGIN,
      Math.min(anchorCenter - rect.width / 2, maxLeft)
    );

    const above =
      anchor.bottom + POPOVER_GAP + rect.height >
      window.innerHeight - VIEWPORT_MARGIN;
    const top = above
      ? anchor.top - POPOVER_GAP - rect.height
      : anchor.bottom + POPOVER_GAP;

    setPlacement({ left, top, above, arrowLeft: anchorCenter - left });
  }, [anchor, group]);

  const label = getArtifactLabel(group.dataType);
  const count = group.entries.length;

  return (
    <div
      ref={panelRef}
      data-artifact-popover
      className={`artifact-popover${placement?.above ? " is-above" : ""}`}
      style={{
        position: "fixed",
        left: placement?.left ?? 0,
        top: placement?.top ?? 0,
        // Hidden for the single frame before the layout effect measures it.
        visibility: placement ? "visible" : "hidden",
        ["--arrow-left" as string]: `${placement?.arrowLeft ?? 0}px`,
      }}
      onMouseEnter={onHoverStart}
      onMouseLeave={onHoverEnd}
    >
      <div className="artifact-popover-header">
        <span className="artifact-popover-type">{label}</span>
        <span className="artifact-popover-count">{count} stored</span>
      </div>

      <div className="artifact-popover-entries" role="menu">
        {group.entries.map((entry) => (
          <div
            key={entry.reference}
            className="artifact-popover-entry"
            role="menuitem"
            tabIndex={0}
            onClick={() => onInjectRef(entry.reference)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onInjectRef(entry.reference);
              }
            }}
          >
            <div className="artifact-popover-description">
              {getArtifactDescription(entry)}
            </div>
            <code className="artifact-popover-reference">{entry.reference}</code>
          </div>
        ))}
      </div>

      <div className="artifact-popover-hint">
        {count > 1
          ? "Click an entry to insert its reference"
          : "Click to insert its reference"}
      </div>
    </div>
  );
}

/**
 * Horizontal strip showing stored data artifacts.
 *
 * Artifacts are grouped into one chip per data type, badged with a count when
 * the group holds several. Hovering a chip opens a popover listing every entry
 * with its description; clicking an entry injects its reference into the chat
 * input. Hidden when no artifacts exist.
 */
export default function StorageStrip({
  onInjectRef,
  artifacts = [],
  isLoading = false,
}: StorageStripProps) {
  const [popover, setPopover] = useState<PopoverState | null>(null);
  const closeTimerRef = useRef<number | null>(null);

  const cancelClose = useCallback(() => {
    if (closeTimerRef.current !== null) {
      window.clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }
  }, []);

  const dismiss = useCallback(() => {
    cancelClose();
    setPopover(null);
  }, [cancelClose]);

  const scheduleClose = useCallback(() => {
    cancelClose();
    closeTimerRef.current = window.setTimeout(() => {
      closeTimerRef.current = null;
      setPopover((current) => (current?.pinned ? current : null));
    }, POPOVER_CLOSE_DELAY_MS);
  }, [cancelClose]);

  const handleHoverStart = useCallback(
    (group: ArtifactGroup, element: HTMLElement) => {
      cancelClose();
      setPopover((current) => ({
        group,
        anchor: element.getBoundingClientRect(),
        // Keep a pin only while hovering the chip that owns it.
        pinned: current?.pinned === true && current.group.dataType === group.dataType,
      }));
    },
    [cancelClose]
  );

  // Clicking a multi-entry chip toggles the popover pinned open.
  const handleActivate = useCallback(
    (group: ArtifactGroup, element: HTMLElement) => {
      cancelClose();
      setPopover((current) =>
        current?.pinned && current.group.dataType === group.dataType
          ? null
          : { group, anchor: element.getBoundingClientRect(), pinned: true }
      );
    },
    [cancelClose]
  );

  const handleInjectRef = useCallback(
    (reference: string) => {
      dismiss();
      onInjectRef(reference);
    },
    [dismiss, onInjectRef]
  );

  useEffect(() => cancelClose, [cancelClose]);

  // A pinned popover stays until dismissed explicitly. Clicks inside the strip
  // are left to the chip handlers so they can toggle their own popover.
  useEffect(() => {
    if (!popover?.pinned) return;

    const handlePointerDown = (event: MouseEvent) => {
      const target = event.target as Element | null;
      if (target?.closest("[data-artifact-popover]") || target?.closest(".storage-strip")) {
        return;
      }
      dismiss();
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        dismiss();
      }
    };

    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [popover?.pinned, dismiss]);

  // Memoized so group identities stay stable across renders — the popover's
  // layout effect keys off them.
  const groups = useMemo(
    () => groupArtifacts(collapseSkillArtifacts(artifacts)),
    [artifacts]
  );

  // Resolve against the current groups so an open popover follows storage
  // updates that arrive mid-hover rather than showing a stale entry list.
  const activeGroup = popover
    ? groups.find((group) => group.dataType === popover.group.dataType)
    : undefined;

  // Don't render anything if empty or loading
  if (isLoading || groups.length === 0) {
    return null;
  }

  return (
    <div className="storage-strip">
      <div className="storage-strip-label">Storage:</div>
      <div className="storage-strip-items">
        {groups.map((group) => (
          <ArtifactItem
            key={group.dataType}
            group={group}
            isOpen={popover?.group.dataType === group.dataType}
            onInjectRef={handleInjectRef}
            onHoverStart={handleHoverStart}
            onHoverEnd={scheduleClose}
            onActivate={handleActivate}
          />
        ))}
      </div>

      {/* Rendered outside the scroll container with fixed positioning */}
      {popover && activeGroup && (
        <ArtifactPopover
          state={{ ...popover, group: activeGroup }}
          onInjectRef={handleInjectRef}
          onHoverStart={cancelClose}
          onHoverEnd={scheduleClose}
        />
      )}
    </div>
  );
}
