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
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
  ReactNode,
} from "react";
import { createRoot } from "react-dom/client";
import { createPluginUI } from "molstar/lib/mol-plugin-ui";
import { PluginUIContext } from "molstar/lib/mol-plugin-ui/context";
import { PluginConfig } from "molstar/lib/mol-plugin/config";
import { Color } from "molstar/lib/mol-util/color";
import type { Expression } from "molstar/lib/mol-script/language/expression";

import "molstar/lib/mol-plugin-ui/skin/dark.scss";

// Import query utilities
import {
  ligandQuery,
  residuesNearLigandQuery,
  chainQuery,
  residueQuery,
  atomQuery,
  polymerQuery,
  backboneQuery,
  sidechainQuery,
  alphaCarbonQuery,
  withinDistanceQuery,
} from "../lib/molstar/queries";
import type { LabelAtom, LabelResidue } from "../lib/molstar/queries";
import { ResiduePicker, withRightClickFreed, type PickerState } from "../lib/molstar/residuePicker";
import DesignRolesPanel from "./DesignRolesPanel";
import { disableDepthClipping } from "../lib/molstar/clipping";
import {
  applyViewerStyle,
  registerStyleThemes,
  VIEWER_COLOUR_MODE,
  VIEWER_STYLE,
  type ChainRoles,
} from "../lib/molstar/styles";
import {
  showResidues,
  showAtoms,
  paintLayers,
  clearPaint,
  type PaintLayer,
  type ShownCounts,
} from "../lib/molstar/residues";

import {
  queryStructure,
  getResidueInfo,
  getAtomInfo,
  highlightSelection,
  clearHighlights,
  selectAtoms,
  clearSelection,
  focusOnSelection,
  measureDistance,
  measureAngle,
  measureDihedral,
  getCenterOfMass,
  getBoundingBox,
  getViewerState,
} from "../lib/molstar/selections";

import type {
  QueryResult,
  ResidueInfo,
  AtomInfo,
  HighlightOptions,
  ChainSelector,
  ResidueSelector,
  AtomSelector,
  ViewerStateInfo,
} from "../lib/molstar/types";

// Adapter for React 18's createRoot to match Molstar's expected render signature
function renderReact18(component: ReactNode, container: Element) {
  const root = createRoot(container);
  root.render(component);
  return root;
}

/**
 * Structure text formats the viewer can parse. These are Molstar's own
 * `parseTrajectory` format ids.
 */
export type StructureFormat = "pdb" | "mmcif";

/**
 * Extended interface for programmatic interaction with the Molstar viewer.
 */
export interface MolstarViewerRef {
  // === Loading Methods ===
  /** Load a molecule from SDF data */
  loadSdf: (sdfData: string, label?: string) => Promise<void>;
  /**
   * Load a protein structure. Boltz-2 emits mmCIF and nothing else, so the
   * format has to be passed rather than assumed.
   */
  loadPdb: (
    pdbData: string,
    label?: string,
    format?: StructureFormat,
    roles?: ChainRoles | null
  ) => Promise<void>;
  /** Clear all loaded structures */
  clear: () => Promise<void>;

  // === Query Methods ===
  /** Execute a custom query expression and return results */
  query: (expression: Expression) => QueryResult | null;
  /** Get residues matching a query */
  getResidues: (expression: Expression) => ResidueInfo[];
  /** Get atoms matching a query */
  getAtoms: (expression: Expression) => AtomInfo[];

  // === Preset Queries ===
  /** Select ligand (non-polymer) atoms */
  selectLigand: () => QueryResult | null;
  /** Select residues within distance of ligand */
  selectResiduesNearLigand: (radiusAngstroms: number) => QueryResult | null;
  /** Select atoms in a specific chain */
  selectChain: (chainId: string) => QueryResult | null;
  /** Select specific residue(s) */
  selectResidues: (
    selector: ResidueSelector,
    chainId?: string
  ) => QueryResult | null;
  /** Select backbone atoms only */
  selectBackbone: () => QueryResult | null;
  /** Select sidechain atoms only */
  selectSidechain: () => QueryResult | null;
  /** Select alpha carbon atoms */
  selectAlphaCarbons: () => QueryResult | null;

  // === Visual Methods ===
  /** Highlight atoms matching a query */
  highlight: (
    expression: Expression,
    options?: HighlightOptions
  ) => Promise<void>;
  /** Clear all highlights */
  clearHighlights: () => void;
  /** Create a persistent selection */
  select: (
    expression: Expression,
    options?: { add?: boolean }
  ) => Promise<void>;
  /** Clear the current selection */
  clearSelection: () => void;
  /** Focus camera on atoms matching a query */
  focusOn: (expression: Expression) => Promise<void>;
  /**
   * Select residues given by label ids, replacing the current selection.
   * Returns how many were found in the loaded structure.
   */
  showResidues: (residues: LabelResidue[], options?: { focus?: boolean }) => number;
  /**
   * Select whole residues plus individual atoms of other residues, replacing
   * the current selection. Used for selections with only some atoms chosen.
   */
  showAtoms: (
    residues: LabelResidue[],
    atoms: LabelAtom[],
    options?: { focus?: boolean }
  ) => ShownCounts;
  /**
   * Offer a selection the assistant made as something the user can take as a
   * pick and give a role ("Use as…"). Replaces any earlier offer.
   */
  offerSelection: (label: string, residues: LabelResidue[], atoms: LabelAtom[]) => void;
  /** Paint colour layers over the structure, replacing earlier ones. */
  paintLayers: (layers: PaintLayer[]) => Promise<void>;
  /** Remove painted colour layers. */
  clearPaint: () => Promise<void>;
  /**
   * Store the residues the user has picked (right mouse button) as selections
   * on the backend, if they changed since last stored. Resolves to the new
   * references; never rejects, because the user's message must still go out.
   */
  commitPicks: (sessionId: string | undefined) => Promise<string[]>;

  // === Measurement Methods ===
  /** Calculate distance between two atoms */
  measureDistance: (atom1: AtomInfo, atom2: AtomInfo) => number;
  /** Calculate angle between three atoms */
  measureAngle: (atom1: AtomInfo, atom2: AtomInfo, atom3: AtomInfo) => number;
  /** Calculate dihedral angle between four atoms */
  measureDihedral: (
    atom1: AtomInfo,
    atom2: AtomInfo,
    atom3: AtomInfo,
    atom4: AtomInfo
  ) => number;
  /** Get center of mass of a selection */
  getCenterOfMass: (expression: Expression) => [number, number, number] | null;
  /** Get bounding box of a selection */
  getBoundingBox: (
    expression: Expression
  ) => { min: [number, number, number]; max: [number, number, number] } | null;

  // === Query Builders (for custom queries) ===
  queries: {
    ligand: () => Expression;
    polymer: () => Expression;
    backbone: () => Expression;
    sidechain: () => Expression;
    alphaCarbons: () => Expression;
    chain: (selector: ChainSelector) => Expression;
    residue: (selector: ResidueSelector, chain?: ChainSelector) => Expression;
    atom: (
      selector: AtomSelector,
      residue?: ResidueSelector,
      chain?: ChainSelector
    ) => Expression;
    withinDistance: (
      target: Expression,
      radiusAngstroms: number,
      wholeResidues?: boolean
    ) => Expression;
    residuesNearLigand: (radiusAngstroms: number) => Expression;
  };

  // === State Query (for LLM context) ===
  /** Get the current viewer state for LLM context */
  getState: () => ViewerStateInfo;

  // === Direct Plugin Access (advanced usage) ===
  getPlugin: () => PluginUIContext | null;
}

interface MolstarViewerProps {
  className?: string;
  /** The session role selections are stored under, so a role is visible to the agent at once. */
  sessionId?: string;
}

const MolstarViewer = forwardRef<MolstarViewerRef, MolstarViewerProps>(
  ({ className, sessionId }, ref) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const pluginRef = useRef<PluginUIContext | null>(null);
    // Track labels and formats for loaded structures (keyed by structure hash)
    const structureLabelsRef = useRef<Map<string, { label: string; format: string }>>(new Map());
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [isEmpty, setIsEmpty] = useState(true);
    const pickerRef = useRef<ResiduePicker | null>(null);
    const sessionIdRef = useRef<string | undefined>(sessionId);
    useEffect(() => {
      sessionIdRef.current = sessionId;
      pickerRef.current?.setSession(sessionId);
    }, [sessionId]);
    const [pickerState, setPickerState] = useState<PickerState | null>(null);

    // Chain roles of the loaded design, if it is one: target and binder.
    const rolesRef = useRef<ChainRoles | null>(null);

    const applyLook = async () => {
      const plugin = pluginRef.current;
      if (!plugin) return;
      try {
        await applyViewerStyle(plugin, VIEWER_STYLE, VIEWER_COLOUR_MODE, rolesRef.current);
      } catch (err) {
        console.error("Failed to apply viewer style:", err);
      }
    };

    useEffect(() => {
      const initPlugin = async () => {
        if (!containerRef.current || pluginRef.current) return;

        try {
          const plugin = await createPluginUI({
            target: containerRef.current,
            render: renderReact18,
            spec: {
              // Right-click is for picking residues, not re-centring the camera.
              ...withRightClickFreed(),
              config: [[PluginConfig.VolumeStreaming.Enabled, false]],
              layout: {
                initial: {
                  isExpanded: false,
                  showControls: false,
                  controlsDisplay: "reactive",
                },
              },
            },
          });

          // Set dark background to match theme
          const renderer = plugin.canvas3d?.props.renderer;
          if (renderer) {
            plugin.canvas3d?.setProps({
              renderer: {
                ...renderer,
                backgroundColor: Color(0x0a0e14),
              },
            });
          }

          registerStyleThemes(plugin);
          // A left click focuses a residue; do not clip or fog the rest of the scene.
          disableDepthClipping(plugin);
          pluginRef.current = plugin;

          const picker = new ResiduePicker(plugin, setPickerState);
          picker.attach();
          picker.setSession(sessionIdRef.current);
          pickerRef.current = picker;
          setPickerState(picker.getState());
          // The role catalog is the backend's; without it the buttons stay hidden.
          fetch("/api/design-roles")
            .then((response) => (response.ok ? response.json() : null))
            .then((payload) => {
              if (payload && pickerRef.current === picker) picker.setRegistry(payload);
            })
            .catch((err) => console.error("Failed to load design roles:", err));
        } catch (err) {
          console.error("Failed to initialize Molstar:", err);
          setError("Failed to initialize 3D viewer");
        }
      };

      initPlugin();

      return () => {
        pickerRef.current?.detach();
        pickerRef.current = null;
        if (pluginRef.current) {
          pluginRef.current.dispose();
          pluginRef.current = null;
        }
      };
    }, []);

    useImperativeHandle(ref, () => ({
      // === Loading Methods ===
      loadSdf: async (sdfData: string, label?: string) => {
        const plugin = pluginRef.current;
        if (!plugin) {
          throw new Error("Plugin not initialized");
        }

        setIsLoading(true);
        setError(null);

        try {
          // Clear existing structures and labels
          await plugin.clear();
          structureLabelsRef.current.clear();
          pickerRef.current?.reset();

          const structureLabel = label || "Molecule";

          // Load SDF data
          const data = await plugin.builders.data.rawData({
            data: sdfData,
            label: structureLabel,
          });

          const trajectory = await plugin.builders.structure.parseTrajectory(
            data,
            "mol"
          );

          await plugin.builders.structure.hierarchy.applyPreset(
            trajectory,
            "default"
          );

          // Store the label for this structure
          const structures = plugin.managers.structure.hierarchy.current.structures;
          if (structures.length > 0) {
            const structure = structures[0].cell?.obj?.data;
            if (structure) {
              structureLabelsRef.current.set(structure.hashCode.toString(), {
                label: structureLabel,
                format: "sdf",
              });
            }
          }

          rolesRef.current = null;
          await applyLook();

          // Reset camera to focus on the molecule
          plugin.canvas3d?.requestCameraReset();

          setIsEmpty(false);
        } catch (err) {
          console.error("Failed to load SDF:", err);
          setError("Failed to load molecule");
          throw err;
        } finally {
          setIsLoading(false);
        }
      },

      loadPdb: async (
        pdbData: string,
        label?: string,
        format: StructureFormat = "pdb",
        roles: ChainRoles | null = null
      ) => {
        const plugin = pluginRef.current;
        if (!plugin) {
          throw new Error("Plugin not initialized");
        }

        setIsLoading(true);
        setError(null);

        try {
          // Clear existing structures and labels
          await plugin.clear();
          structureLabelsRef.current.clear();
          pickerRef.current?.reset();

          const structureLabel = label || "Protein Structure";

          // Load structure data
          const data = await plugin.builders.data.rawData({
            data: pdbData,
            label: structureLabel,
          });

          const trajectory = await plugin.builders.structure.parseTrajectory(
            data,
            format
          );

          await plugin.builders.structure.hierarchy.applyPreset(
            trajectory,
            "default"
          );

          // Store the label for this structure
          const structures = plugin.managers.structure.hierarchy.current.structures;
          if (structures.length > 0) {
            const structure = structures[0].cell?.obj?.data;
            if (structure) {
              structureLabelsRef.current.set(structure.hashCode.toString(), {
                label: structureLabel,
                format,
              });
            }
          }

          rolesRef.current = roles;
          await applyLook();

          // Reset camera to focus on the structure
          plugin.canvas3d?.requestCameraReset();

          setIsEmpty(false);
        } catch (err) {
          console.error(`Failed to load ${format}:`, err);
          setError("Failed to load protein structure");
          throw err;
        } finally {
          setIsLoading(false);
        }
      },

      clear: async () => {
        const plugin = pluginRef.current;
        if (plugin) {
          await plugin.clear();
          structureLabelsRef.current.clear();
          pickerRef.current?.reset();
          setIsEmpty(true);
          setError(null);
        }
      },

      // === Query Methods ===
      query: (expression: Expression) => {
        const plugin = pluginRef.current;
        if (!plugin) return null;
        return queryStructure(plugin, expression);
      },

      getResidues: (expression: Expression) => {
        const plugin = pluginRef.current;
        if (!plugin) return [];
        return getResidueInfo(plugin, expression);
      },

      getAtoms: (expression: Expression) => {
        const plugin = pluginRef.current;
        if (!plugin) return [];
        return getAtomInfo(plugin, expression);
      },

      // === Preset Queries ===
      selectLigand: () => {
        const plugin = pluginRef.current;
        if (!plugin) return null;
        return queryStructure(plugin, ligandQuery());
      },

      selectResiduesNearLigand: (radiusAngstroms: number) => {
        const plugin = pluginRef.current;
        if (!plugin) return null;
        return queryStructure(plugin, residuesNearLigandQuery(radiusAngstroms));
      },

      selectChain: (chainId: string) => {
        const plugin = pluginRef.current;
        if (!plugin) return null;
        return queryStructure(plugin, chainQuery({ chainId }));
      },

      selectResidues: (selector: ResidueSelector, chainId?: string) => {
        const plugin = pluginRef.current;
        if (!plugin) return null;
        const chain = chainId ? { chainId } : undefined;
        return queryStructure(plugin, residueQuery(selector, chain));
      },

      selectBackbone: () => {
        const plugin = pluginRef.current;
        if (!plugin) return null;
        return queryStructure(plugin, backboneQuery());
      },

      selectSidechain: () => {
        const plugin = pluginRef.current;
        if (!plugin) return null;
        return queryStructure(plugin, sidechainQuery());
      },

      selectAlphaCarbons: () => {
        const plugin = pluginRef.current;
        if (!plugin) return null;
        return queryStructure(plugin, alphaCarbonQuery());
      },

      // === Visual Methods ===
      highlight: async (expression: Expression, options?: HighlightOptions) => {
        const plugin = pluginRef.current;
        if (!plugin) return;
        await highlightSelection(plugin, expression, options);
      },

      clearHighlights: () => {
        const plugin = pluginRef.current;
        if (!plugin) return;
        clearHighlights(plugin);
      },

      select: async (expression: Expression, options?: { add?: boolean }) => {
        const plugin = pluginRef.current;
        if (!plugin) return;
        await selectAtoms(plugin, expression, options);
      },

      clearSelection: () => {
        const plugin = pluginRef.current;
        if (!plugin) return;
        clearSelection(plugin);
      },

      focusOn: async (expression: Expression) => {
        const plugin = pluginRef.current;
        if (!plugin) return;
        await focusOnSelection(plugin, expression);
      },

      showAtoms: (
        residues: LabelResidue[],
        atoms: LabelAtom[],
        options?: { focus?: boolean }
      ) => {
        const plugin = pluginRef.current;
        if (!plugin) return { residues: 0, atoms: 0 };
        return showAtoms(plugin, residues, atoms, options);
      },

      showResidues: (residues: LabelResidue[], options?: { focus?: boolean }) => {
        const plugin = pluginRef.current;
        if (!plugin) return 0;
        return showResidues(plugin, residues, options);
      },

      offerSelection: (label: string, residues: LabelResidue[], atoms: LabelAtom[]) => {
        pickerRef.current?.offerAgentSelection(label, residues, atoms);
      },

      paintLayers: async (layers: PaintLayer[]) => {
        const plugin = pluginRef.current;
        if (!plugin) return;
        await paintLayers(plugin, layers);
      },

      clearPaint: async () => {
        const plugin = pluginRef.current;
        if (!plugin) return;
        await clearPaint(plugin);
      },

      commitPicks: async (sessionId: string | undefined) => {
        const picker = pickerRef.current;
        if (!picker) return [];
        try {
          return await picker.commit(sessionId);
        } catch (err) {
          console.error("Failed to store picked residues:", err);
          return [];
        }
      },

      // === Measurement Methods ===
      measureDistance,
      measureAngle,
      measureDihedral,

      getCenterOfMass: (expression: Expression) => {
        const plugin = pluginRef.current;
        if (!plugin) return null;
        return getCenterOfMass(plugin, expression);
      },

      getBoundingBox: (expression: Expression) => {
        const plugin = pluginRef.current;
        if (!plugin) return null;
        return getBoundingBox(plugin, expression);
      },

      // === Query Builders ===
      queries: {
        ligand: ligandQuery,
        polymer: polymerQuery,
        backbone: backboneQuery,
        sidechain: sidechainQuery,
        alphaCarbons: alphaCarbonQuery,
        chain: (selector: ChainSelector) => chainQuery(selector),
        residue: (selector: ResidueSelector, chain?: ChainSelector) =>
          residueQuery(selector, chain),
        atom: (
          selector: AtomSelector,
          residue?: ResidueSelector,
          chain?: ChainSelector
        ) => atomQuery(selector, residue, chain),
        withinDistance: (
          target: Expression,
          radiusAngstroms: number,
          wholeResidues = true
        ) =>
          withinDistanceQuery(target, {
            radius: radiusAngstroms,
            wholeResidues,
          }),
        residuesNearLigand: residuesNearLigandQuery,
      },

      // === State Query (for LLM context) ===
      getState: (): ViewerStateInfo => {
        const plugin = pluginRef.current;
        if (!plugin) {
          console.log("[MolstarViewer] getState: plugin not initialized, returning empty");
          return {
            isEmpty: true,
            structures: [],
            hasSelection: false,
            selectionInfo: null,
          };
        }
        const state = getViewerState(
          plugin,
          structureLabelsRef.current,
          pickerRef.current?.describe()
        );
        console.log("[MolstarViewer] getState:", JSON.stringify(state, null, 2));
        return state;
      },

      // === Direct Plugin Access ===
      getPlugin: () => pluginRef.current,
    }));

    return (
      <div className={`molstar-viewer ${className || ""}`}>
        <div ref={containerRef} className="molstar-container" />

        {isEmpty && !isLoading && !error && (
          <div className="molstar-placeholder">
            <div className="placeholder-icon">🔬</div>
            <p>3D Molecule Viewer</p>
            <span>Molecules will appear here when discussed in chat</span>
          </div>
        )}

        {!isEmpty && pickerState && (
          <DesignRolesPanel state={pickerState} picker={pickerRef.current} />
        )}

        {isLoading && (
          <div className="molstar-loading">
            <span className="loading-spinner"></span>
            <span>Loading 3D structure...</span>
          </div>
        )}

        {error && (
          <div className="molstar-error">
            <span>{error}</span>
          </div>
        )}
      </div>
    );
  }
);

MolstarViewer.displayName = "MolstarViewer";

export default MolstarViewer;
