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

import { useCallback, useRef, useState, useMemo, useEffect } from "react";
import { Group, Panel, Separator, useDefaultLayout } from "react-resizable-panels";
import Chat, { ToolEvent, ChatRef, AvailableArtifact, StorageUpdateArtifact, ModelInfo } from "./components/Chat";
import MolstarViewer, { MolstarViewerRef } from "./components/MolstarViewer";
import StorageStrip, { Artifact } from "./components/StorageStrip";
import TrainingMode from "./components/TrainingMode";
import ToolManagerPanel from "./components/ToolManagerPanel";
import AddExtensionModal from "./components/AddExtensionModal";
import FeedbackModal from "./components/FeedbackModal";
import { createSession } from "./session";
import type { ViewerStateInfo } from "./lib/molstar/types";
import { serializeAtoms } from "./lib/molstar/selectionPayload";
import { parseHexColour } from "./lib/molstar/residues";
import {
  describeError,
  parseMolQL,
  postMolQLFailure,
} from "./lib/molstar/molqlErrors";

/**
 * Generate a UUID v4, with fallback for older browsers that don't support crypto.randomUUID()
 */
/**
 * Offer the atoms a MolQL query matched as something the user can take as a
 * pick ("Use as…"). Polymer atoms only: only those can play a role.
 */
function offerQueryResult(
  viewer: MolstarViewerRef,
  expression: Parameters<MolstarViewerRef["getAtoms"]>[0],
  name: string | undefined
): void {
  const atoms = viewer
    .getAtoms(expression)
    .filter((a) => a.residue.seqNumber > 0)
    .map((a) => ({ chain: a.residue.chainId, seq: a.residue.seqNumber, name: a.name }));
  viewer.offerSelection(name || "Assistant's query", [], atoms);
}

/**
 * Fetch JSON, retrying transient failures.
 *
 * A single attempt is not enough for startup data: the dev backend runs under
 * uvicorn --reload and is briefly unreachable after any edit, so a page load
 * that lands in that window would otherwise leave the app permanently missing
 * whatever the call was for. Also treats a non-2xx as an error rather than
 * handing an HTML error page to res.json().
 */
async function fetchJsonWithRetry<T>(
  url: string,
  { attempts = 4, baseDelayMs = 400 }: { attempts?: number; baseDelayMs?: number } = {},
): Promise<T> {
  let lastError: unknown;
  for (let attempt = 0; attempt < attempts; attempt++) {
    try {
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`${url} responded ${res.status}`);
      }
      return (await res.json()) as T;
    } catch (error) {
      lastError = error;
      if (attempt < attempts - 1) {
        await new Promise((resolve) => setTimeout(resolve, baseDelayMs * 2 ** attempt));
      }
    }
  }
  throw lastError;
}

const PROVIDER_LABELS: Record<string, string> = {
  openai_compat: "NVIDIA",
  anthropic: "Anthropic",
  openai: "OpenAI",
};

/** Models grouped by provider, groups in the order the backend lists them. */
function groupModelsByProvider(models: ModelInfo[]): [string, ModelInfo[]][] {
  const groups = new Map<string, ModelInfo[]>();
  for (const m of models) {
    const provider = m.provider ?? "openai_compat";
    groups.set(provider, [...(groups.get(provider) ?? []), m]);
  }
  return [...groups.entries()];
}

function modelProviderCount(models: ModelInfo[]): number {
  return new Set(models.map((m) => m.provider ?? "openai_compat")).size;
}

// Selection info for display in viewer panel
interface SelectionDisplayInfo {
  summary: string;
  atomCount: number;
  residueCount: number;
  action: string;
}

function App() {
  // Check if we're in training mode based on URL path
  const isTrainingMode = window.location.pathname.startsWith("/training");

  // If in training mode, render the training UI
  if (isTrainingMode) {
    return <TrainingMode />;
  }

  // Otherwise render normal Aevom UI
  return <MainApp />;
}

// Hamburger menu icon for tool panel toggle
const HamburgerIcon = () => (
  <svg
    width="20"
    height="20"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <line x1="3" y1="12" x2="21" y2="12"></line>
    <line x1="3" y1="6" x2="21" y2="6"></line>
    <line x1="3" y1="18" x2="21" y2="18"></line>
  </svg>
);

function MainApp() {
  const viewerRef = useRef<MolstarViewerRef>(null);
  const chatRef = useRef<ChatRef>(null);
  const eventChainRef = useRef<Promise<void>>(Promise.resolve());
  const [currentLabel, setCurrentLabel] = useState<string | null>(null);
  const [isLoadingMolecule, setIsLoadingMolecule] = useState(false);
  const [moleculeError, setMoleculeError] = useState<string | null>(null);
  const [queryInfo, setQueryInfo] = useState<string | null>(null);
  const [selectionInfo, setSelectionInfo] =
    useState<SelectionDisplayInfo | null>(null);
  const [storageAccessEnabled, setStorageAccessEnabled] = useState(true);
  // Defaults to false, unlike storageAccessEnabled: an optimistic true would
  // flash a Feedback button into deployments that have the feature switched off.
  const [feedbackEnabled, setFeedbackEnabled] = useState(false);
  const [httpRequestToolEnabled, setHttpRequestToolEnabled] = useState(false);
  const [feedbackModalOpen, setFeedbackModalOpen] = useState(false);

  // Model selection state
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [modelsFailed, setModelsFailed] = useState(false);
  const activeModelRef = useRef<string>("");

  // Resizable panel layout persistence
  const { defaultLayout, onLayoutChanged } = useDefaultLayout({
    id: "aevom-main-layout",
  });

  // Tool manager panel state
  const [toolPanelOpen, setToolPanelOpen] = useState(false);
  const [addServerModalOpen, setAddServerModalOpen] = useState(false);
  const [toolPanelRefreshKey, setToolPanelRefreshKey] = useState(0);

  // Session state: UUID from the backend, bound to this browser's owner cookie.
  // Created fresh on every page load so refresh = clean slate.
  const [sessionId, setSessionId] = useState<string>("");
  const [sessionCopyFeedback, setSessionCopyFeedback] = useState<string | null>(null);

  const copySessionId = useCallback(() => {
    if (!sessionId) return;
    navigator.clipboard.writeText(sessionId).then(
      () => {
        setSessionCopyFeedback("Copied");
        setTimeout(() => setSessionCopyFeedback(null), 2000);
      },
      () => setSessionCopyFeedback("Copy failed"),
    );
  }, [sessionId]);

  // Create session on mount. There is no local fallback: a session the backend
  // did not mint is not bound to this browser's owner cookie, so every
  // session-scoped endpoint would reject it.
  useEffect(() => {
    createSession()
      .then(setSessionId)
      .catch((err) => console.error("Failed to create session:", err));
  }, []);

  // Fetch app info and available models on mount
  const loadModels = useCallback(async () => {
    try {
      const data = await fetchJsonWithRetry<{
        models?: ModelInfo[];
        default?: string;
      }>("/api/models");
      setAvailableModels(data.models || []);
      setModelsFailed(false);
      const defaultModel = data.default || data.models?.[0]?.id || "";
      setSelectedModel((current) => current || defaultModel);
      activeModelRef.current = activeModelRef.current || defaultModel;
    } catch (err) {
      console.error("Failed to fetch models:", err);
      setModelsFailed(true);
    }
  }, []);

  useEffect(() => {
    fetchJsonWithRetry<{ features?: Record<string, boolean> }>("/api/info")
      .then((data) => {
        if (data.features?.storage_access_api === false) {
          setStorageAccessEnabled(false);
        }
        if (data.features?.feedback_email === true) {
          setFeedbackEnabled(true);
        }
        if (data.features?.http_request_tool === true) {
          setHttpRequestToolEnabled(true);
        }
      })
      .catch((err) => console.error("Failed to fetch app info:", err));

    loadModels();
  }, [loadModels]);

  // Recover the model list if it was lost while the tab was hidden or the
  // backend was restarting. Without this the only cure is a manual reload.
  useEffect(() => {
    if (availableModels.length > 0) return;
    const retry = () => {
      if (document.visibilityState === "visible") loadModels();
    };
    window.addEventListener("focus", retry);
    document.addEventListener("visibilitychange", retry);
    return () => {
      window.removeEventListener("focus", retry);
      document.removeEventListener("visibilitychange", retry);
    };
  }, [availableModels.length, loadModels]);

  // Storage artifacts state
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [artifactsLoading, setArtifactsLoading] = useState(true);
  const [storageRefreshTrigger, setStorageRefreshTrigger] = useState(0);

  // Fetch artifacts when refresh trigger changes
  useEffect(() => {
    if (!sessionId) return;
    const fetchArtifacts = async () => {
      try {
        const url = `/api/storage/artifacts?session_id=${encodeURIComponent(sessionId)}`;
        const response = await fetch(url);
        if (response.ok) {
          const data = await response.json();
          setArtifacts(data.artifacts || []);
        }
      } catch (error) {
        console.error("Failed to fetch artifacts:", error);
      } finally {
        setArtifactsLoading(false);
      }
    };
    fetchArtifacts();
  }, [sessionId, storageRefreshTrigger]);

  // Convert artifacts to format expected by Chat's autocomplete
  const availableArtifacts: AvailableArtifact[] = useMemo(
    () =>
      artifacts.map((a) => ({
        reference: a.reference,
        data_type: a.data_type,
        metadata: a.metadata,
      })),
    [artifacts]
  );

  // Handler for injecting a reference from the storage strip
  const handleInjectRef = useCallback((reference: string) => {
    chatRef.current?.insertBubble(reference);
  }, []);

  // Handler for when chat response completes - refresh storage strip
  const handleResponseComplete = useCallback(() => {
    setStorageRefreshTrigger((prev) => prev + 1);
  }, []);

  // Handler for real-time storage updates between tool calls
  const handleStorageUpdate = useCallback((newArtifacts: StorageUpdateArtifact[]) => {
    // Convert StorageUpdateArtifact to Artifact format and update state
    setArtifacts(newArtifacts.map(a => ({
      reference: a.reference,
      data_type: a.data_type,
      description: a.description ?? null,
      metadata: a.metadata ?? {},
      created_at: a.created_at ?? null,
    })));
    setArtifactsLoading(false);
  }, []);

  // Handler for when MCP server is added - refresh tool panel
  const handleServerAdded = useCallback(() => {
    setToolPanelRefreshKey((prev) => prev + 1);
  }, []);

  // Reset all state when the user sends a message with a different model
  const resetForModelSwitch = useCallback(async (newModelId: string): Promise<string> => {
    viewerRef.current?.clear();
    setArtifacts([]);
    setArtifactsLoading(false);
    setCurrentLabel(null);
    setMoleculeError(null);
    setQueryInfo(null);
    setSelectionInfo(null);
    activeModelRef.current = newModelId;
    sessionStorage.removeItem("aevom_session_id");
    const newSessionId = await createSession();
    setSessionId(newSessionId);
    return newSessionId;
  }, []);

  // Get current viewer state for sending with chat requests
  const getViewerState = useCallback((): ViewerStateInfo => {
    if (!viewerRef.current) {
      return {
        isEmpty: true,
        structures: [],
        hasSelection: false,
        selectionInfo: null,
      };
    }
    return viewerRef.current.getState();
  }, []);

  // Store residues the user picked by hand as a selection before each message,
  // so the agent sees them in this turn. The session may differ from React
  // state right after a model switch, so the one Chat is sending in wins.
  const commitPicks = useCallback(
    async (sendSessionId: string | undefined) =>
      viewerRef.current?.commitPicks(sendSessionId ?? sessionId) ?? [],
    [sessionId],
  );

  const processToolEvent = useCallback(async (event: ToolEvent) => {
    // Handle SDF 3D visualization tool events (small molecules)
    if (event.type === "visualize_3d" && event.sdf) {
      setCurrentLabel(event.label || "Molecule");
      setIsLoadingMolecule(true);
      setMoleculeError(null);
      setQueryInfo(null);

      try {
        await viewerRef.current?.loadSdf(event.sdf, event.label);
      } catch (err) {
        console.error("Failed to load molecule:", err);
        setMoleculeError(
          err instanceof Error ? err.message : "Failed to load molecule"
        );
      } finally {
        setIsLoadingMolecule(false);
      }
    }

    // Handle PDB 3D visualization tool events (protein structures)
    if (event.type === "visualize_pdb" && event.pdb) {
      setCurrentLabel(event.label || "Protein Structure");
      setIsLoadingMolecule(true);
      setMoleculeError(null);
      setQueryInfo(null);

      try {
        await viewerRef.current?.loadPdb(event.pdb, event.label, event.format, event.roles ?? null);
      } catch (err) {
        console.error("Failed to load protein structure:", err);
        setMoleculeError(
          err instanceof Error
            ? err.message
            : "Failed to load protein structure"
        );
      } finally {
        setIsLoadingMolecule(false);
      }
    }

    // Show a selection the agent made (or the user did) by residue number.
    // Residues with only some atoms chosen arrive as `atoms`, so the viewer
    // does not light up atoms nobody picked.
    if (event.type === "show_selection" && (event.residues || event.atoms)) {
      const viewer = viewerRef.current;
      if (!viewer) return;
      const residues = event.residues ?? [];
      const atoms = event.atoms ?? [];
      const colour = parseHexColour(event.colour);
      const found = viewer.showAtoms(residues, atoms, { focus: event.focus !== false });
      if (colour !== null && (found.residues > 0 || found.atoms > 0)) {
        await viewer.paintLayers([{ residues, atoms, colour }]);
      }
      if (found.residues > 0 || found.atoms > 0) {
        viewer.offerSelection(event.selection_ref || "Assistant's selection", residues, atoms);
      }
      if (found.residues < residues.length || found.atoms < atoms.length) {
        console.warn(
          `show_selection ${event.selection_ref}: ${found.residues}/${residues.length} residues and ` +
            `${found.atoms}/${atoms.length} atoms are in the loaded structure`
        );
      }
      setQueryInfo(null);
    }

    // Handle structure query tool events (MolQL queries)
    if (event.type === "structure_query" && event.query?.molql) {
      // Use session_id from the event (matches the graph context) to avoid
      // mismatch when React state has a stale/new session after hot-reload.
      const postSid = event.session_id || sessionId;
      const selectionUrl = postSid
        ? `/api/selection?session_id=${encodeURIComponent(postSid)}`
        : "/api/selection";
      const molql = event.query.molql;
      const queryId = event.query_id;
      const queryAction = event.query.action || "select";

      // Any failure must still POST: the backend waits on this POST and, with
      // none, tells the model the viewer is not answering. The message says why
      // the query failed so the model can fix its syntax instead of guessing.
      const reportFailure = (detail: string) => {
        setQueryInfo("Error: Failed to execute query");
        if (queryId) {
          void postMolQLFailure(selectionUrl, queryId, molql, queryAction, detail);
        }
      };

      try {
        const parsed = parseMolQL(molql);
        if (parsed.error !== undefined) {
          console.error("Failed to parse MolQL expression:", molql);
          setQueryInfo("Error: Invalid MolQL expression");
          if (queryId) {
            void postMolQLFailure(selectionUrl, queryId, molql, queryAction, parsed.error);
          }
          return;
        }
        const expression = parsed.expression;

        const action = queryAction;
        const viewer = viewerRef.current;
        if (!viewer) return;

        // Always execute: select + focus
        // This unified behavior ensures:
        // 1. Selection is created for downstream analysis (python_execute)
        // 2. Camera focuses on selection for user visibility
        // 3. Selection info is POSTed back to backend for LLM context
        await viewer.select(expression);
        await viewer.focusOn(expression);
        offerQueryResult(viewer, expression, event.query.name);

        // POST selection data to backend for agent access
        // This enables the agent to analyze the selection with python_execute
        if (event.query_id) {
          try {
            const queryResult = viewer.query(expression);
            const atoms = viewer.getAtoms(expression);
            
            // Limit atoms to prevent oversized payloads (max 10,000 atoms)
            const MAX_ATOMS = 10000;
            const atomsToSend = atoms.slice(0, MAX_ATOMS);
            
            const atomCount = queryResult?.atomCount || atoms.length;
            const residueCount = queryResult?.residueCount || 0;
            const summary = queryResult?.summary || `Selected ${atoms.length} atoms`;
            
            const selectionData = {
              query_id: event.query_id,
              molql: event.query.molql,
              action: action,
              atoms: serializeAtoms(atomsToSend),
              atom_count: atomCount,
              residue_count: residueCount,
              summary: summary,
              ...(event.query.name ? { name: event.query.name } : {}),
            };

            // Update UI to show selection is active
            setSelectionInfo({
              summary: summary,
              atomCount: atomCount,
              residueCount: residueCount,
              action: action,
            });

            // POST to backend asynchronously (don't await - fire and forget for speed)
            fetch(selectionUrl, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(selectionData),
            }).then(response => {
              if (response.ok) {
                console.log(`Selection data posted for query ${event.query_id}: ${atomsToSend.length} atoms`);
              } else {
                console.error(`Failed to post selection data: ${response.status}`);
              }
            }).catch(err => {
              console.error("Error posting selection data:", err);
            });
            
            if (atoms.length > MAX_ATOMS) {
              console.warn(`Selection truncated: ${atoms.length} atoms -> ${MAX_ATOMS} atoms`);
            }
          } catch (err) {
            console.error("Failed to extract/post selection data:", err);
            reportFailure(describeError(err));
          }
        }

        console.log(
          `MolQL query executed: ${event.query.molql.substring(
            0,
            50
          )}..., action: ${action}`
        );
      } catch (err) {
        // An expression that parses but does not compile (e.g. an unimplemented
        // symbol such as `in`) throws from viewer.select before any POST.
        console.error("Failed to execute MolQL query:", err);
        reportFailure(describeError(err));
      }
    }
  }, [sessionId]);

  const handleToolEvent = useCallback(
    (event: ToolEvent) => {
      eventChainRef.current = eventChainRef.current
        .then(() => processToolEvent(event))
        .catch((err) => console.error("Tool event processing error:", err));
    },
    [processToolEvent],
  );

  return (
    <div className="app">
      <header className="app-header">
        <div className="logo">
          <span className="logo-icon">⚗️</span>
          <h1>aevom</h1>
        </div>
        <div className="header-right">
          <div className="header-info">
            <p className="tagline">AI Molecular Scientist</p>
            <p className="version">{import.meta.env.VERSION_NUMBER}</p>
            {availableModels.length > 0 ? (
              <select
                className="model-selector"
                value={selectedModel || ""}
                onChange={(e) => setSelectedModel(e.target.value)}
                title="Select AI model"
              >
                {groupModelsByProvider(availableModels).map(([provider, models]) => {
                  const options = models.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.pretty_name}
                    </option>
                  ));
                  // A single provider needs no group heading.
                  return modelProviderCount(availableModels) > 1 ? (
                    <optgroup key={provider} label={PROVIDER_LABELS[provider] ?? provider}>
                      {options}
                    </optgroup>
                  ) : (
                    options
                  );
                })}
              </select>
            ) : (
              // Keep the control present when the list is unavailable. Hiding it
              // made a transient fetch failure look like the feature had been
              // removed, with a manual reload the only way to find out otherwise.
              <select
                className="model-selector"
                // Stays enabled on failure so the click can retry; a disabled
                // select swallows mouse events.
                disabled={!modelsFailed}
                value=""
                title={
                  modelsFailed
                    ? "Could not load models. Click to retry."
                    : "Loading models..."
                }
                onMouseDown={(e) => {
                  if (modelsFailed) {
                    e.preventDefault();
                    loadModels();
                  }
                }}
              >
                <option value="">
                  {modelsFailed ? "Models unavailable — retry" : "Loading models..."}
                </option>
              </select>
            )}
          </div>
          <button
            className="tool-panel-toggle"
            onClick={() => setToolPanelOpen(!toolPanelOpen)}
            title={toolPanelOpen ? "Close tool manager" : "Open tool manager"}
          >
            <HamburgerIcon />
          </button>
        </div>
      </header>
      <main className="app-main split-layout">
        <Group
          orientation="horizontal"
          defaultLayout={defaultLayout}
          onLayoutChanged={onLayoutChanged}
        >
          <Panel
            id="chat"
            className="chat-panel"
            defaultSize="30%"
            minSize="360px"
            maxSize="50%"
          >
            <StorageStrip
              onInjectRef={handleInjectRef}
              artifacts={artifacts}
              isLoading={artifactsLoading}
            />
            <Chat
              ref={chatRef}
              onToolEvent={handleToolEvent}
              getViewerState={getViewerState}
              beforeSend={commitPicks}
              sessionId={sessionId}
              onResponseComplete={handleResponseComplete}
              onStorageUpdate={handleStorageUpdate}
              availableArtifacts={availableArtifacts}
              selectedModel={selectedModel}
              activeModel={activeModelRef.current}
              onModelSwitch={resetForModelSwitch}
            />
          </Panel>

          <Separator className="panel-separator" />

          <Panel
            id="viewer"
            className="viewer-panel"
            minSize="400px"
          >
            <MolstarViewer ref={viewerRef} sessionId={sessionId || undefined} />
            {(currentLabel || queryInfo || selectionInfo) && (
              <div className="viewer-info">
                {currentLabel && (
                  <code className="smiles-display">{currentLabel}</code>
                )}
                {isLoadingMolecule && (
                  <span className="viewer-status">Loading...</span>
                )}
                {moleculeError && (
                  <span className="viewer-error">{moleculeError}</span>
                )}
                {queryInfo && <span className="query-info">{queryInfo}</span>}
                {selectionInfo && (
                  <div className="selection-info">
                    <span className="selection-badge">
                      ✓ Selection Active
                    </span>
                    <span className="selection-details">
                      {selectionInfo.atomCount} atoms, {selectionInfo.residueCount} residues
                    </span>
                  </div>
                )}
              </div>
            )}
          </Panel>
        </Group>

        {/* Tool Manager Side Panel */}
        <ToolManagerPanel
          key={toolPanelRefreshKey}
          isOpen={toolPanelOpen}
          onClose={() => setToolPanelOpen(false)}
          sessionId={sessionId}
          onAddServer={() => setAddServerModalOpen(true)}
          storageAccessEnabled={storageAccessEnabled}
          onCopySessionId={copySessionId}
          sessionCopyFeedback={sessionCopyFeedback}
          feedbackEnabled={feedbackEnabled}
          onOpenFeedback={() => setFeedbackModalOpen(true)}
        />
      </main>

      {/* Add MCP Server / Skill Modal */}
      <AddExtensionModal
        isOpen={addServerModalOpen}
        onClose={() => setAddServerModalOpen(false)}
        sessionId={sessionId}
        onAdded={handleServerAdded}
        httpRequestToolEnabled={httpRequestToolEnabled}
      />

      {/* Free-text Feedback Modal */}
      <FeedbackModal
        isOpen={feedbackModalOpen}
        onClose={() => setFeedbackModalOpen(false)}
      />
    </div>
  );
}

export default App;
