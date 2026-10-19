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

import React, { useState, useEffect, useCallback, useRef, useMemo } from "react";
import Chat, { ToolEvent, ChatRef, AvailableArtifact, StorageUpdateArtifact } from "./Chat";
import MolstarViewer, { MolstarViewerRef } from "./MolstarViewer";
import StorageStrip, { Artifact } from "./StorageStrip";
import type { ViewerStateInfo } from "../lib/molstar/types";
import { serializeAtoms } from "../lib/molstar/selectionPayload";
import { createSession } from "../session";
import {
  describeError,
  parseMolQL,
  postMolQLFailure,
} from "../lib/molstar/molqlErrors";

// ============================================================================
// Types
// ============================================================================

type PromptMode = "production" | "oracle" | "weak";

interface TrainingQuery {
  id: string;
  query_text: string;
  expected_tool: string | null;
  category: string | null;
  created_at: string;
  is_active: boolean;
  status: "none" | "chosen_only" | "rejected_only" | "complete" | "multiple";
  chosen_count: number;
  rejected_count: number;
  total_responses: number;
}

interface StatusSummary {
  total_queries: number;
  complete_pairs: number;
  in_progress: number;
  not_started: number;
}

interface QueriesResponse {
  queries: TrainingQuery[];
  summary: StatusSummary;
}

interface VersionsResponse {
  current_version: string;
  versions: string[];
}

// ============================================================================
// Helper Functions
// ============================================================================

// Status icon and color mappings (NVIDIA theme)
const STATUS_CONFIG: Record<
  TrainingQuery["status"],
  { icon: string; color: string; label: string }
> = {
  none: { icon: "○", color: "#6b6b6b", label: "Not started" },
  chosen_only: { icon: "◐", color: "#76B900", label: "Has chosen, needs rejected" },
  rejected_only: { icon: "◑", color: "#f85149", label: "Has rejected, needs chosen" },
  complete: { icon: "✓", color: "#76B900", label: "Pair complete!" },
  multiple: { icon: "✓✓", color: "#8ed100", label: "Multiple pairs" },
};

// Mode color mappings (NVIDIA theme - neutral grey backgrounds)
const MODE_CONFIG: Record<PromptMode, { color: string; bgColor: string; label: string }> = {
  production: { color: "#76B900", bgColor: "rgba(255, 255, 255, 0.05)", label: "Production" },
  oracle: { color: "#00b4d8", bgColor: "rgba(255, 255, 255, 0.05)", label: "Oracle" },
  weak: { color: "#f0883e", bgColor: "rgba(255, 255, 255, 0.05)", label: "Weak" },
};

// ============================================================================
// Components
// ============================================================================

interface QueryItemProps {
  query: TrainingQuery;
  isSelected: boolean;
  onClick: () => void;
  onDelete: (queryId: string) => void;
}

function QueryItem({ query, isSelected, onClick, onDelete }: QueryItemProps) {
  const statusConfig = STATUS_CONFIG[query.status];

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent triggering onClick
    onDelete(query.id);
  };
  
  return (
    <div
      className={`training-query-item ${isSelected ? "selected" : ""}`}
      onClick={onClick}
      title={`${statusConfig.label}\n${query.chosen_count} chosen, ${query.rejected_count} rejected`}
    >
      <span
        className="query-status-icon"
        style={{ color: statusConfig.color }}
      >
        {statusConfig.icon}
      </span>
      <span className="query-text">
        {query.query_text.length > 50
          ? query.query_text.slice(0, 50) + "..."
          : query.query_text}
      </span>
      {query.category && (
        <span className="query-category">{query.category}</span>
      )}
      <button
        className="query-delete-btn"
        onClick={handleDelete}
        title="Delete query"
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="3 6 5 6 21 6"></polyline>
          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
        </svg>
      </button>
    </div>
  );
}

interface AddQueryFormProps {
  onAdd: (text: string, expectedTool: string, category: string) => Promise<void>;
  isAdding: boolean;
}

function AddQueryForm({ onAdd, isAdding }: AddQueryFormProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [text, setText] = useState("");
  const [expectedTool, setExpectedTool] = useState("");
  const [category, setCategory] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim()) return;
    await onAdd(text.trim(), expectedTool.trim(), category.trim());
    setText("");
    setExpectedTool("");
    setCategory("");
    setIsExpanded(false);
  };

  if (!isExpanded) {
    return (
      <button
        className="training-add-query-btn"
        onClick={() => setIsExpanded(true)}
      >
        + Add Query
      </button>
    );
  }

  return (
    <form className="training-add-query-form" onSubmit={handleSubmit}>
      <textarea
        placeholder="Enter test query..."
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={2}
        autoFocus
      />
      <input
        type="text"
        placeholder="Expected tool (for oracle mode)"
        value={expectedTool}
        onChange={(e) => setExpectedTool(e.target.value)}
      />
      <input
        type="text"
        placeholder="Category (optional)"
        value={category}
        onChange={(e) => setCategory(e.target.value)}
      />
      <div className="form-buttons">
        <button type="button" onClick={() => setIsExpanded(false)}>
          Cancel
        </button>
        <button type="submit" disabled={!text.trim() || isAdding}>
          {isAdding ? "Adding..." : "Add"}
        </button>
      </div>
    </form>
  );
}

// ============================================================================
// Oracle Prompt Editor
// ============================================================================

interface OraclePromptEditorProps {
  queryId: string;
  onPromptChange: (prompt: string | null) => void;
}

function OraclePromptEditor({ queryId, onPromptChange }: OraclePromptEditorProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [originalPrompt, setOriginalPrompt] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Fetch oracle prompt when query changes or editor expands
  useEffect(() => {
    if (!isExpanded) return;

    const fetchPrompt = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch(`/api/training/queries/${queryId}/oracle-prompt`);
        if (response.ok) {
          const data = await response.json();
          setPrompt(data.oracle_prompt || "");
          setOriginalPrompt(data.oracle_prompt || "");
          setLastUpdated(data.oracle_prompt_updated_at);
        } else {
          setError("Failed to load oracle prompt");
        }
      } catch (err) {
        setError("Failed to load oracle prompt");
        console.error("Failed to fetch oracle prompt:", err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchPrompt();
  }, [queryId, isExpanded]);

  // Notify parent when prompt changes (for use in chat requests)
  useEffect(() => {
    onPromptChange(prompt || null);
  }, [prompt, onPromptChange]);

  const handleSave = async () => {
    setIsSaving(true);
    setError(null);
    try {
      const response = await fetch(`/api/training/queries/${queryId}/oracle-prompt`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ oracle_prompt: prompt }),
      });
      if (response.ok) {
        const data = await response.json();
        setOriginalPrompt(prompt);
        setLastUpdated(data.oracle_prompt_updated_at);
      } else {
        setError("Failed to save oracle prompt");
      }
    } catch (err) {
      setError("Failed to save oracle prompt");
      console.error("Failed to save oracle prompt:", err);
    } finally {
      setIsSaving(false);
    }
  };

  const hasChanges = prompt !== originalPrompt;

  return (
    <div className="oracle-prompt-editor">
      <button
        className="oracle-prompt-toggle"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <span className="toggle-icon">{isExpanded ? "▼" : "▶"}</span>
        <span className="toggle-label">Oracle Prompt</span>
        {originalPrompt && <span className="has-prompt-badge">Custom</span>}
      </button>

      {isExpanded && (
        <div className="oracle-prompt-content">
          {isLoading ? (
            <div className="oracle-prompt-loading">Loading...</div>
          ) : (
            <>
              <textarea
                className="oracle-prompt-textarea"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Enter a custom oracle prompt for this query. Leave empty to use the default template."
                rows={10}
              />
              {error && <div className="oracle-prompt-error">{error}</div>}
              <div className="oracle-prompt-footer">
                {lastUpdated && (
                  <span className="oracle-prompt-updated">
                    Updated: {new Date(lastUpdated).toLocaleString()}
                  </span>
                )}
                <button
                  className="oracle-prompt-save-btn"
                  onClick={handleSave}
                  disabled={!hasChanges || isSaving}
                >
                  {isSaving ? "Saving..." : "Save"}
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export default function TrainingMode() {
  // URL params for mode
  const searchParams = new URLSearchParams(window.location.search);
  const initialMode = (searchParams.get("mode") as PromptMode) || "production";

  // State
  const [mode, setMode] = useState<PromptMode>(initialMode);
  const [queries, setQueries] = useState<TrainingQuery[]>([]);
  const [summary, setSummary] = useState<StatusSummary | null>(null);
  const [selectedQueryId, setSelectedQueryId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isAddingQuery, setIsAddingQuery] = useState(false);
  const [filterStatus, setFilterStatus] = useState<"all" | "incomplete">("all");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  
  // Version filtering state
  const [availableVersions, setAvailableVersions] = useState<string[]>([]);
  const [currentVersion, setCurrentVersion] = useState<string>("0.0.0-dev");
  const [minVersionFilter, setMinVersionFilter] = useState<string | null>(null);

  // Refs
  const viewerRef = useRef<MolstarViewerRef>(null);
  const chatRef = useRef<ChatRef>(null);

  // Generate session ID scoped to this training session
  const [sessionId, setSessionId] = useState("");
  useEffect(() => {
    createSession()
      .then(setSessionId)
      .catch((err) => console.error("Failed to create training session:", err));
  }, []);

  // Viewer state
  const [currentLabel, setCurrentLabel] = useState<string | null>(null);
  const [isLoadingMolecule, setIsLoadingMolecule] = useState(false);

  // Storage artifacts state
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [artifactsLoading, setArtifactsLoading] = useState(true);
  const [storageRefreshTrigger, setStorageRefreshTrigger] = useState(0);

  // Current selected query
  const selectedQuery = queries.find((q) => q.id === selectedQueryId);

  // Oracle prompt state (for passing to chat requests)
  const [currentOraclePrompt, setCurrentOraclePrompt] = useState<string | null>(null);

  // Auto-fetch oracle prompt when query is selected in oracle mode
  useEffect(() => {
    if (mode !== "oracle" || !selectedQueryId) {
      setCurrentOraclePrompt(null);
      return;
    }

    const fetchOraclePrompt = async () => {
      try {
        const response = await fetch(`/api/training/queries/${selectedQueryId}/oracle-prompt`);
        if (response.ok) {
          const data = await response.json();
          setCurrentOraclePrompt(data.oracle_prompt || null);
        }
      } catch (err) {
        console.error("Failed to fetch oracle prompt:", err);
      }
    };

    fetchOraclePrompt();
  }, [selectedQueryId, mode]);

  // Filtered queries based on status filter
  const filteredQueries = useMemo(() => {
    if (filterStatus === "all") return queries;
    return queries.filter((q) => q.status !== "complete" && q.status !== "multiple");
  }, [queries, filterStatus]);

  // Update URL when mode changes
  useEffect(() => {
    const url = new URL(window.location.href);
    url.searchParams.set("mode", mode);
    window.history.replaceState({}, "", url.toString());
  }, [mode]);

  // Track initialization state to prevent race conditions
  const [isInitialized, setIsInitialized] = useState(false);
  
  // Fetch queries with status
  const fetchQueries = useCallback(async (version: string | null, showLoading = false) => {
    if (showLoading) {
      setIsLoading(true);
    }
    try {
      const url = version 
        ? `/api/training/queries?min_version=${encodeURIComponent(version)}`
        : "/api/training/queries";
      const response = await fetch(url);
      if (response.ok) {
        const data: QueriesResponse = await response.json();
        setQueries(data.queries);
        setSummary(data.summary);
      }
    } catch (error) {
      console.error("Failed to fetch training queries:", error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Initialize: fetch versions first, then fetch queries with correct filter
  useEffect(() => {
    let cancelled = false;
    
    const initialize = async () => {
      let versionToUse: string | null = null;
      
      // Step 1: Fetch versions
      try {
        const response = await fetch("/api/training/versions");
        if (cancelled) return;
        
        if (response.ok) {
          const data: VersionsResponse = await response.json();
          setAvailableVersions(data.versions);
          setCurrentVersion(data.current_version);
          versionToUse = data.current_version;
          setMinVersionFilter(data.current_version);
        }
      } catch (error) {
        if (cancelled) return;
        console.error("Failed to fetch versions:", error);
      }
      
      // Step 2: Fetch queries with the version we just determined
      try {
        const url = versionToUse 
          ? `/api/training/queries?min_version=${encodeURIComponent(versionToUse)}`
          : "/api/training/queries";
        const response = await fetch(url);
        if (cancelled) return;
        
        if (response.ok) {
          const data: QueriesResponse = await response.json();
          setQueries(data.queries);
          setSummary(data.summary);
        }
      } catch (error) {
        if (cancelled) return;
        console.error("Failed to fetch training queries:", error);
      } finally {
        if (!cancelled) {
          setIsLoading(false);
          setIsInitialized(true);
        }
      }
    };
    
    initialize();
    
    // Cleanup function to handle React StrictMode double-mount
    return () => {
      cancelled = true;
    };
  }, []);

  // Track the previous filter value to detect actual changes
  const prevMinVersionFilterRef = useRef<string | null | undefined>(undefined);
  
  // Fetch queries when version filter changes (only after initialization)
  useEffect(() => {
    // Don't run until initialization is complete
    if (!isInitialized) return;
    
    // Skip if filter hasn't actually changed (prevents re-fetch on initial isInitialized flip)
    if (prevMinVersionFilterRef.current === minVersionFilter) return;
    
    // Track the current value
    const isFirstRun = prevMinVersionFilterRef.current === undefined;
    prevMinVersionFilterRef.current = minVersionFilter;
    
    // Skip the first run after initialization (that data was already fetched)
    if (isFirstRun) return;
    
    // Fetch with the new filter
    fetchQueries(minVersionFilter, true);
  }, [minVersionFilter, isInitialized, fetchQueries]);

  // Fetch storage artifacts
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

  // Handler for when a chat response completes (refresh storage)
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

  // Use refs to avoid recreating SSE connection on every filter change
  // while still having access to current values
  const fetchQueriesRef = useRef(fetchQueries);
  const minVersionFilterRef = useRef(minVersionFilter);
  useEffect(() => {
    fetchQueriesRef.current = fetchQueries;
    minVersionFilterRef.current = minVersionFilter;
  }, [fetchQueries, minVersionFilter]);

  // Subscribe to SSE for real-time query updates across tabs
  useEffect(() => {
    const eventSource = new EventSource("/api/training/events");

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === "query_created" && data.query) {
          // Add new query to beginning of list (sorted by created_at desc)
          setQueries((prev) => {
            // Avoid duplicates (in case this tab created the query)
            if (prev.some((q) => q.id === data.query.id)) {
              return prev;
            }
            return [data.query, ...prev];
          });
          // Update summary
          setSummary((prev) =>
            prev
              ? {
                  ...prev,
                  total_queries: prev.total_queries + 1,
                  not_started: prev.not_started + 1,
                }
              : prev
          );
        } else if (data.type === "query_deleted" && data.id) {
          // Remove query from list
          setQueries((prev) => prev.filter((q) => q.id !== data.id));
          // Update summary (assume it was not_started or in_progress)
          setSummary((prev) =>
            prev
              ? {
                  ...prev,
                  total_queries: Math.max(0, prev.total_queries - 1),
                }
              : prev
          );
          // Clear selection if deleted query was selected
          if (selectedQueryId === data.id) {
            setSelectedQueryId(null);
          }
        } else if (data.type === "feedback_submitted" && data.query_id) {
          // Refresh queries to get updated status from Phoenix
          // Use a slightly longer delay to ensure Phoenix has processed the annotation
          // Use refs to get current values without recreating the SSE connection
          setTimeout(() => fetchQueriesRef.current(minVersionFilterRef.current), 1000);
        }
      } catch (err) {
        console.error("Failed to parse SSE event:", err);
      }
    };

    eventSource.onerror = (err) => {
      console.error("SSE connection error:", err);
      // EventSource will automatically reconnect
    };

    return () => {
      eventSource.close();
    };
  }, [selectedQueryId]);

  // Refresh status after feedback is submitted
  const refreshAfterFeedback = useCallback(() => {
    // Delay to let Phoenix process the annotation
    // The SSE event will also trigger a refresh for other tabs
    // Use refs to get current values
    setTimeout(() => fetchQueriesRef.current(minVersionFilterRef.current), 1000);
  }, []);

  // Add a new query
  const handleAddQuery = async (text: string, expectedTool: string, category: string) => {
    setIsAddingQuery(true);
    try {
      const response = await fetch("/api/training/queries", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query_text: text,
          expected_tool: expectedTool || null,
          category: category || null,
        }),
      });
      if (response.ok) {
        await fetchQueries(minVersionFilter);
      }
    } catch (error) {
      console.error("Failed to add query:", error);
    } finally {
      setIsAddingQuery(false);
    }
  };

  // Send query to chat
  const handleQueryClick = (query: TrainingQuery) => {
    setSelectedQueryId(query.id);
    // Set the query text in the chat input
    chatRef.current?.setInputText(query.query_text);
  };

  // Delete a query
  const handleDeleteQuery = async (queryId: string) => {
    try {
      const response = await fetch(`/api/training/queries/${queryId}`, {
        method: "DELETE",
      });
      if (response.ok) {
        // Remove from local state (SSE will also broadcast this)
        setQueries((prev) => prev.filter((q) => q.id !== queryId));
        // Clear selection if deleted query was selected
        if (selectedQueryId === queryId) {
          setSelectedQueryId(null);
        }
      }
    } catch (error) {
      console.error("Failed to delete query:", error);
    }
  };

  // Viewer state getter for chat
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

  // Tool event handler
  const handleToolEvent = useCallback(async (event: ToolEvent) => {
    if (event.type === "visualize_3d" && event.sdf) {
      setCurrentLabel(event.label || "Molecule");
      setIsLoadingMolecule(true);
      try {
        await viewerRef.current?.loadSdf(event.sdf, event.label);
      } catch (err) {
        console.error("Failed to load molecule:", err);
      } finally {
        setIsLoadingMolecule(false);
      }
    }

    if (event.type === "visualize_pdb" && event.pdb) {
      setCurrentLabel(event.label || "Protein Structure");
      setIsLoadingMolecule(true);
      try {
        await viewerRef.current?.loadPdb(event.pdb, event.label, event.format, event.roles ?? null);
      } catch (err) {
        console.error("Failed to load protein structure:", err);
      } finally {
        setIsLoadingMolecule(false);
      }
    }

    if (event.type === "show_selection" && (event.residues || event.atoms)) {
      viewerRef.current?.showAtoms(event.residues ?? [], event.atoms ?? [], {
        focus: event.focus !== false,
      });
    }

    if (event.type === "structure_query" && event.query?.molql) {
      const selectionUrl = `/api/selection?session_id=${encodeURIComponent(sessionId)}`;
      const molql = event.query.molql;
      const queryId = event.query_id;
      const queryAction = event.query.action || "select";

      // A failed query must still POST, or the backend waits out its timeout
      // and the model is told the viewer is not answering. See molqlErrors.ts.
      const reportFailure = (detail: string) => {
        if (queryId) {
          void postMolQLFailure(selectionUrl, queryId, molql, queryAction, detail);
        }
      };

      try {
        const parsed = parseMolQL(molql);
        if (parsed.error !== undefined) {
          reportFailure(parsed.error);
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

        // Post selection data to backend
        if (event.query_id) {
          const queryResult = viewer.query(expression);
          const atoms = viewer.getAtoms(expression);
          const atomsToSend = atoms.slice(0, 10000);

          fetch(`/api/selection?session_id=${encodeURIComponent(sessionId)}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              query_id: event.query_id,
              molql: event.query.molql,
              action: action,
              atoms: serializeAtoms(atomsToSend),
              atom_count: queryResult?.atomCount || atoms.length,
              residue_count: queryResult?.residueCount || 0,
              summary: queryResult?.summary || `Selected ${atoms.length} atoms`,
              ...(event.query.name ? { name: event.query.name } : {}),
            }),
          }).catch(console.error);
        }
      } catch (err) {
        console.error("Failed to execute MolQL query:", err);
        reportFailure(describeError(err));
      }
    }
  }, [sessionId]);

  // Mode config for styling
  const modeConfig = MODE_CONFIG[mode];

  return (
    <div className="training-mode">
      {/* Header banner showing current mode */}
      <header
        className="training-header"
        style={{ backgroundColor: modeConfig.bgColor, borderColor: modeConfig.color }}
      >
        <div className="training-header-left">
          <span className="training-badge" style={{ color: modeConfig.color }}>
            TRAINING MODE
          </span>
          <span className="training-mode-label" style={{ color: modeConfig.color }}>
            {modeConfig.label.toUpperCase()}
          </span>
        </div>
        <div className="training-mode-selector">
          {(["production", "oracle", "weak"] as PromptMode[]).map((m) => (
            <button
              key={m}
              className={`mode-btn ${mode === m ? "active" : ""}`}
              onClick={() => setMode(m)}
              style={{
                borderColor: mode === m ? MODE_CONFIG[m].color : "transparent",
                backgroundColor: mode === m ? MODE_CONFIG[m].bgColor : "transparent",
                color: mode === m ? MODE_CONFIG[m].color : "var(--text-secondary)",
              }}
            >
              {MODE_CONFIG[m].label}
            </button>
          ))}
        </div>
        <div className="training-version-filter">
          <label htmlFor="version-filter">Min Version:</label>
          <select
            id="version-filter"
            value={minVersionFilter ?? ""}
            onChange={(e) => setMinVersionFilter(e.target.value || null)}
            className="version-select"
            title="Only count feedback from this version or newer"
          >
            <option value="">All versions</option>
            {availableVersions.map((v) => (
              <option key={v} value={v}>
                {v}{v === currentVersion ? " (current)" : ""}
              </option>
            ))}
          </select>
        </div>
        {summary && (
          <div className="training-progress">
            <span className="progress-text">
              {summary.complete_pairs}/{summary.total_queries} pairs
            </span>
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{
                  width: `${(summary.complete_pairs / Math.max(summary.total_queries, 1)) * 100}%`,
                }}
              />
            </div>
          </div>
        )}
      </header>

      <div className="training-main">
        {/* Sidebar with query list */}
        <aside className={`training-sidebar ${sidebarCollapsed ? "collapsed" : ""}`}>
          <button
            className="sidebar-toggle"
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {sidebarCollapsed ? "»" : "«"}
          </button>

          {!sidebarCollapsed && (
            <>
              <div className="sidebar-header">
                <h3>Test Queries</h3>
                <select
                  value={filterStatus}
                  onChange={(e) => setFilterStatus(e.target.value as "all" | "incomplete")}
                  className="filter-select"
                >
                  <option value="all">All</option>
                  <option value="incomplete">Incomplete only</option>
                </select>
              </div>

              <div className="query-list">
                {isLoading ? (
                  <div className="loading-queries">Loading queries...</div>
                ) : filteredQueries.length === 0 ? (
                  <div className="no-queries">
                    {queries.length === 0
                      ? "No queries yet. Add one below!"
                      : "All queries complete!"}
                  </div>
                ) : (
                  filteredQueries.map((query) => (
                    <QueryItem
                      key={query.id}
                      query={query}
                      isSelected={query.id === selectedQueryId}
                      onClick={() => handleQueryClick(query)}
                      onDelete={handleDeleteQuery}
                    />
                  ))
                )}
              </div>

              {/* Oracle Prompt Editor - show when a query is selected */}
              {selectedQuery && (
                <div className="oracle-prompt-section">
                  <OraclePromptEditor
                    queryId={selectedQuery.id}
                    onPromptChange={setCurrentOraclePrompt}
                  />
                </div>
              )}

              <div className="sidebar-footer">
                <AddQueryForm onAdd={handleAddQuery} isAdding={isAddingQuery} />
              </div>
            </>
          )}
        </aside>

        {/* Main content area */}
        <div className="training-content">
          <div className="training-chat-panel">
            <StorageStrip
              onInjectRef={handleInjectRef}
              artifacts={artifacts}
              isLoading={artifactsLoading}
            />
            {selectedQuery ? (
              <div className="selected-query-banner">
                <span className="query-label">Query:</span>
                <span className="query-preview">{selectedQuery.query_text}</span>
                {selectedQuery.expected_tool && mode === "oracle" && (
                  <span className="expected-tool">
                    Expected: {selectedQuery.expected_tool}
                  </span>
                )}
              </div>
            ) : (
              <div className="selected-query-banner warning">
                <span className="query-label">⚠️ No query selected</span>
                <span className="query-preview">
                  Select a query from the sidebar to track feedback for DPO pairs
                </span>
              </div>
            )}
            <TrainingChat
              ref={chatRef}
              mode={mode}
              selectedQuery={selectedQuery}
              oraclePrompt={currentOraclePrompt}
              sessionId={sessionId}
              onToolEvent={handleToolEvent}
              getViewerState={getViewerState}
              onFeedbackSubmitted={refreshAfterFeedback}
              onResponseComplete={handleResponseComplete}
              onStorageUpdate={handleStorageUpdate}
              availableArtifacts={availableArtifacts}
            />
          </div>

          <div className="training-viewer-panel">
            <MolstarViewer ref={viewerRef} />
            {currentLabel && (
              <div className="viewer-info">
                <code className="smiles-display">{currentLabel}</code>
                {isLoadingMolecule && <span className="viewer-status">Loading...</span>}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Training-specific Chat Wrapper
// ============================================================================

interface TrainingChatProps {
  mode: PromptMode;
  selectedQuery: TrainingQuery | undefined;
  oraclePrompt: string | null;
  sessionId: string;
  onToolEvent: (event: ToolEvent) => void;
  getViewerState: () => ViewerStateInfo;
  onFeedbackSubmitted: () => void;
  onResponseComplete: () => void;
  onStorageUpdate?: (artifacts: StorageUpdateArtifact[]) => void;
  availableArtifacts: AvailableArtifact[];
}

/**
 * A wrapper around the Chat component that:
 * 1. Sends the prompt_mode and training_query_id with each request
 * 2. Triggers status refresh after feedback
 */
const TrainingChat = React.forwardRef<ChatRef, TrainingChatProps>(
  function TrainingChat(
    {
      mode,
      selectedQuery,
      oraclePrompt,
      sessionId,
      onToolEvent,
      getViewerState,
      onFeedbackSubmitted,
      onResponseComplete,
      onStorageUpdate,
      availableArtifacts,
    },
    ref
  ) {
    return (
      <Chat
        ref={ref}
        onToolEvent={onToolEvent}
        getViewerState={getViewerState}
        sessionId={sessionId}
        onResponseComplete={onResponseComplete}
        onStorageUpdate={onStorageUpdate}
        availableArtifacts={availableArtifacts}
        customFetchOptions={{
          prompt_mode: mode,
          training_query_id: selectedQuery?.id,
          expected_tool: mode === "oracle" ? (selectedQuery?.expected_tool ?? undefined) : undefined,
          oracle_prompt: mode === "oracle" ? (oraclePrompt ?? undefined) : undefined,
        }}
        onFeedbackSubmitted={onFeedbackSubmitted}
      />
    );
  }
);
