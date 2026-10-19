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

import { useState, useEffect, useCallback } from "react";

/** MCP Server from the backend */
export interface MCPServer {
  id: string;
  name: string;
  url: string;
  status: string;
  tools_count: number;
  /** Whether these tools are reachable by the agent.
   *
   * A newly added server is connected and enumerated but not authorised, so
   * `tools` lists what it offers rather than what the model can call. Absent
   * on responses from a backend predating the review gate, which is why the
   * UI treats undefined as confirmed rather than blocking. */
  confirmed?: boolean;
  tools: {
    name: string;
    description: string;
    full_name?: string;
  }[];
}

/** A self-hosted NIM registered for this session.
 *
 * Unlike an MCP server, a NIM announces a model rather than a toolset. What
 * it contributes is decided by whether Aevom has an adapter for that model,
 * so `adapter` being null is a normal state and not an error. */
export interface NIMServer {
  id: string;
  name: string;
  url: string;
  model_id: string | null;
  model_version: string | null;
  ready: boolean;
  confirmed?: boolean;
  adapter: string | null;
  adapter_label: string | null;
  adapter_summary: string;
  tools: string[];
  tools_count: number;
  /** Built-in tools this NIM serves in place of the hosted NVIDIA endpoint. */
  repointed_tools: string[];
}

/** Tool info from the backend */
export interface ToolInfo {
  name: string;
  description: string;
  source: string; // "builtin" or server name
}

/** Skill from the backend */
export interface Skill {
  id: string;
  name: string;
  description: string;
  allowed_tools: string[];
  resources: string[];
  source: string; // "builtin" or "session"
  removable: boolean;
}

/** Stored API key (secret) from the backend. Value is never returned. */
export interface Secret {
  id: string;
  name: string;
  created_at: number;
}

interface ToolManagerPanelProps {
  isOpen: boolean;
  onClose: () => void;
  sessionId: string;
  onAddServer: () => void;
  storageAccessEnabled: boolean;
  onCopySessionId: () => void;
  sessionCopyFeedback: string | null;
  /** Gated by `features.feedback_email` from /api/info. */
  feedbackEnabled: boolean;
  onOpenFeedback: () => void;
}

// Chevron icon for expand/collapse
const FeedbackIcon = () => (
  <svg
    width="16"
    height="16"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"></path>
  </svg>
);

const ChevronIcon = ({ expanded }: { expanded: boolean }) => (
  <svg
    width="12"
    height="12"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    style={{
      transform: expanded ? "rotate(90deg)" : "rotate(0deg)",
      transition: "transform 0.2s ease",
    }}
  >
    <polyline points="9 18 15 12 9 6"></polyline>
  </svg>
);

// Remove/trash icon
const RemoveIcon = () => (
  <svg
    width="14"
    height="14"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <polyline points="3 6 5 6 21 6"></polyline>
    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
  </svg>
);

// Plus icon for add button
const PlusIcon = () => (
  <svg
    width="16"
    height="16"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <line x1="12" y1="5" x2="12" y2="19"></line>
    <line x1="5" y1="12" x2="19" y2="12"></line>
  </svg>
);

// Server/plug icon
const ServerIcon = () => (
  <svg
    width="16"
    height="16"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <rect x="2" y="2" width="20" height="8" rx="2" ry="2"></rect>
    <rect x="2" y="14" width="20" height="8" rx="2" ry="2"></rect>
    <line x1="6" y1="6" x2="6.01" y2="6"></line>
    <line x1="6" y1="18" x2="6.01" y2="18"></line>
  </svg>
);

// GPU/chip icon, for a model the user hosts themselves
const NimIcon = () => (
  <svg
    width="16"
    height="16"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <rect x="6" y="6" width="12" height="12" rx="2" ry="2"></rect>
    <line x1="9" y1="2" x2="9" y2="6"></line>
    <line x1="15" y1="2" x2="15" y2="6"></line>
    <line x1="9" y1="18" x2="9" y2="22"></line>
    <line x1="15" y1="18" x2="15" y2="22"></line>
    <line x1="2" y1="9" x2="6" y2="9"></line>
    <line x1="2" y1="15" x2="6" y2="15"></line>
    <line x1="18" y1="9" x2="22" y2="9"></line>
    <line x1="18" y1="15" x2="22" y2="15"></line>
  </svg>
);

// Book/skill icon
const SkillIcon = () => (
  <svg
    width="16"
    height="16"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
    <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
  </svg>
);

// Tool/wrench icon
const ToolIcon = () => (
  <svg
    width="14"
    height="14"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"></path>
  </svg>
);

// Key icon for session section
const KeyIcon = () => (
  <svg
    width="16"
    height="16"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4"></path>
  </svg>
);

// Copy icon (two overlapping rectangles)
const CopyIcon = () => (
  <svg
    width="14"
    height="14"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
  </svg>
);

// Check icon for copy feedback
const CheckIcon = () => (
  <svg
    width="14"
    height="14"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <polyline points="20 6 9 17 4 12"></polyline>
  </svg>
);

/** Truncate a description for use in a native title tooltip. */
const truncateTooltip = (
  text: string | undefined,
  max = 1024,
): string | undefined => {
  if (!text) return undefined;
  if (text.length <= max) return text;
  return text.slice(0, max) + "…";
};

export default function ToolManagerPanel({
  isOpen,
  onClose,
  sessionId,
  onAddServer,
  storageAccessEnabled,
  onCopySessionId,
  sessionCopyFeedback,
  feedbackEnabled,
  onOpenFeedback,
}: ToolManagerPanelProps) {
  const [mcpServers, setMcpServers] = useState<MCPServer[]>([]);
  const [nimServers, setNimServers] = useState<NIMServer[]>([]);
  const [builtinTools, setBuiltinTools] = useState<ToolInfo[]>([]);
  const [skills, setSkills] = useState<Skill[]>([]);
  const [secrets, setSecrets] = useState<Secret[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [builtinExpanded, setBuiltinExpanded] = useState(false);
  const [mcpExpanded, setMcpExpanded] = useState<Record<string, boolean>>({});
  const [skillsExpanded, setSkillsExpanded] = useState(false);
  const [secretsExpanded, setSecretsExpanded] = useState(false);
  const [nimExpanded, setNimExpanded] = useState<Record<string, boolean>>({});
  const [removingServer, setRemovingServer] = useState<string | null>(null);
  const [confirmingServer, setConfirmingServer] = useState<string | null>(null);
  const [removingNim, setRemovingNim] = useState<string | null>(null);
  const [confirmingNim, setConfirmingNim] = useState<string | null>(null);
  const [removingSkill, setRemovingSkill] = useState<string | null>(null);
  const [removingSecret, setRemovingSecret] = useState<string | null>(null);

  // Fetch tools and MCP servers
  const fetchTools = useCallback(async () => {
    try {
      setIsLoading(true);

      // Fetch all tools
      const toolsResponse = await fetch(
        `/api/mcp/tools?session_id=${encodeURIComponent(sessionId)}`,
      );
      if (toolsResponse.ok) {
        const toolsData = await toolsResponse.json();
        setBuiltinTools(
          toolsData.tools.filter((t: ToolInfo) => t.source === "builtin"),
        );
      }

      // Fetch MCP servers
      const serversResponse = await fetch(
        `/api/mcp/servers?session_id=${encodeURIComponent(sessionId)}`,
      );
      if (serversResponse.ok) {
        const serversData = await serversResponse.json();
        setMcpServers(serversData.servers || []);
      }

      // Fetch registered NIMs
      const nimResponse = await fetch(
        `/api/nim/servers?session_id=${encodeURIComponent(sessionId)}`,
      );
      if (nimResponse.ok) {
        const nimData = await nimResponse.json();
        setNimServers(nimData.servers || []);
      }

      // Fetch skills
      const skillsResponse = await fetch(
        `/api/skills/?session_id=${encodeURIComponent(sessionId)}`,
      );
      if (skillsResponse.ok) {
        const skillsData = await skillsResponse.json();
        setSkills(skillsData.skills || []);
      }

      // Fetch stored API keys (names only)
      const secretsResponse = await fetch(
        `/api/secrets/?session_id=${encodeURIComponent(sessionId)}`,
      );
      if (secretsResponse.ok) {
        const secretsData = await secretsResponse.json();
        setSecrets(secretsData.secrets || []);
      }
    } catch (error) {
      console.error("Failed to fetch tools:", error);
    } finally {
      setIsLoading(false);
    }
  }, [sessionId]);

  // Fetch on mount and when panel opens
  useEffect(() => {
    if (isOpen) {
      fetchTools();
    }
  }, [isOpen, fetchTools]);

  // Handle removing an MCP server
  const handleRemoveServer = async (serverId: string) => {
    if (removingServer) return;

    setRemovingServer(serverId);
    try {
      const response = await fetch(
        `/api/mcp/servers/${serverId}?session_id=${encodeURIComponent(sessionId)}`,
        { method: "DELETE" },
      );

      if (response.ok) {
        setMcpServers((prev) => prev.filter((s) => s.id !== serverId));
      } else {
        console.error("Failed to remove server");
      }
    } catch (error) {
      console.error("Error removing server:", error);
    } finally {
      setRemovingServer(null);
    }
  };

  // Authorise an MCP server's tools. Until this runs the backend withholds
  // them from the agent entirely -- they are neither offered to the model nor
  // dispatchable -- so this is what actually turns the server on.
  const handleConfirmServer = async (serverId: string) => {
    if (confirmingServer) return;

    setConfirmingServer(serverId);
    try {
      const response = await fetch(
        `/api/mcp/servers/${serverId}/confirm?session_id=${encodeURIComponent(sessionId)}`,
        { method: "POST" },
      );

      if (response.ok) {
        setMcpServers((prev) =>
          prev.map((s) => (s.id === serverId ? { ...s, confirmed: true } : s)),
        );
        // The agent's toolset just changed; refresh so the tool list agrees.
        fetchTools();
      } else {
        console.error("Failed to confirm server");
      }
    } catch (error) {
      console.error("Error confirming server:", error);
    } finally {
      setConfirmingServer(null);
    }
  };

  // Toggle MCP server expansion
  const toggleMcpExpanded = (serverId: string) => {
    setMcpExpanded((prev) => ({
      ...prev,
      [serverId]: !prev[serverId],
    }));
  };

  // Handle removing a registered NIM
  const handleRemoveNim = async (serverId: string) => {
    if (removingNim) return;

    setRemovingNim(serverId);
    try {
      const response = await fetch(
        `/api/nim/servers/${serverId}?session_id=${encodeURIComponent(sessionId)}`,
        { method: "DELETE" },
      );
      if (response.ok) {
        setNimServers((prev) => prev.filter((s) => s.id !== serverId));
        // Removing a NIM can also hand a built-in tool back to the hosted
        // endpoint, so the tool list is no longer accurate either.
        fetchTools();
      } else {
        console.error("Failed to remove NIM");
      }
    } catch (error) {
      console.error("Error removing NIM:", error);
    } finally {
      setRemovingNim(null);
    }
  };

  // Enable a NIM. Until this runs the backend withholds its tools and will
  // not re-point any built-in tool at it, so this is what turns it on.
  const handleConfirmNim = async (serverId: string) => {
    if (confirmingNim) return;

    setConfirmingNim(serverId);
    try {
      const response = await fetch(
        `/api/nim/servers/${serverId}/confirm?session_id=${encodeURIComponent(sessionId)}`,
        { method: "POST" },
      );
      if (response.ok) {
        setNimServers((prev) =>
          prev.map((s) => (s.id === serverId ? { ...s, confirmed: true } : s)),
        );
        fetchTools();
      } else {
        console.error("Failed to confirm NIM");
      }
    } catch (error) {
      console.error("Error confirming NIM:", error);
    } finally {
      setConfirmingNim(null);
    }
  };

  // Toggle NIM expansion
  const toggleNimExpanded = (serverId: string) => {
    setNimExpanded((prev) => ({
      ...prev,
      [serverId]: !prev[serverId],
    }));
  };

  // Handle removing a stored API key
  const handleRemoveSecret = async (secretId: string) => {
    if (removingSecret) return;

    setRemovingSecret(secretId);
    try {
      const response = await fetch(
        `/api/secrets/${secretId}?session_id=${encodeURIComponent(sessionId)}`,
        { method: "DELETE" },
      );
      if (response.ok) {
        setSecrets((prev) => prev.filter((s) => s.id !== secretId));
      } else {
        console.error("Failed to remove key");
      }
    } catch (error) {
      console.error("Error removing key:", error);
    } finally {
      setRemovingSecret(null);
    }
  };

  // Handle removing a skill
  const handleRemoveSkill = async (skillId: string) => {
    if (removingSkill) return;

    setRemovingSkill(skillId);
    try {
      const response = await fetch(
        `/api/skills/${skillId}?session_id=${encodeURIComponent(sessionId)}`,
        { method: "DELETE" },
      );
      if (response.ok) {
        setSkills((prev) => prev.filter((s) => s.id !== skillId));
      } else {
        console.error("Failed to remove skill");
      }
    } catch (error) {
      console.error("Error removing skill:", error);
    } finally {
      setRemovingSkill(null);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="tool-manager-panel">
      {/* Header */}
      <div className="tool-manager-header">
        <h3>Tools</h3>
        <div className="tool-manager-header-actions">
          <button
            className="add-tool-btn"
            onClick={onAddServer}
            title="Add MCP server or skill"
          >
            <PlusIcon />
            <span>Add</span>
          </button>
          <button
            className="close-panel-btn"
            onClick={onClose}
            title="Close panel"
          >
            →
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="tool-manager-content">
        {isLoading ? (
          <div className="tool-manager-loading">Loading tools...</div>
        ) : (
          <>
            {/* Session ID Section (feature-flagged) */}
            {storageAccessEnabled && sessionId && (
              <div className="tool-section session-section">
                <div className="tool-section-header">
                  <KeyIcon />
                  <span>Session ID</span>
                </div>
                <div className="session-id-row">
                  <code className="session-id-value">{sessionId}</code>
                  <button
                    className="session-copy-icon-btn"
                    onClick={onCopySessionId}
                    title={sessionCopyFeedback || "Copy session ID"}
                  >
                    {sessionCopyFeedback ? <CheckIcon /> : <CopyIcon />}
                  </button>
                </div>
              </div>
            )}

            {/* MCP Servers Section */}
            {mcpServers.length > 0 && (
              <div className="tool-section mcp-servers-section">
                <div className="tool-section-header">
                  <ServerIcon />
                  <span>MCP Servers</span>
                  <span className="tool-count">{mcpServers.length}</span>
                </div>

                {mcpServers.map((server) => (
                  <div key={server.id} className="mcp-server-item">
                    <div
                      className="mcp-server-header"
                      onClick={() => toggleMcpExpanded(server.id)}
                    >
                      <ChevronIcon expanded={mcpExpanded[server.id] || false} />
                      <span className="mcp-server-name">{server.name}</span>
                      <span
                        className={`mcp-server-status ${server.status}`}
                        title={server.status}
                      >
                        ●
                      </span>
                      <span className="mcp-server-tool-count">
                        {server.tools_count} tools
                        {server.confirmed === false && " offered"}
                      </span>
                      {server.confirmed === false && (
                        <button
                          className="confirm-server-btn"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleConfirmServer(server.id);
                          }}
                          disabled={confirmingServer === server.id}
                          title={
                            `Review the ${server.tools_count} tool(s) below, then ` +
                            "authorise them. Until you do, the agent cannot see " +
                            "or call any of them."
                          }
                        >
                          {confirmingServer === server.id
                            ? "..."
                            : "Review & enable"}
                        </button>
                      )}
                      <button
                        className="remove-server-btn"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleRemoveServer(server.id);
                        }}
                        disabled={removingServer === server.id}
                        title="Remove server"
                      >
                        {removingServer === server.id ? "..." : <RemoveIcon />}
                      </button>
                    </div>

                    {mcpExpanded[server.id] && (
                      <div className="mcp-server-tools">
                        {server.tools.map((tool, idx) => (
                          <div
                            key={idx}
                            className="tool-item"
                            title={truncateTooltip(tool.description)}
                          >
                            <ToolIcon />
                            <span className="tool-name">{tool.name}</span>
                            {tool.description && (
                              <span className="tool-description">
                                {tool.description.slice(0, 50)}
                                {tool.description.length > 50 ? "..." : ""}
                              </span>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            {/* Self-hosted NIMs Section */}
            {nimServers.length > 0 && (
              <div className="tool-section mcp-servers-section">
                <div className="tool-section-header">
                  <NimIcon />
                  <span>NIMs</span>
                  <span className="tool-count">{nimServers.length}</span>
                </div>

                {nimServers.map((server) => {
                  // A NIM with no adapter is reachable but contributes
                  // nothing, so it must not offer a "Review & enable" button
                  // that would authorise an empty set.
                  const hasAdapter = server.adapter !== null;
                  const contributes =
                    server.tools.length + server.repointed_tools.length;
                  return (
                    <div key={server.id} className="mcp-server-item">
                      <div
                        className="mcp-server-header"
                        onClick={() => toggleNimExpanded(server.id)}
                      >
                        <ChevronIcon
                          expanded={nimExpanded[server.id] || false}
                        />
                        <span className="mcp-server-name">{server.name}</span>
                        <span
                          className={`mcp-server-status ${
                            server.ready ? "connected" : "disconnected"
                          }`}
                          title={server.ready ? "ready" : "not ready"}
                        >
                          ●
                        </span>
                        <span className="mcp-server-tool-count">
                          {hasAdapter
                            ? `${contributes} tool${contributes === 1 ? "" : "s"}`
                            : "no adapter"}
                          {hasAdapter && server.confirmed === false
                            ? " offered"
                            : ""}
                        </span>
                        {hasAdapter && server.confirmed === false && (
                          <button
                            className="confirm-server-btn"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleConfirmNim(server.id);
                            }}
                            disabled={confirmingNim === server.id}
                            title={
                              "Review what this NIM contributes below, then " +
                              "authorise it. Until you do, the agent cannot " +
                              "call it and no built-in tool is re-pointed."
                            }
                          >
                            {confirmingNim === server.id
                              ? "..."
                              : "Review & enable"}
                          </button>
                        )}
                        <button
                          className="remove-server-btn"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleRemoveNim(server.id);
                          }}
                          disabled={removingNim === server.id}
                          title="Remove NIM"
                        >
                          {removingNim === server.id ? "..." : <RemoveIcon />}
                        </button>
                      </div>

                      {nimExpanded[server.id] && (
                        <div className="mcp-server-tools">
                          <div
                            className="tool-item"
                            title={server.adapter_summary || undefined}
                          >
                            <NimIcon />
                            <span className="tool-name">
                              {server.model_id ?? "unknown model"}
                            </span>
                            {server.model_version && (
                              <span className="tool-description">
                                v{server.model_version}
                              </span>
                            )}
                          </div>

                          {server.tools.map((tool) => (
                            <div key={tool} className="tool-item">
                              <ToolIcon />
                              <span className="tool-name">{tool}</span>
                            </div>
                          ))}

                          {server.repointed_tools.map((tool) => (
                            <div
                              key={`repoint-${tool}`}
                              className="tool-item"
                              title={
                                "This NIM serves the built-in tool in place " +
                                "of the hosted NVIDIA endpoint."
                              }
                            >
                              <ToolIcon />
                              <span className="tool-name">{tool}</span>
                              <span className="tool-description">
                                re-pointed here
                              </span>
                            </div>
                          ))}

                          {!hasAdapter && (
                            <div className="tool-item">
                              <span className="tool-description">
                                Aevom has no adapter for this model, so it
                                contributes no tools.
                              </span>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}

            {/* Skills Section */}
            {skills.length > 0 && (
              <div className="tool-section skills-section">
                <button
                  className="tool-section-header clickable"
                  onClick={() => setSkillsExpanded(!skillsExpanded)}
                >
                  <ChevronIcon expanded={skillsExpanded} />
                  <SkillIcon />
                  <span>Skills</span>
                  <span className="tool-count">{skills.length}</span>
                </button>

                {skillsExpanded && (
                  <div className="tool-list">
                    {skills.map((skill) => (
                      <div
                        key={skill.id}
                        className="tool-item"
                        title={truncateTooltip(skill.description)}
                      >
                        <SkillIcon />
                        <span className="tool-name">{skill.name}</span>
                        <span className="skill-source-badge" title={skill.source}>
                          {skill.source === "builtin" ? "built-in" : "session"}
                        </span>
                        {skill.removable && (
                          <button
                            className="remove-server-btn"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleRemoveSkill(skill.id);
                            }}
                            disabled={removingSkill === skill.id}
                            title="Remove skill"
                          >
                            {removingSkill === skill.id ? "..." : <RemoveIcon />}
                          </button>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* API Keys Section */}
            {secrets.length > 0 && (
              <div className="tool-section keys-section">
                <button
                  className="tool-section-header clickable"
                  onClick={() => setSecretsExpanded(!secretsExpanded)}
                >
                  <ChevronIcon expanded={secretsExpanded} />
                  <KeyIcon />
                  <span>API Keys</span>
                  <span className="tool-count">{secrets.length}</span>
                </button>

                {secretsExpanded && (
                  <div className="tool-list">
                    {secrets.map((secret) => (
                      <div key={secret.id} className="tool-item">
                        <KeyIcon />
                        <span className="tool-name">{secret.name}</span>
                        <button
                          className="remove-server-btn"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleRemoveSecret(secret.id);
                          }}
                          disabled={removingSecret === secret.id}
                          title="Remove key"
                        >
                          {removingSecret === secret.id ? "..." : <RemoveIcon />}
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Builtin Tools Section */}
            <div className="tool-section builtin-section">
              <button
                className="tool-section-header clickable"
                onClick={() => setBuiltinExpanded(!builtinExpanded)}
              >
                <ChevronIcon expanded={builtinExpanded} />
                <ToolIcon />
                <span>Built-in Tools</span>
                <span className="tool-count">{builtinTools.length}</span>
              </button>

              {builtinExpanded && (
                <div className="tool-list">
                  {builtinTools.map((tool, idx) => (
                    <div
                      key={idx}
                      className="tool-item"
                      title={truncateTooltip(tool.description)}
                    >
                      <span className="tool-name">{tool.name}</span>
                      {tool.description && (
                        <span className="tool-description">
                          {tool.description.slice(0, 60)}
                          {tool.description.length > 60 ? "..." : ""}
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Empty state, shown only when nothing external is registered */}
            {mcpServers.length === 0 && nimServers.length === 0 && (
              <div className="mcp-empty-state">
                <ServerIcon />
                <p>No MCP servers connected</p>
                <button className="add-server-hint-btn" onClick={onAddServer}>
                  Add your first MCP server
                </button>
              </div>
            )}

            {/* Feedback, last in the panel (feature-flagged) */}
            {feedbackEnabled && (
              <div className="tool-section feedback-section">
                <button
                  className="feedback-btn"
                  onClick={onOpenFeedback}
                  title="Send feedback to the Aevom team"
                >
                  <FeedbackIcon />
                  <span>Feedback</span>
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
