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
  useEffect,
  FormEvent,
  useCallback,
  useMemo,
  forwardRef,
  useImperativeHandle,
} from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github-dark.css";
import { ToolCallTimeline, getBriefSummary } from "./ToolCallTimeline";

// Timeline item representing a step in the reasoning/tool chain
export interface TimelineItem {
  id: string;
  type: "reasoning" | "tool_call";
  iteration: number;
  timestamp: number;
  // For reasoning items
  content?: string;
  // For tool_call items
  toolName?: string;
  toolArgs?: Record<string, unknown>;
  summary?: string;
  resultRef?: string | null;
  toolOutput?: string;
  status?: "pending" | "complete" | "error";
}

interface Message {
  role: "user" | "assistant";
  content: string;
  reasoning?: string;
  timelineItems?: TimelineItem[];
  isStreaming?: boolean;
  isInterrupted?: boolean;
  span_id?: string;
  feedbackSubmitted?: "thumbs_up" | "thumbs_down" | null;
  rateLimitMessage?: string;
}

// Tool event types from the backend
export interface ToolEvent {
  type: string;
  sdf?: string;
  pdb?: string;
  label?: string;
  /** Parser to use for `pdb`. Absent on older events, which are always PDB. */
  format?: "pdb" | "mmcif";
  /** Which label chains are the target and which the designed binder, for designs. */
  roles?: { target: string[]; binder: string[] };
  // Structure query fields (MolQL)
  query_id?: string; // Tool call ID for correlation with selection results
  session_id?: string; // Backend session_id for selection POST (prevents mismatch)
  query?: StructureQuery;
  // show_selection fields: residues are addressed by label ids
  selection_ref?: string;
  residues?: { chain: string; seq: number }[];
  // Residues with only some atoms chosen arrive here instead of in `residues`.
  atoms?: { chain: string; seq: number; name: string }[];
  focus?: boolean;
  colour?: string;
}

// MolQL structure query specification
export interface StructureQuery {
  molql: string;
  action?: "highlight" | "focus" | "select" | "info";
  /** Optional short name the agent gave the resulting selection. */
  name?: string;
}

// Viewer state for LLM context
export interface ViewerState {
  isEmpty: boolean;
  structures: {
    label: string;
    structureType: string;
    format: string;
    chainIds: string[];
    residueCount: number;
    atomCount: number;
    hasPolymer: boolean;
    hasLigand: boolean;
    hasWater: boolean;
    metadata: Record<string, unknown>;
  }[];
  hasSelection: boolean;
  selectionInfo: string | null;
}

/** Available artifact for autocomplete */
export interface AvailableArtifact {
  reference: string;
  data_type: string;
  metadata: Record<string, unknown>;
}

/** Custom options to pass with chat requests (for training mode) */
export interface CustomFetchOptions {
  prompt_mode?: string;
  training_query_id?: string;
  expected_tool?: string;
  oracle_prompt?: string;
}

/** Artifact data from storage_update SSE events */
export interface StorageUpdateArtifact {
  reference: string;
  data_type: string;
  description?: string | null;
  metadata?: Record<string, unknown>;
  created_at?: string | null;
}

/** Model info from /api/models */
export interface ModelInfo {
  id: string;
  pretty_name: string;
  /** "openai_compat" (NVIDIA endpoints) or "anthropic". */
  provider?: string;
  supports_reasoning: boolean;
}

interface ChatProps {
  onToolEvent?: (event: ToolEvent) => void;
  getViewerState?: () => ViewerState;
  /**
   * Awaited just before a message is sent, with the session it is sent in.
   * The viewer uses it to store residues the user picked by hand, so the agent
   * sees them in this turn. Must not throw; a failure must not stop the send.
   */
  beforeSend?: (sessionId: string | undefined) => Promise<unknown>;
  /** Session ID for scoping artifacts to current session */
  sessionId?: string;
  /** Called when a chat response stream completes (after all tool calls are done) */
  onResponseComplete?: () => void;
  /** Called when storage is updated between tool calls (for real-time strip updates) */
  onStorageUpdate?: (artifacts: StorageUpdateArtifact[]) => void;
  /** Available artifacts for @ autocomplete */
  availableArtifacts?: AvailableArtifact[];
  /** Custom options to include in fetch requests (for training mode) */
  customFetchOptions?: CustomFetchOptions;
  /** Called after feedback is successfully submitted */
  onFeedbackSubmitted?: () => void;
  /** Currently selected model ID (from header dropdown) */
  selectedModel?: string;
  /** Last model used for a request (to detect model switch) */
  activeModel?: string;
  /** Called when submitting with a different model; resets state, resolves to the new sessionId */
  onModelSwitch?: (newModelId: string) => Promise<string>;
}

/** Imperative handle for Chat component */
export interface ChatRef {
  /** Insert a reference bubble at the current cursor position */
  insertBubble: (reference: string) => void;
  /** Set the input text content */
  setInputText: (text: string) => void;
}

/**
 * Pre-process streaming markdown content to hide incomplete syntax.
 * This prevents flickering when markdown elements arrive in chunks.
 */
function sanitizeStreamingMarkdown(
  content: string,
  isStreaming: boolean,
): string {
  if (!isStreaming || !content) return content;

  let result = content;

  // Hide incomplete image syntax: ![...](...) where the closing ) is missing
  // Match ![anything](anything-without-closing-paren at end of string
  result = result.replace(/!\[[^\]]*\]\([^)]*$/g, "");

  // Hide incomplete image alt text: ![... at end without ]
  result = result.replace(/!\[[^\]]*$/g, "");

  // Hide incomplete link syntax: [...](...) where the closing ) is missing
  // But don't match if it looks like it might be a complete link followed by more text
  result = result.replace(/\[([^\]]*)\]\([^)]*$/g, "");

  // Hide incomplete link text: [... at end without ]
  // But be careful not to remove valid content that just happens to end with [
  result = result.replace(/\[[^\]]*$/g, (match) => {
    // Only remove if it looks like the start of a link (has some content)
    return match.length > 1 ? "" : match;
  });

  // Hide incomplete inline code that spans end of content
  const backtickCount = (result.match(/`/g) || []).length;
  if (backtickCount % 2 !== 0) {
    // Odd number of backticks - find and hide the incomplete one at the end
    const lastBacktick = result.lastIndexOf("`");
    if (lastBacktick !== -1) {
      result = result.substring(0, lastBacktick);
    }
  }

  return result;
}

// Module-level cache to track which images have successfully loaded.
// This prevents the loading spinner from reappearing when ReactMarkdown
// re-renders and recreates the MoleculeImage component.
const loadedImageCache = new Set<string>();

// Custom image component for molecule visualizations with stable layout
const MoleculeImage = ({ src, alt }: { src?: string; alt?: string }) => {
  // Check if this image was already loaded (survives component re-mounts)
  const alreadyLoaded = src ? loadedImageCache.has(src) : false;

  const [isLoading, setIsLoading] = useState(!alreadyLoaded);
  const [showLoader, setShowLoader] = useState(false);
  const [hasError, setHasError] = useState(false);
  const loaderTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // If src changes, reset states appropriately
  useEffect(() => {
    if (src) {
      if (loadedImageCache.has(src)) {
        setIsLoading(false);
        setShowLoader(false);
        setHasError(false);
      } else {
        setIsLoading(true);
        setHasError(false);
        // Delay showing loader to prevent flash for cached images
        loaderTimeoutRef.current = setTimeout(() => {
          setShowLoader(true);
        }, 150);
      }
    }

    return () => {
      if (loaderTimeoutRef.current) {
        clearTimeout(loaderTimeoutRef.current);
      }
    };
  }, [src]);

  if (!src) return null;

  const handleLoad = () => {
    if (loaderTimeoutRef.current) {
      clearTimeout(loaderTimeoutRef.current);
    }
    loadedImageCache.add(src);
    setIsLoading(false);
    setShowLoader(false);
  };

  const handleError = () => {
    if (loaderTimeoutRef.current) {
      clearTimeout(loaderTimeoutRef.current);
    }
    setIsLoading(false);
    setShowLoader(false);
    setHasError(true);
  };

  return (
    <figure className="molecule-figure">
      {/* Container with fixed aspect ratio to prevent layout shift */}
      <div className="molecule-image-container">
        {showLoader && isLoading && !hasError && (
          <div className="molecule-loading-overlay">
            <span className="loading-spinner"></span>
            <span>Loading molecule...</span>
          </div>
        )}
        {hasError ? (
          <div className="molecule-error">
            <span>Failed to load molecule image</span>
          </div>
        ) : (
          <img
            src={src}
            alt={alt || "Molecule structure"}
            className={`molecule-image ${isLoading ? "loading" : "loaded"}`}
            onLoad={handleLoad}
            onError={handleError}
          />
        )}
      </div>
      {alt && !hasError && (
        <figcaption className="molecule-caption">{alt}</figcaption>
      )}
    </figure>
  );
};

// Collapsible reasoning/thinking section with timeline
interface ReasoningSectionProps {
  reasoning: string;
  timelineItems?: TimelineItem[];
  isStreaming?: boolean;
  isInterrupted?: boolean;
}

const ReasoningSection = ({
  reasoning,
  timelineItems = [],
  isStreaming,
  isInterrupted,
}: ReasoningSectionProps) => {
  const [isExpanded, setIsExpanded] = useState(false);

  // Nothing to show if no reasoning and no timeline
  if (!reasoning && timelineItems.length === 0) return null;

  // Get brief summary for the header
  const briefSummary =
    timelineItems.length > 0
      ? getBriefSummary(timelineItems)
      : isStreaming
        ? "Thinking..."
        : "Show thinking";

  // Check if there are any tool calls in the timeline
  const hasToolCalls = timelineItems.some((item) => item.type === "tool_call");

  return (
    <div
      className={`reasoning-section ${isExpanded ? "expanded" : "collapsed"} ${
        hasToolCalls ? "has-tools" : ""
      }`}
    >
      <button
        className="reasoning-toggle"
        onClick={() => setIsExpanded(!isExpanded)}
        aria-expanded={isExpanded}
      >
        <span className={`reasoning-icon ${isStreaming ? "streaming" : ""}`}>
          ⚗️
        </span>
        <span className="reasoning-label">{briefSummary}</span>
        <span className={`reasoning-chevron ${isExpanded ? "rotated" : ""}`}>
          ▶
        </span>
      </button>

      {/* Timeline shown below header when collapsed (if there are tool calls) */}
      {!isExpanded && hasToolCalls && (
        <div className="reasoning-timeline-preview">
          <ToolCallTimeline
            items={timelineItems}
            isStreaming={isStreaming}
            isExpanded={false}
            isInterrupted={isInterrupted}
          />
        </div>
      )}

      {/* Expanded content: interleaved reasoning and tool calls */}
      {isExpanded && (
        <div className="reasoning-content">
          {timelineItems.length > 0 ? (
            <ToolCallTimeline
              items={timelineItems}
              isStreaming={isStreaming}
              isExpanded={true}
              isInterrupted={isInterrupted}
            />
          ) : (
            <pre>{reasoning}</pre>
          )}
        </div>
      )}
    </div>
  );
};

// SVG icons for feedback buttons (outlined style)
const ThumbUpIcon = ({ selected = false }: { selected?: boolean }) => (
  <svg
    width="16"
    height="16"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={selected ? 2.5 : 1.5}
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3" />
  </svg>
);

const ThumbDownIcon = ({ selected = false }: { selected?: boolean }) => (
  <svg
    width="16"
    height="16"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={selected ? 2.5 : 1.5}
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17" />
  </svg>
);

// Stop icon for cancelling generation (square shape)
const StopIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
    <rect x="6" y="6" width="12" height="12" rx="2" />
  </svg>
);

// Feedback buttons component for thumbs up/down on assistant messages
interface FeedbackButtonsProps {
  span_id: string;
  feedbackSubmitted: "thumbs_up" | "thumbs_down" | null;
  onFeedback: (span_id: string, label: "thumbs_up" | "thumbs_down") => void;
  isStreaming?: boolean;
}

const FeedbackButtons = ({
  span_id,
  feedbackSubmitted,
  onFeedback,
  isStreaming,
}: FeedbackButtonsProps) => {
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleClick = async (label: "thumbs_up" | "thumbs_down") => {
    if (feedbackSubmitted || isSubmitting || isStreaming) return;

    setIsSubmitting(true);
    try {
      await onFeedback(span_id, label);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Don't show feedback buttons while streaming
  if (isStreaming) return null;

  return (
    <div className="feedback-buttons">
      <button
        className={`feedback-btn thumbs-up ${
          feedbackSubmitted === "thumbs_up" ? "selected" : ""
        } ${feedbackSubmitted && feedbackSubmitted !== "thumbs_up" ? "disabled" : ""}`}
        onClick={() => handleClick("thumbs_up")}
        disabled={!!feedbackSubmitted || isSubmitting}
        title="This response was helpful"
      >
        <ThumbUpIcon selected={feedbackSubmitted === "thumbs_up"} />
      </button>
      <button
        className={`feedback-btn thumbs-down ${
          feedbackSubmitted === "thumbs_down" ? "selected" : ""
        } ${feedbackSubmitted && feedbackSubmitted !== "thumbs_down" ? "disabled" : ""}`}
        onClick={() => handleClick("thumbs_down")}
        disabled={!!feedbackSubmitted || isSubmitting}
        title="This response was not helpful"
      >
        <ThumbDownIcon selected={feedbackSubmitted === "thumbs_down"} />
      </button>
    </div>
  );
};

function UserAvatarIcon() {
  return (
    <svg
      className="avatar-icon"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
      <circle cx="12" cy="7" r="4" />
    </svg>
  );
}

// Memoized message bubble component to prevent unnecessary re-renders
interface MessageBubbleProps {
  message: Message;
  onFeedback?: (span_id: string, label: "thumbs_up" | "thumbs_down") => void;
}

const MessageBubble = ({ message, onFeedback }: MessageBubbleProps) => {
  // Memoize the sanitized content to prevent unnecessary recalculations
  const displayContent = useMemo(() => {
    if (!message.content) return "";
    return sanitizeStreamingMarkdown(
      message.content,
      message.isStreaming ?? false,
    );
  }, [message.content, message.isStreaming]);

  return (
    <div className={`message ${message.role}`}>
      <div className="message-avatar">
        {message.role === "user" ? <UserAvatarIcon /> : "🧬"}
      </div>
      <div className="message-content">
        <div className="message-role">
          {message.role === "user" ? "You" : "Aevom"}
          {message.role === "assistant" && message.rateLimitMessage && (
            <span className="rate-limit-status">
              {message.rateLimitMessage}
            </span>
          )}
        </div>
        {/* Show reasoning section for assistant messages */}
        {message.role === "assistant" &&
          (message.reasoning ||
            (message.timelineItems && message.timelineItems.length > 0)) && (
            <ReasoningSection
              reasoning={message.reasoning || ""}
              timelineItems={message.timelineItems}
              isStreaming={message.isStreaming}
              isInterrupted={message.isInterrupted}
            />
          )}
        <div className="message-text">
          {message.content ? (
            message.role === "assistant" ? (
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                rehypePlugins={[rehypeHighlight]}
                components={{
                  img: ({ src, alt }) => <MoleculeImage src={src} alt={alt} />,
                }}
              >
                {displayContent}
              </ReactMarkdown>
            ) : (
              message.content
            )
          ) : message.isStreaming ? (
            message.reasoning ? (
              <span className="thinking-status">Processing...</span>
            ) : (
              <span className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </span>
            )
          ) : message.isInterrupted ? (
            // Message was interrupted by user
            <span className="interrupted-status">Interrupted</span>
          ) : message.reasoning ? (
            // Streaming complete with reasoning but no content - show fallback
            <span className="no-content-message">
              <em>Response completed. See reasoning above for details.</em>
            </span>
          ) : (
            // No content and no reasoning - shouldn't normally happen
            <span className="no-content-message">
              <em>No response generated.</em>
            </span>
          )}
        </div>
        {/* Feedback buttons for assistant messages */}
        {message.role === "assistant" && message.span_id && onFeedback && (
          <FeedbackButtons
            span_id={message.span_id}
            feedbackSubmitted={message.feedbackSubmitted ?? null}
            onFeedback={onFeedback}
            isStreaming={message.isStreaming}
          />
        )}
      </div>
    </div>
  );
};

/**
 * Get a short display label for a reference (e.g., "SDF" from "sdf_abc123")
 */
function getRefTypeLabel(ref: string): string {
  const parts = ref.split("_");
  if (parts.length > 0) {
    return parts[0].toUpperCase();
  }
  return "REF";
}

/**
 * Get the short ID portion of a reference (e.g., "abc123" from "sdf_abc123")
 */
function getRefShortId(ref: string): string {
  const parts = ref.split("_");
  if (parts.length > 1) {
    return parts.slice(1).join("_").slice(0, 6);
  }
  return ref.slice(0, 6);
}

/** One message as the backend records it. */
interface RecordedMessage {
  role: string;
  content: string;
  reasoning?: string | null;
  tool_calls?:
    | { id: string; name: string; args: Record<string, unknown> }[]
    | null;
  tool_call_id?: string | null;
  name?: string | null;
}

/**
 * Rebuild the rendered conversation from the backend's record.
 *
 * The record is flat -- user, assistant, tool, assistant -- while this
 * component renders one bubble per exchange with the intervening tool calls
 * folded into its timeline. So assistant messages that only requested tools
 * become timeline items on the assistant bubble that eventually answers.
 */
function hydrateMessages(records: RecordedMessage[]): Message[] {
  const rendered: Message[] = [];
  let timeline: TimelineItem[] = [];
  let iteration = 0;

  const flushInterrupted = () => {
    if (timeline.length === 0) return;
    rendered.push({
      role: "assistant",
      content: "",
      timelineItems: timeline,
      isInterrupted: true,
    });
    timeline = [];
  };

  for (const record of records) {
    if (record.role === "user") {
      flushInterrupted();
      rendered.push({ role: "user", content: record.content });
      iteration = 0;
      continue;
    }

    if (record.role === "tool") {
      const item = timeline.find((entry) => entry.id === record.tool_call_id);
      if (item) {
        item.toolOutput = record.content;
        item.status = record.content.startsWith("Error") ? "error" : "complete";
      }
      continue;
    }

    if (record.role !== "assistant") continue;

    if (record.reasoning) {
      timeline.push({
        id: `reasoning-${rendered.length}-${timeline.length}`,
        type: "reasoning",
        iteration,
        timestamp: Date.now(),
        content: record.reasoning,
      });
    }

    if (record.tool_calls && record.tool_calls.length > 0) {
      iteration += 1;
      for (const call of record.tool_calls) {
        timeline.push({
          id: call.id,
          type: "tool_call",
          iteration,
          timestamp: Date.now(),
          toolName: call.name,
          toolArgs: call.args,
          summary: `${call.name} called`,
          status: "pending",
        });
      }
      continue;
    }

    rendered.push({
      role: "assistant",
      content: record.content,
      reasoning: record.reasoning ?? undefined,
      timelineItems: timeline,
    });
    timeline = [];
  }

  flushInterrupted();
  return rendered;
}

const Chat = forwardRef<ChatRef, ChatProps>(function Chat(
  {
    onToolEvent,
    getViewerState,
    beforeSend,
    sessionId,
    onResponseComplete,
    onStorageUpdate,
    availableArtifacts = [],
    customFetchOptions,
    onFeedbackSubmitted,
    selectedModel,
    activeModel,
    onModelSwitch,
  },
  ref,
) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isEmpty, setIsEmpty] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const isUserAtBottomRef = useRef(true);
  const scrollTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const editorRef = useRef<HTMLDivElement>(null);

  // AbortController for cancelling in-flight requests
  const abortControllerRef = useRef<AbortController | null>(null);
  // Request ID counter to track which request is current (prevents race conditions on interrupt)
  const requestIdRef = useRef(0);

  // Render whatever the backend already has for this session. Local state
  // exists only to show the response as it streams; the record is the backend's.
  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;

    fetch(`/api/conversation/${encodeURIComponent(sessionId)}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (cancelled || !data?.messages?.length) return;
        setMessages(hydrateMessages(data.messages));
      })
      .catch((err) => console.error("Failed to load conversation:", err));

    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  // Autocomplete state
  const [showMentionMenu, setShowMentionMenu] = useState(false);
  const [mentionQuery, setMentionQuery] = useState("");
  const [mentionStartOffset, setMentionStartOffset] = useState<number | null>(
    null,
  );
  const [mentionStartNode, setMentionStartNode] = useState<Node | null>(null);
  const [selectedMentionIndex, setSelectedMentionIndex] = useState(0);
  const mentionMenuRef = useRef<HTMLDivElement>(null);

  // Filter artifacts based on mention query
  const filteredArtifacts = useMemo(() => {
    if (!mentionQuery) return availableArtifacts;
    const query = mentionQuery.toLowerCase();
    return availableArtifacts.filter((artifact) => {
      const ref = artifact.reference.toLowerCase();
      const type = artifact.data_type.toLowerCase();
      const name = String(artifact.metadata?.name || "").toLowerCase();
      return (
        ref.includes(query) || type.includes(query) || name.includes(query)
      );
    });
  }, [availableArtifacts, mentionQuery]);

  /**
   * Create a bubble element for a reference
   */
  const createBubbleElement = useCallback(
    (reference: string): HTMLSpanElement => {
      const bubble = document.createElement("span");
      bubble.className = "ref-bubble";
      bubble.contentEditable = "false";
      bubble.dataset.ref = reference;
      bubble.innerHTML = `<span class="ref-bubble-type">${getRefTypeLabel(
        reference,
      )}</span><span class="ref-bubble-id">${getRefShortId(reference)}</span>`;
      return bubble;
    },
    [],
  );

  // Store the last known cursor position for bubble insertion
  const lastSelectionRef = useRef<{ node: Node; offset: number } | null>(null);

  /**
   * Save cursor position when selection changes within the editor
   */
  const saveCursorPosition = useCallback(() => {
    const selection = window.getSelection();
    const editor = editorRef.current;
    if (!selection || !editor || selection.rangeCount === 0) return;

    const range = selection.getRangeAt(0);
    // Only save if the selection is within our editor
    if (editor.contains(range.startContainer)) {
      lastSelectionRef.current = {
        node: range.startContainer,
        offset: range.startOffset,
      };
    }
  }, []);

  /**
   * Insert a reference bubble at the current cursor position
   */
  const insertBubble = useCallback(
    (reference: string) => {
      const editor = editorRef.current;
      if (!editor) return;

      // Create the bubble
      const bubble = createBubbleElement(reference);

      // Try to restore the last known cursor position
      let range = document.createRange();
      const savedPos = lastSelectionRef.current;

      if (savedPos && editor.contains(savedPos.node)) {
        // Restore to saved position
        try {
          range.setStart(savedPos.node, savedPos.offset);
          range.collapse(true);
        } catch {
          // If restore fails, insert at end
          range.selectNodeContents(editor);
          range.collapse(false);
        }
      } else {
        // No saved position - insert at end
        range.selectNodeContents(editor);
        range.collapse(false);
      }

      // Insert the bubble
      range.insertNode(bubble);

      // Add a space after the bubble for easier typing
      const space = document.createTextNode("\u00A0");
      range.setStartAfter(bubble);
      range.insertNode(space);

      // Move cursor after the space
      range.setStartAfter(space);
      range.collapse(true);

      // Update selection
      const selection = window.getSelection();
      if (selection) {
        selection.removeAllRanges();
        selection.addRange(range);
      }

      // Save new cursor position
      lastSelectionRef.current = {
        node: space,
        offset: 1,
      };

      // Focus the editor
      editor.focus();

      // Update empty state
      setIsEmpty(false);
    },
    [createBubbleElement],
  );

  /**
   * Set the input text content programmatically
   */
  const setInputText = useCallback((text: string) => {
    const editor = editorRef.current;
    if (!editor) return;
    editor.textContent = text;
    setIsEmpty(text.length === 0);
    editor.focus();
    // Move cursor to end
    const selection = window.getSelection();
    if (selection) {
      const range = document.createRange();
      range.selectNodeContents(editor);
      range.collapse(false);
      selection.removeAllRanges();
      selection.addRange(range);
    }
  }, []);

  // Expose imperative methods via ref
  useImperativeHandle(ref, () => ({ insertBubble, setInputText }), [
    insertBubble,
    setInputText,
  ]);

  /**
   * Check if we should show the mention menu based on cursor position
   */
  const checkForMentionTrigger = useCallback(() => {
    const selection = window.getSelection();
    const editor = editorRef.current;
    if (!selection || !editor || selection.rangeCount === 0) {
      setShowMentionMenu(false);
      return;
    }

    const range = selection.getRangeAt(0);
    if (!editor.contains(range.startContainer)) {
      setShowMentionMenu(false);
      return;
    }

    // Get text content - handle both text nodes and element nodes
    let textContent = "";
    let cursorOffset = range.startOffset;

    if (range.startContainer.nodeType === Node.TEXT_NODE) {
      textContent = range.startContainer.textContent || "";
    } else if (range.startContainer === editor) {
      // Cursor is directly in the editor element
      // Get all text content before cursor position
      const textNodes: string[] = [];
      const walkNodes = (node: Node) => {
        if (node.nodeType === Node.TEXT_NODE) {
          textNodes.push(node.textContent || "");
        } else if (node instanceof HTMLElement && !node.dataset.ref) {
          // Skip bubble elements
          for (const child of node.childNodes) {
            walkNodes(child);
          }
        }
      };

      for (let i = 0; i < cursorOffset && i < editor.childNodes.length; i++) {
        walkNodes(editor.childNodes[i]);
      }
      textContent = textNodes.join("");
      cursorOffset = textContent.length;
    } else {
      // Some other node type we can't handle
      setShowMentionMenu(false);
      return;
    }

    // Look backwards for "@" character
    let atIndex = -1;
    for (let i = cursorOffset - 1; i >= 0; i--) {
      const char = textContent[i];
      if (char === "@") {
        atIndex = i;
        break;
      }
      // Stop if we hit a space or newline (not part of mention)
      if (char === " " || char === "\n" || char === "\u00A0") {
        break;
      }
    }

    if (atIndex >= 0) {
      // Extract the query after @
      const query = textContent.slice(atIndex + 1, cursorOffset);
      setMentionQuery(query);
      setMentionStartOffset(atIndex);
      setMentionStartNode(range.startContainer);
      setShowMentionMenu(true);
      setSelectedMentionIndex(0);
    } else {
      setShowMentionMenu(false);
    }
  }, []);

  /**
   * Insert a mention bubble, replacing the @query text
   */
  const insertMention = useCallback(
    (artifact: AvailableArtifact) => {
      const editor = editorRef.current;
      if (!editor || mentionStartNode === null || mentionStartOffset === null)
        return;

      const selection = window.getSelection();
      if (!selection) return;

      // Create range to delete the @query text
      const range = document.createRange();
      try {
        range.setStart(mentionStartNode, mentionStartOffset);
        // Get current cursor position
        const currentRange = selection.getRangeAt(0);
        range.setEnd(currentRange.startContainer, currentRange.startOffset);
        range.deleteContents();
      } catch {
        // If range manipulation fails, just insert at current position
      }

      // Create and insert the bubble
      const bubble = createBubbleElement(artifact.reference);
      range.insertNode(bubble);

      // Add a space after the bubble
      const space = document.createTextNode("\u00A0");
      range.setStartAfter(bubble);
      range.insertNode(space);

      // Move cursor after the space
      range.setStartAfter(space);
      range.collapse(true);
      selection.removeAllRanges();
      selection.addRange(range);

      // Save cursor position
      lastSelectionRef.current = {
        node: space,
        offset: 1,
      };

      // Close menu and update state
      setShowMentionMenu(false);
      setMentionQuery("");
      setMentionStartNode(null);
      setMentionStartOffset(null);
      setIsEmpty(false);

      // Keep focus on editor
      editor.focus();
    },
    [createBubbleElement, mentionStartNode, mentionStartOffset],
  );

  /**
   * Handle keyboard navigation in mention menu
   */
  const handleMentionKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (!showMentionMenu || filteredArtifacts.length === 0) return false;

      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          setSelectedMentionIndex((prev) =>
            prev < filteredArtifacts.length - 1 ? prev + 1 : 0,
          );
          return true;

        case "ArrowUp":
          e.preventDefault();
          setSelectedMentionIndex((prev) =>
            prev > 0 ? prev - 1 : filteredArtifacts.length - 1,
          );
          return true;

        case "Tab":
        case "Enter":
        case " ":
          if (e.key === " " && mentionQuery.length === 0) {
            // Don't capture space if no query yet (user might want to type space after @)
            return false;
          }
          e.preventDefault();
          insertMention(filteredArtifacts[selectedMentionIndex]);
          return true;

        case "Escape":
          e.preventDefault();
          setShowMentionMenu(false);
          return true;

        default:
          return false;
      }
    },
    [
      showMentionMenu,
      filteredArtifacts,
      selectedMentionIndex,
      insertMention,
      mentionQuery,
    ],
  );

  // Reset selected index when filtered results change
  useEffect(() => {
    setSelectedMentionIndex(0);
  }, [filteredArtifacts.length]);

  // Scroll selected item into view
  useEffect(() => {
    if (showMentionMenu && mentionMenuRef.current) {
      const selectedItem = mentionMenuRef.current.querySelector(
        `[data-index="${selectedMentionIndex}"]`,
      );
      if (selectedItem) {
        selectedItem.scrollIntoView({ block: "nearest" });
      }
    }
  }, [selectedMentionIndex, showMentionMenu]);

  /**
   * Serialize the editor content to a string for sending
   */
  const getEditorContent = useCallback((): string => {
    const editor = editorRef.current;
    if (!editor) return "";

    let result = "";
    const walk = (node: Node) => {
      if (node.nodeType === Node.TEXT_NODE) {
        result += node.textContent;
      } else if (node instanceof HTMLElement) {
        if (node.dataset.ref) {
          // This is a bubble - serialize as @ref:xxx
          result += `@ref:${node.dataset.ref}`;
        } else if (node.tagName === "BR") {
          result += "\n";
        } else {
          // Recurse into children
          for (const child of node.childNodes) {
            walk(child);
          }
        }
      }
    };

    for (const child of editor.childNodes) {
      walk(child);
    }

    // Clean up: replace non-breaking spaces with regular spaces
    return result.replace(/\u00A0/g, " ").trim();
  }, []);

  /**
   * Clear the editor content
   */
  const clearEditor = useCallback(() => {
    const editor = editorRef.current;
    if (editor) {
      editor.innerHTML = "";
      setIsEmpty(true);
    }
  }, []);

  /**
   * Handle input changes to track empty state and check for mentions
   */
  const handleEditorInput = useCallback(() => {
    const content = getEditorContent();
    setIsEmpty(content.length === 0);
    // Check if we should show mention menu (slight delay to ensure selection is updated)
    requestAnimationFrame(() => {
      checkForMentionTrigger();
    });
  }, [getEditorContent, checkForMentionTrigger]);

  // Check if user is scrolled near the bottom
  const checkIfAtBottom = useCallback(() => {
    const container = messagesContainerRef.current;
    if (!container) return true;
    const threshold = 100; // pixels from bottom
    return (
      container.scrollHeight - container.scrollTop - container.clientHeight <
      threshold
    );
  }, []);

  // Smooth scroll to bottom, but only if user was already at bottom
  const scrollToBottom = useCallback((force = false) => {
    if (!force && !isUserAtBottomRef.current) return;

    // Cancel any pending scroll
    if (scrollTimeoutRef.current) {
      clearTimeout(scrollTimeoutRef.current);
    }

    // Debounce scroll to prevent too many calls during streaming
    scrollTimeoutRef.current = setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: "instant" });
    }, 16); // roughly one frame
  }, []);

  // Track scroll position
  const handleScroll = useCallback(() => {
    isUserAtBottomRef.current = checkIfAtBottom();
  }, [checkIfAtBottom]);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
      }
    };
  }, []);

  // Handle feedback submission (thumbs up/down)
  const handleFeedback = useCallback(
    async (span_id: string, label: "thumbs_up" | "thumbs_down") => {
      try {
        const response = await fetch("/api/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            span_id,
            label,
            score: label === "thumbs_up" ? 1 : 0,
            metadata: {
              session_id: sessionId,
              // Version number for filtering exported data
              version_number: import.meta.env.VERSION_NUMBER,
              // Include training metadata if in training mode
              ...(customFetchOptions?.training_query_id && {
                training_query_id: customFetchOptions.training_query_id,
              }),
              ...(customFetchOptions?.prompt_mode && {
                prompt_mode: customFetchOptions.prompt_mode,
              }),
            },
          }),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Update the message to show feedback was submitted
        setMessages((prevMessages) =>
          prevMessages.map((msg) =>
            msg.span_id === span_id
              ? { ...msg, feedbackSubmitted: label }
              : msg,
          ),
        );

        // Notify parent that feedback was submitted (for training mode status refresh)
        onFeedbackSubmitted?.();
      } catch (error) {
        console.error("Failed to submit feedback:", error);
        // Could show an error toast here
      }
    },
    [sessionId, customFetchOptions, onFeedbackSubmitted],
  );

  /**
   * Stop the current generation without losing state
   */
  const handleStop = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    // isLoading will be set to false in the sendMessage catch/finally block
  }, []);

  const sendMessage = async (e: FormEvent, forceInterrupt = false) => {
    e.preventDefault();
    const messageContent = getEditorContent();
    if (!messageContent) return;

    // If already loading and not forcing interrupt, ignore
    if (isLoading && !forceInterrupt) return;

    // If interrupting, mark current streaming message as interrupted BEFORE aborting
    // This must happen before we add new messages to ensure we update the right one
    let messagesAfterInterrupt = messages;
    if (isLoading && forceInterrupt) {
      // Mark the current streaming assistant message as interrupted
      messagesAfterInterrupt = messages.map((msg) =>
        msg.role === "assistant" && msg.isStreaming
          ? { ...msg, isStreaming: false, isInterrupted: true }
          : msg,
      );
      setMessages(messagesAfterInterrupt);

      // Abort the previous request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
    }

    // Increment request ID to track this request (prevents race conditions)
    requestIdRef.current += 1;
    const thisRequestId = requestIdRef.current;

    // Detect model switch: if the selected model differs from last-used, reset state
    let effectiveSessionId = sessionId;
    let messagesForRequest = messagesAfterInterrupt;
    if (
      selectedModel &&
      activeModel &&
      selectedModel !== activeModel &&
      onModelSwitch
    ) {
      try {
        effectiveSessionId = await onModelSwitch(selectedModel);
      } catch (err) {
        console.error("Failed to start a session for the new model:", err);
        return;
      }
      messagesForRequest = [];
      setMessages([]);
    }

    const userMessage: Message = { role: "user", content: messageContent };
    const updatedMessages = [...messagesForRequest, userMessage];
    setMessages(updatedMessages);
    clearEditor();
    setIsLoading(true);

    // Force scroll to bottom when sending a message
    isUserAtBottomRef.current = true;

    // Add placeholder for assistant response (marked as streaming)
    setMessages([
      ...updatedMessages,
      { role: "assistant", content: "", isStreaming: true },
    ]);

    // Create new AbortController for this request
    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    try {
      // Let the viewer store the user's hand-picked residues first, so they
      // are in the session's selection state when the backend builds context.
      if (beforeSend) {
        try {
          await beforeSend(effectiveSessionId);
        } catch (err) {
          console.error("beforeSend failed; sending anyway:", err);
        }
      }

      // Get current viewer state to send with the request
      const viewerState = getViewerState ? getViewerState() : null;
      console.log("[Chat] Sending viewer_state with request:", viewerState);

      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          // Only the new message. The backend owns the transcript, including
          // the tool calls and reasoning this component never had.
          message: messageContent,
          viewer_state: viewerState,
          session_id: effectiveSessionId,
          ...(selectedModel && { model_id: selectedModel }),
          // Training mode parameters (if provided)
          ...(customFetchOptions?.prompt_mode && {
            prompt_mode: customFetchOptions.prompt_mode,
          }),
          ...(customFetchOptions?.training_query_id && {
            training_query_id: customFetchOptions.training_query_id,
          }),
          ...(customFetchOptions?.expected_tool && {
            expected_tool: customFetchOptions.expected_tool,
          }),
          ...(customFetchOptions?.oracle_prompt && {
            oracle_prompt: customFetchOptions.oracle_prompt,
          }),
        }),
        signal: abortController.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error("No response body");
      }

      let assistantContent = "";
      let assistantReasoning = "";
      let rateLimitMessage = "";
      let timelineItems: TimelineItem[] = [];
      let buffer = "";
      let currentSpanId = "";

      const updateAssistantMessage = (isStreaming: boolean) => {
        setMessages([
          ...updatedMessages,
          {
            role: "assistant",
            content: assistantContent,
            reasoning: assistantReasoning,
            timelineItems: [...timelineItems],
            isStreaming,
            span_id: currentSpanId || undefined,
            rateLimitMessage: rateLimitMessage || undefined,
          },
        ]);
      };

      // Helper to process a parsed SSE event
      const processEvent = (parsed: Record<string, unknown>) => {
        // Handle span_id event (for feedback annotation)
        if (parsed.span_id) {
          currentSpanId = parsed.span_id as string;
          updateAssistantMessage(true);
        }

        if (parsed.rate_limit_status !== undefined) {
          if (parsed.rate_limit_status === null) {
            rateLimitMessage = "";
          } else {
            const status = parsed.rate_limit_status as {
              message: string;
              attempt: number;
            };
            rateLimitMessage = status.message;
          }
          updateAssistantMessage(true);
        }

        if (parsed.tool_event && onToolEvent) {
          onToolEvent(parsed.tool_event as ToolEvent);
        }

        // Handle storage_update events (real-time storage strip updates)
        if (parsed.storage_update && onStorageUpdate) {
          const update = parsed.storage_update as {
            artifacts: StorageUpdateArtifact[];
          };
          onStorageUpdate(update.artifacts);
        }

        // Handle tool_call_start events
        if (parsed.tool_call_start) {
          const toolStart = parsed.tool_call_start as {
            id: string;
            name: string;
            args: Record<string, unknown>;
            summary: string;
          };
          const iteration = (parsed.iteration as number) || 1;

          timelineItems.push({
            id: toolStart.id,
            type: "tool_call",
            iteration,
            timestamp: Date.now(),
            toolName: toolStart.name,
            toolArgs: toolStart.args,
            summary: toolStart.summary,
            status: "pending",
          });
          updateAssistantMessage(true);
        }

        // Handle tool_call_end events
        if (parsed.tool_call_end) {
          const toolEnd = parsed.tool_call_end as {
            id: string;
            name: string;
            result_ref: string | null;
            success: boolean;
            output_preview?: string;
          };

          const itemIndex = timelineItems.findIndex(
            (item) => item.id === toolEnd.id,
          );
          if (itemIndex !== -1) {
            timelineItems[itemIndex] = {
              ...timelineItems[itemIndex],
              resultRef: toolEnd.result_ref,
              toolOutput: toolEnd.output_preview || undefined,
              status: toolEnd.success ? "complete" : "error",
            };
            updateAssistantMessage(true);
          }
        }

        // Handle tool_call_update events (deferred output resolution)
        if (parsed.tool_call_update) {
          const toolUpdate = parsed.tool_call_update as {
            id: string;
            output_preview?: string;
          };

          const itemIndex = timelineItems.findIndex(
            (item) => item.id === toolUpdate.id,
          );
          if (itemIndex !== -1) {
            timelineItems[itemIndex] = {
              ...timelineItems[itemIndex],
              toolOutput: toolUpdate.output_preview || undefined,
            };
            updateAssistantMessage(true);
          }
        }

        // Handle reasoning chunks (model thinking)
        if (parsed.reasoning) {
          const iteration = (parsed.iteration as number) ?? 1;
          const reasoningText = parsed.reasoning as string;
          assistantReasoning += reasoningText;

          // Append to the last item if it's already reasoning; otherwise
          // start a new block. This ensures reasoning from workflow
          // subgraphs is interleaved chronologically with tool calls
          // rather than lumped into a single item.
          const lastItem = timelineItems[timelineItems.length - 1];
          if (lastItem && lastItem.type === "reasoning") {
            timelineItems[timelineItems.length - 1] = {
              ...lastItem,
              content: (lastItem.content || "") + reasoningText,
            };
          } else {
            timelineItems.push({
              id: `reasoning-${iteration}-${Date.now()}`,
              type: "reasoning",
              iteration,
              timestamp: Date.now(),
              content: reasoningText,
            });
          }
          updateAssistantMessage(true);
        }

        if (parsed.content) {
          rateLimitMessage = "";
          assistantContent += parsed.content as string;
          updateAssistantMessage(true);
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Accumulate chunk data - large messages may span multiple chunks
        buffer += decoder.decode(value, { stream: true });

        // SSE format: each message is "data: <json>\n"
        // Split on newlines and keep incomplete final line in buffer
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.slice(6);
            if (data.trim() === "[DONE]") continue;
            try {
              const parsed = JSON.parse(data);
              processEvent(parsed);
            } catch {
              // Skip incomplete or invalid JSON (can happen with chunked delivery)
            }
          }
        }
      }

      // Process any remaining complete message in buffer
      if (buffer.startsWith("data: ")) {
        const data = buffer.slice(6);
        if (data.trim() !== "[DONE]") {
          try {
            const parsed = JSON.parse(data);
            processEvent(parsed);
          } catch {
            // Ignore incomplete data at stream end
          }
        }
      }

      // Mark streaming as complete - this will trigger full markdown rendering
      setMessages([
        ...updatedMessages,
        {
          role: "assistant",
          content: assistantContent,
          reasoning: assistantReasoning || undefined,
          timelineItems: timelineItems.length > 0 ? timelineItems : undefined,
          isStreaming: false,
          span_id: currentSpanId || undefined,
        },
      ]);

      // Notify parent that response is complete (artifacts are now committed)
      onResponseComplete?.();
    } catch (error) {
      // Handle abort gracefully - don't show error message
      if (error instanceof Error && error.name === "AbortError") {
        console.log("[Chat] Request aborted by user");
        // Only update messages if this is still the current request
        // (If not, we already marked the message as interrupted in sendMessage before starting the new request)
        if (thisRequestId === requestIdRef.current) {
          setMessages((prevMessages) => {
            const lastMessage = prevMessages[prevMessages.length - 1];
            if (
              lastMessage &&
              lastMessage.role === "assistant" &&
              lastMessage.isStreaming
            ) {
              return [
                ...prevMessages.slice(0, -1),
                {
                  ...lastMessage,
                  isStreaming: false,
                  isInterrupted: true,
                  content: lastMessage.content || "",
                },
              ];
            }
            return prevMessages;
          });
        }
      } else {
        console.error("Error sending message:", error);
        setMessages([
          ...updatedMessages,
          {
            role: "assistant",
            content: "Sorry, there was an error processing your request.",
            isStreaming: false,
          },
        ]);
      }
    } finally {
      // Only update loading state if this is still the current request
      // (prevents race condition when interrupting to send a new message)
      if (thisRequestId === requestIdRef.current) {
        setIsLoading(false);
        abortControllerRef.current = null;
      }
    }
  };

  return (
    <div className="chat-container">
      <div
        className="messages"
        ref={messagesContainerRef}
        onScroll={handleScroll}
      >
        {messages.length === 0 ? (
          <div className="welcome-message">
            <h2>Welcome to aevom</h2>
            <p>
              I'm an AI assistant specialized in molecular design. Ask me about
              molecules, protein-ligand interactions, or let me help you design
              and analyze molecular structures.
            </p>
            <div className="suggestions">
              <button
                onClick={() => {
                  if (editorRef.current) {
                    editorRef.current.textContent =
                      "Analyze the the crystal structure of Gefitinib-resistant double mutant Epidermal Growth Factor Receptor kinase domain. Based on your analysis of the structure, identify key stabilizing interactions, show me the key residues involved, and explain the basis of your conclusions.";
                    setIsEmpty(false);
                  }
                }}
              >
                Figure out what explains the activity of Gefitinib.
              </button>
              <button
                onClick={() => {
                  if (editorRef.current) {
                    editorRef.current.textContent =
                      "Predict the 3d structure of the PGK1 docked to imatinib. Analyze the protein-ligand interactions. Identify and explain how the ligand is stabilized in the pocket.";
                    setIsEmpty(false);
                  }
                }}
              >
                Analyze the interations between PGK1 and Imatinib.
              </button>
              <button
                onClick={() => {
                  if (editorRef.current) {
                    editorRef.current.textContent =
                      "What is the 3D structure of tovorafenib? What functional groups play a role in its activity?";
                    setIsEmpty(false);
                  }
                }}
              >
                What is the 3D structure of tovorafenib?
              </button>
              <button
                onClick={() => {
                  if (editorRef.current) {
                    editorRef.current.textContent =
                      "Show me how GLP-1 drugs work. Fetch a structure of a GLP-1 analog bound to its receptor. Highlight the residues of the binding region and show it to me.";
                    setIsEmpty(false);
                  }
                }}
              >
                Show me how GLP-1 drugs work.
              </button>
              <button
                onClick={() => {
                  if (editorRef.current) {
                    editorRef.current.textContent =
                      "Lead optimization pass on CDK2: fetch PDB 1FVV with the crystallographic ligand removed (that also gives you the pocket center), run one quick probe-grid scan of the ATP pocket to characterize it, then generate a single new de novo candidate with GenMol sized for that pocket. Dock that GenMol candidate directly into the pocket with Vina using the SMILES GenMol returns, and show me the final docked complex.";
                    setIsEmpty(false);
                  }
                }}
              >
                Design and dock a new candidate against CDK2.
              </button>
            </div>
          </div>
        ) : (
          messages.map((message, index) => (
            <MessageBubble
              key={index}
              message={message}
              onFeedback={handleFeedback}
            />
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      <form className="input-area" onSubmit={sendMessage}>
        <div className="input-wrapper">
          <div
            ref={editorRef}
            className="chat-input-editor"
            contentEditable
            onInput={handleEditorInput}
            onKeyUp={saveCursorPosition}
            onClick={saveCursorPosition}
            onFocus={saveCursorPosition}
            onKeyDown={(e) => {
              // Handle mention menu navigation first
              if (handleMentionKeyDown(e)) {
                return;
              }
              // Handle send on Enter (interrupt and send if loading)
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                if (isLoading && !isEmpty) {
                  // Interrupt current request and send new message
                  sendMessage(e as unknown as FormEvent, true);
                } else {
                  sendMessage(e as unknown as FormEvent);
                }
              }
            }}
            onBlur={() => {
              // Delay hiding menu to allow click on menu items
              setTimeout(() => setShowMentionMenu(false), 150);
            }}
            data-placeholder="Ask about molecules, structures, or molecular design... (type @ to reference artifacts)"
            suppressContentEditableWarning
          />

          {/* Mention autocomplete menu */}
          {showMentionMenu && filteredArtifacts.length > 0 && (
            <div ref={mentionMenuRef} className="mention-menu">
              {filteredArtifacts.map((artifact, index) => (
                <div
                  key={artifact.reference}
                  data-index={index}
                  className={`mention-item ${
                    index === selectedMentionIndex ? "selected" : ""
                  }`}
                  onMouseDown={(e) => {
                    e.preventDefault(); // Prevent blur
                    insertMention(artifact);
                  }}
                  onMouseEnter={() => setSelectedMentionIndex(index)}
                >
                  <span className="mention-item-icon">
                    {artifact.data_type === "pdb"
                      ? "🔬"
                      : artifact.data_type === "sequence"
                        ? "🧬"
                        : artifact.data_type === "sdf" ||
                            artifact.data_type === "mol2"
                          ? "⬡"
                          : artifact.data_type === "safe"
                            ? "🧩"
                            : artifact.data_type === "selection"
                              ? "✦"
                              : artifact.data_type === "msa"
                                ? "▦"
                                : "📦"}
                  </span>
                  <span className="mention-item-type">
                    {artifact.data_type.toUpperCase()}
                  </span>
                  <span className="mention-item-ref">{artifact.reference}</span>
                  {artifact.metadata?.name != null && (
                    <span className="mention-item-name">
                      {String(artifact.metadata.name).slice(0, 20)}
                    </span>
                  )}
                </div>
              ))}
              <div className="mention-hint">
                ↑↓ navigate · Tab/Enter select · Esc close
              </div>
            </div>
          )}

          {/* No matches message */}
          {showMentionMenu &&
            filteredArtifacts.length === 0 &&
            availableArtifacts.length > 0 && (
              <div className="mention-menu">
                <div className="mention-empty">No matching artifacts</div>
              </div>
            )}

          {/* No artifacts available message */}
          {showMentionMenu && availableArtifacts.length === 0 && (
            <div className="mention-menu">
              <div className="mention-empty">No artifacts in storage yet</div>
            </div>
          )}
        </div>
        {/* Stop button - only visible while loading */}
        {isLoading && (
          <button
            type="button"
            className="stop-btn"
            onClick={handleStop}
            title="Stop generating"
          >
            <StopIcon />
          </button>
        )}
        {/* Submit button - changes behavior when loading with input */}
        <button
          type="submit"
          disabled={isEmpty && !isLoading}
          onClick={(e) => {
            if (isLoading && !isEmpty) {
              e.preventDefault();
              sendMessage(e as unknown as FormEvent, true);
            }
          }}
          title={
            isLoading && !isEmpty
              ? "Interrupt and send message"
              : isLoading
                ? "Generating..."
                : "Send message"
          }
          className={isLoading && !isEmpty ? "interrupt-mode" : ""}
        >
          {isLoading && isEmpty ? (
            <span className="loading-spinner"></span>
          ) : (
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="currentColor"
            >
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
            </svg>
          )}
        </button>
      </form>
    </div>
  );
});

export default Chat;
