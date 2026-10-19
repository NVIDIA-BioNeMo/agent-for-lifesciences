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

/**
 * ToolCallTimeline - A vertical timeline showing tool calls and reasoning steps.
 *
 * Every step carries one of two icons - an alembic for reasoning, a wrench for
 * tool calls - alongside a brief summary.
 */

import { useRef, useEffect } from "react";
import { TimelineItem } from "./Chat";

const EXPANDED_NCHARS_INPUT = 500;
const EXPANDED_NCHARS_OUTPUT = 200;

const REASONING_ICON = "⚗️";

function WrenchIcon() {
  return (
    <svg
      className="timeline-wrench"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />
    </svg>
  );
}

interface ToolCallTimelineProps {
  items: TimelineItem[];
  isStreaming?: boolean;
  isExpanded?: boolean;
  isInterrupted?: boolean;
}

/**
 * Get the last N lines of text for the streaming tail view.
 */
function getLastLines(text: string, n: number): string {
  const lines = text.split("\n");
  return lines.slice(-n).join("\n");
}

/**
 * Streaming reasoning preview: shows last 3 lines, auto-scrolls as new text arrives.
 */
function StreamingReasoningPreview({ content }: { content: string }) {
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (ref.current) {
      ref.current.scrollTop = ref.current.scrollHeight;
    }
  }, [content]);

  return (
    <span ref={ref} className="timeline-reasoning-preview streaming">
      {getLastLines(content, 3)}
    </span>
  );
}

/**
 * Collapsed timeline view - shows just icons and brief summaries
 */
function CollapsedTimeline({
  items,
  isStreaming,
  isInterrupted,
}: {
  items: TimelineItem[];
  isStreaming?: boolean;
  isInterrupted?: boolean;
}) {
  if (items.length === 0) return null;

  return (
    <div className="timeline-collapsed">
      {items.map((item, index) => {
        const isActiveReasoning =
          item.type === "reasoning" &&
          isStreaming === true &&
          index === items.length - 1;

        return (
          <div
            key={item.id}
            className={`timeline-item ${item.type} ${
              item.status === "pending" && !isInterrupted ? "pending" : ""
            }`}
          >
            {/* Connecting line */}
            {index < items.length - 1 && (
              <div className="timeline-connector" />
            )}

            {/* Icon */}
            <div className="timeline-icon">
              {item.type === "tool_call" ? <WrenchIcon /> : REASONING_ICON}
            </div>

            {/* Summary text */}
            <div className="timeline-summary">
              {item.type === "tool_call" ? (
                <>
                  <span className="timeline-tool-name">{item.summary}</span>
                  {item.status === "pending" && !isInterrupted && (
                    <span className="timeline-pending-indicator" />
                  )}
                  {item.status === "error" && (
                    <span className="timeline-error-indicator">✗</span>
                  )}
                </>
              ) : isActiveReasoning ? (
                <StreamingReasoningPreview content={item.content || ""} />
              ) : (
                <span className="timeline-reasoning-preview">
                  {(item.content || "").split("\n")[0]}
                </span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

/**
 * Expanded timeline view - shows full details interleaved
 */
function ExpandedTimeline({
  items,
  isInterrupted,
}: {
  items: TimelineItem[];
  isInterrupted?: boolean;
}) {
  if (items.length === 0) return null;

  return (
    <div className="timeline-expanded">
      {items.map((item, index) => (
        <div
          key={item.id}
          className={`timeline-item-expanded ${item.type} ${
            item.status === "pending" && !isInterrupted ? "pending" : ""
          }`}
        >
          {/* Connecting line */}
          {index < items.length - 1 && (
            <div className="timeline-connector-expanded" />
          )}

          {/* Icon */}
          <div className="timeline-icon-expanded">
            {item.type === "tool_call" ? <WrenchIcon /> : REASONING_ICON}
          </div>

          {/* Content */}
          <div className="timeline-content-expanded">
            {item.type === "tool_call" ? (
              <ToolCallDetail item={item} isInterrupted={isInterrupted} />
            ) : (
              <ReasoningDetail item={item} />
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

/**
 * Detailed view of a tool call
 */
function ToolCallDetail({
  item,
  isInterrupted,
}: {
  item: TimelineItem;
  isInterrupted?: boolean;
}) {
  const formatToolCall = () => {
    const name = item.toolName || "Unknown tool";
    const args = item.toolArgs || {};
    const argsStr = Object.entries(args)
      .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
      .join(", ");
    const full = `${name}(${argsStr})`;
    if (full.length > EXPANDED_NCHARS_INPUT) {
      return full.slice(0, EXPANDED_NCHARS_INPUT) + "...";
    }
    return full;
  };

  return (
    <div className="tool-call-detail">
      <div className="tool-call-header">
        <span className="tool-call-name">{item.toolName}</span>
        {item.status === "pending" && !isInterrupted && (
          <span className="tool-call-status pending">Running...</span>
        )}
        {item.status === "complete" && (
          <span className="tool-call-status complete">✓</span>
        )}
        {item.status === "error" && (
          <span className="tool-call-status error">Failed</span>
        )}
      </div>
      <div className="tool-call-description">{formatToolCall()}</div>
      {item.toolOutput && (
        <div className="tool-call-output">
          {item.toolOutput.length > EXPANDED_NCHARS_OUTPUT
            ? item.toolOutput.slice(0, EXPANDED_NCHARS_OUTPUT) + "..."
            : item.toolOutput}
        </div>
      )}
    </div>
  );
}

/**
 * Detailed view of reasoning content
 */
function ReasoningDetail({ item }: { item: TimelineItem }) {
  return (
    <div className="reasoning-detail">
      <pre className="reasoning-text">{item.content}</pre>
    </div>
  );
}

/**
 * Truncate reasoning text for preview
 */
function truncateReasoning(text: string, maxLength: number): string {
  // Get first line or first maxLength chars
  const firstLine = text.split("\n")[0];
  const truncated = firstLine.slice(0, maxLength);

  if (firstLine.length > maxLength || text.includes("\n")) {
    return truncated.trim() + "...";
  }
  return truncated;
}

/**
 * Get brief summary for collapsed view
 */
export function getBriefSummary(items: TimelineItem[]): string {
  if (items.length === 0) return "";

  // Get the most recent item
  const lastItem = items[items.length - 1];

  if (lastItem.type === "tool_call") {
    return lastItem.summary || `${lastItem.toolName} called`;
  } else {
    return truncateReasoning(lastItem.content || "", 40);
  }
}

/**
 * Main timeline component
 */
export function ToolCallTimeline({
  items,
  isStreaming,
  isExpanded,
  isInterrupted,
}: ToolCallTimelineProps) {
  if (items.length === 0) return null;

  return isExpanded ? (
    <ExpandedTimeline items={items} isInterrupted={isInterrupted} />
  ) : (
    <CollapsedTimeline
      items={items}
      isStreaming={isStreaming}
      isInterrupted={isInterrupted}
    />
  );
}

export default ToolCallTimeline;
