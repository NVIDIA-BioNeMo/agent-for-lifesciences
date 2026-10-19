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

import { useState, useEffect, useRef, useCallback, FormEvent } from "react";

interface FeedbackModalProps {
  isOpen: boolean;
  onClose: () => void;
}

/** Mirrors MIN_MESSAGE_CHARS / MAX_MESSAGE_CHARS in api/feedback/config.py. */
const MIN_CHARS = 30;
const MAX_CHARS = 5000;

/** How long the "thanks" state stays up before the modal closes itself. */
const SUCCESS_DISMISS_MS = 1400;

const CloseIcon = () => (
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
    <line x1="18" y1="6" x2="6" y2="18"></line>
    <line x1="6" y1="6" x2="18" y2="18"></line>
  </svg>
);

const FeedbackIcon = () => (
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
    <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"></path>
  </svg>
);

const ExpandIcon = () => (
  <svg
    width="18"
    height="18"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <polyline points="15 3 21 3 21 9"></polyline>
    <polyline points="9 21 3 21 3 15"></polyline>
    <line x1="21" y1="3" x2="14" y2="10"></line>
    <line x1="3" y1="21" x2="10" y2="14"></line>
  </svg>
);

const CollapseIcon = () => (
  <svg
    width="18"
    height="18"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <polyline points="4 14 10 14 10 20"></polyline>
    <polyline points="20 10 14 10 14 4"></polyline>
    <line x1="14" y1="10" x2="21" y2="3"></line>
    <line x1="3" y1="21" x2="10" y2="14"></line>
  </svg>
);

/**
 * Free-text feedback, relayed by the backend to a configured inbox.
 *
 * Rendered only when `/api/info` reports `features.feedback_email`; the endpoint
 * behind it answers 404 when the same flag is off, so the gate holds even if
 * this component is somehow mounted.
 */
export function FeedbackModal({ isOpen, onClose }: FeedbackModalProps) {
  const [message, setMessage] = useState("");
  const [expanded, setExpanded] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sent, setSent] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const count = message.length;
  const overLimit = count > MAX_CHARS;
  const tooShort = message.trim().length < MIN_CHARS;
  const canSend = !submitting && !overLimit && !tooShort;

  // Every open starts from a clean slate: a half-typed message left over from a
  // cancelled attempt is more surprising than helpful.
  useEffect(() => {
    if (!isOpen) return;
    setMessage("");
    setExpanded(false);
    setSubmitting(false);
    setError(null);
    setSent(false);
    // Focus after the open animation has committed the node to the DOM.
    const id = window.setTimeout(() => textareaRef.current?.focus(), 0);
    return () => window.clearTimeout(id);
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !submitting) onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isOpen, submitting, onClose]);

  const handleSubmit = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      if (!canSend) return;

      setSubmitting(true);
      setError(null);
      try {
        const response = await fetch("/api/feedback/email", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: message.trim() }),
        });
        if (!response.ok) {
          const body = await response.json().catch(() => null);
          throw new Error(body?.detail || `Request failed (${response.status})`);
        }
        setSent(true);
        window.setTimeout(onClose, SUCCESS_DISMISS_MS);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Could not send feedback");
        setSubmitting(false);
      }
    },
    [canSend, message, onClose],
  );

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget && !submitting) onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="modal-backdrop" onClick={handleBackdropClick}>
      <div
        className={`modal-container feedback-modal ${expanded ? "expanded" : ""}`}
      >
        <div className="modal-header">
          <div className="modal-title">
            <FeedbackIcon />
            <h2>Feedback</h2>
          </div>
          <div className="feedback-header-actions">
            <button
              type="button"
              className="modal-close-btn feedback-expand-btn"
              onClick={() => setExpanded((v) => !v)}
              title={expanded ? "Shrink" : "Expand"}
              aria-label={expanded ? "Shrink the editor" : "Expand the editor"}
            >
              {expanded ? <CollapseIcon /> : <ExpandIcon />}
            </button>
            <button
              type="button"
              className="modal-close-btn"
              onClick={onClose}
              disabled={submitting}
              title="Close"
            >
              <CloseIcon />
            </button>
          </div>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="modal-body">
            {sent ? (
              <div className="feedback-sent">
                <span className="feedback-sent-mark">✓</span>
                <span>Thanks — your feedback has been sent.</span>
              </div>
            ) : (
              <>
                <label className="feedback-label" htmlFor="feedback-message">
                  Tell us what is working, what is not, or what is missing.
                </label>
                <textarea
                  id="feedback-message"
                  ref={textareaRef}
                  className="feedback-textarea"
                  rows={expanded ? 20 : 6}
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  disabled={submitting}
                  placeholder="Your feedback..."
                  // Deliberately no maxLength: the counter turning red and the
                  // Send button disabling are both states you can only reach by
                  // typing past the cap.
                />
                {error && (
                  <div className="modal-error">
                    <span className="error-icon">!</span>
                    <span>{error}</span>
                  </div>
                )}
              </>
            )}
          </div>

          {!sent && (
            <div className="modal-footer feedback-footer">
              <div className="feedback-status">
                <span
                  className={`feedback-counter ${overLimit ? "over-limit" : ""}`}
                >
                  {count}/{MAX_CHARS} characters
                </span>
                {tooShort && !overLimit && (
                  <span className="feedback-hint">
                    Minimum {MIN_CHARS} characters
                  </span>
                )}
              </div>
              <div className="feedback-actions">
                <button
                  type="button"
                  className="modal-btn secondary"
                  onClick={onClose}
                  disabled={submitting}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="modal-btn primary"
                  disabled={!canSend}
                >
                  {submitting ? "Sending..." : "Send"}
                </button>
              </div>
            </div>
          )}
        </form>
      </div>
    </div>
  );
}

export default FeedbackModal;
