#!/usr/bin/env python3
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

"""Aevom tool-use campaign harness.

Sends a chat task to the local backend, waits for the SSE stream to close, then
pulls that request's Phoenix trace and extracts a structured run record: tools
called (with success / error / args preview), iteration count, hit_limit, final
response, and the resolved LLM model. Records are appended to a JSONL log for
before/after comparison, and a coverage matrix tracks which single tools and
which ordered producer->consumer tool-pairs have been exercised.

This is instrumentation only -- it never mutates product code or storage.

Usage:
  # single task
  python3 scripts/campaign_run.py --task-id molinfo-single \
      --message "What is the molecular weight of aspirin? Use your tools."

  # a suite of tasks (JSON list of {id, message, [session]}); tasks with the
  # same session label share one backend session
  python3 scripts/campaign_run.py --suite scripts/campaign_tasks.json

  # just re-summarise the most recent trace for a message (no new send)
  python3 scripts/campaign_run.py --task-id x --message "..." --trace-only

Outputs (under --outdir, default scratch dir):
  runs.jsonl        one JSON record per run
  coverage.json     {tools: {name: count}, pairs: {"A->B": count}}
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phoenix_query import graphql_query, AEVOM_PROJECT_ID  # noqa: E402

# Reuse the eval driver's browser stand-in so viewer-dependent tools
# (structure_query) actually resolve instead of timing out. Without a molstar
# attached the backend blocks on /api/selection for its full timeout and the
# model is forced down a hand-rolled fallback -- an unrealistic scenario that
# production (browser) and the eval harness (this same bridge) never hit.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "evals"))
try:
    from aevom_eval.driver.molstar_bridge import MolstarBridge  # noqa: E402
except Exception:  # pragma: no cover - bridge optional
    MolstarBridge = None

BACKEND = os.environ.get("AEVOM_BACKEND_URL", "http://localhost:5180")


def resolve_molstar_base() -> str | None:
    """URL of the molstar-headless MolQL engine (the browser stand-in).

    Prefers $MOLSTAR_BASE; otherwise reads the container's IP off the docker
    network (the port is not published to the host). Returns None if it can't
    be found, in which case the harness runs viewer-less (legacy behaviour).
    """
    env = os.environ.get("MOLSTAR_BASE")
    if env:
        return env.rstrip("/")
    try:
        out = subprocess.run(
            ["docker", "inspect", "aevom-molstar-headless-1", "--format",
             "{{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}"],
            capture_output=True, text=True, timeout=10,
        )
        ips = out.stdout.split()
        if ips:
            return f"http://{ips[0]}:8020"
    except Exception:
        pass
    return None
DEFAULT_OUTDIR = os.environ.get(
    "CAMPAIGN_OUTDIR",
    "/home/ubuntu/.claude-tmp/claude-1000/-workspaces-aevom/"
    "602edab9-dd97-4254-8111-c033c38993fa/scratchpad/campaign",
)


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #
# The backend only accepts sessions it minted, bound to an owner token. A task's
# "session" is therefore a label: tasks sharing a label share one minted
# session, and one token owns every session this run creates.
_owner_token: str | None = None
_sessions: dict[str, str] = {}


def open_session(label: str) -> str:
    """The backend session for ``label``, creating it on first use."""
    global _owner_token
    if label in _sessions:
        return _sessions[label]
    headers = {"Content-Type": "application/json"}
    if _owner_token:
        headers["Authorization"] = f"Bearer {_owner_token}"
    req = urllib.request.Request(
        f"{BACKEND}/api/session/create", data=b"{}", headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    _owner_token = body["owner_token"]
    _sessions[label] = body["session_id"]
    return body["session_id"]


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
def send_chat(message: str, session_id: str, max_tokens: int = 4096,
              timeout: float = 600.0, molstar_base: str | None = None) -> dict:
    """POST /api/chat and consume the SSE stream to completion.

    When ``molstar_base`` is set and the MolstarBridge is importable, each SSE
    ``tool_event`` is dispatched to a bridge that plays browser -- loading the
    structure from visualize_pdb events and answering structure_query selections
    against molstar-headless, exactly as the eval driver and a real browser do.
    """
    bridge = None
    if molstar_base and MolstarBridge is not None:
        bridge = MolstarBridge(BACKEND, molstar_base, _owner_token)

    payload = json.dumps(
        {"message": message, "session_id": session_id, "max_tokens": max_tokens}
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{BACKEND}/api/chat",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_owner_token}",
        },
    )
    content_parts, reasoning_len, usage, n_events = [], 0, None, 0
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                n_events += 1
                try:
                    ev = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                # Answer viewer-dependent tools synchronously: the backend is
                # blocked on /api/selection and the SSE stream is our only feed.
                if bridge is not None and ev.get("tool_event"):
                    try:
                        bridge.note_tool_event(ev["tool_event"], session_id)
                    except Exception:
                        pass
                if "content" in ev:
                    content_parts.append(ev["content"])
                elif "reasoning" in ev:
                    reasoning_len += len(ev["reasoning"])
                elif "usage" in ev:
                    usage = ev["usage"]
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        return {"ok": False, "transport_error": str(e),
                "wall_s": round(time.time() - t0, 1)}
    return {
        "ok": True,
        "final_content": "".join(content_parts),
        "reasoning_len": reasoning_len,
        "usage": usage,
        "n_events": n_events,
        "viewer": ({"queries": len(bridge.events),
                    "loaded": bool(bridge.loaded_pdb)} if bridge else None),
        "wall_s": round(time.time() - t0, 1),
    }


# --------------------------------------------------------------------------- #
# Trace pull
# --------------------------------------------------------------------------- #
def _fetch_spans(limit: int = 200) -> list[dict]:
    q = f'''{{
      node(id: "{AEVOM_PROJECT_ID}") {{
        ... on Project {{
          spans(first: {limit}, sort: {{col: startTime, dir: desc}}) {{
            edges {{ node {{
              name spanId parentId statusCode startTime endTime latencyMs
              attributes context {{ traceId }}
            }} }}
          }}
        }}
      }}
    }}'''
    res = graphql_query(q, quiet=True)
    node = (res.get("data") or {}).get("node") or {}
    out = []
    for edge in (node.get("spans") or {}).get("edges", []):
        s = edge["node"]
        if isinstance(s.get("attributes"), str):
            try:
                s["attributes"] = json.loads(s["attributes"])
            except json.JSONDecodeError:
                s["attributes"] = {}
        s["traceId"] = (s.get("context") or {}).get("traceId")
        out.append(s)
    return out


def find_trace(message: str, since_iso: str, tries: int = 12,
               delay: float = 2.0) -> list[dict]:
    """Find all spans of the chat_request whose user_message matches `message`
    and which started at/after `since_iso`. Retries while spans flush."""
    for _ in range(tries):
        spans = _fetch_spans()
        root = None
        for s in spans:
            if s["name"] != "chat_request":
                continue
            if s["startTime"] < since_iso:
                continue
            chat = (s.get("attributes") or {}).get("chat", {})
            um = chat.get("user_message") or chat.get("user_message_preview") or ""
            if message[:80] in um:
                root = s
                break
        if root and root.get("traceId"):
            trace_spans = [s for s in spans if s.get("traceId") == root["traceId"]]
            # ensure children flushed: chat_request has closed (has final attrs)
            if (root["attributes"].get("chat", {}).get("total_iterations")
                    is not None) or len(trace_spans) > 1:
                return trace_spans
        time.sleep(delay)
    return []


def summarise(spans: list[dict]) -> dict:
    root = next((s for s in spans if s["name"] == "chat_request"), None)
    chat = (root or {}).get("attributes", {}).get("chat", {}) if root else {}
    tools, model = [], None
    for s in spans:
        attrs = s.get("attributes", {})
        if s["name"].startswith("tool."):
            t = attrs.get("tool", {})
            res = t.get("result", {}) or {}
            args_raw = (t.get("raw", {}) or {}).get("arguments")
            tools.append({
                "name": s["name"][len("tool."):],
                "success": res.get("success"),
                "status": s.get("statusCode"),
                "error": t.get("error"),
                "content_type": res.get("content_type"),
                "args_preview": (args_raw[:200] if isinstance(args_raw, str)
                                 else json.dumps(t.get("arg", {}))[:200]),
                "latency_ms": s.get("latencyMs"),
            })
        elif s["name"].startswith("llm.") and not model:
            model = attrs.get("llm", {}).get("model")
    # tools are collected out of order; sort by startTime
    tool_spans = sorted(
        [s for s in spans if s["name"].startswith("tool.")],
        key=lambda s: s["startTime"])
    tool_order = [s["name"][len("tool."):] for s in tool_spans]
    # Iteration count / limit are derived from spans directly, since the
    # summary attrs on chat_request are not reliably populated in this path.
    n_iters = sum(1 for s in spans if s["name"].startswith("reasoning_iteration"))
    return {
        "model": model,
        "n_iterations": chat.get("total_iterations") or n_iters or None,
        "n_tool_calls": chat.get("total_tool_calls") or len(tool_order),
        "hit_iteration_limit": bool(chat.get("hit_iteration_limit")),
        "final_response": (chat.get("final_response") or "")[:600],
        "tool_order": tool_order,
        "tools": tools,
        "any_tool_error": any(
            (t["success"] is False) or (t["status"] == "ERROR") or t["error"]
            for t in tools),
        "n_spans": len(spans),
    }


# --------------------------------------------------------------------------- #
# Coverage matrix
# --------------------------------------------------------------------------- #
def update_coverage(cov_path: str, tool_order: list[str]) -> None:
    cov = {"tools": {}, "pairs": {}}
    if os.path.exists(cov_path):
        with open(cov_path) as f:
            cov = json.load(f)
    for t in tool_order:
        cov["tools"][t] = cov["tools"].get(t, 0) + 1
    for a, b in zip(tool_order, tool_order[1:]):
        if a != b:
            key = f"{a}->{b}"
            cov["pairs"][key] = cov["pairs"].get(key, 0) + 1
    with open(cov_path, "w") as f:
        json.dump(cov, f, indent=2, sort_keys=True)


# --------------------------------------------------------------------------- #
def run_one(task_id: str, message: str, session: str, outdir: str,
            max_tokens: int, trace_only: bool, send_timeout: float = 600.0,
            molstar_base: str | None = None) -> dict:
    os.makedirs(outdir, exist_ok=True)
    # For a fresh send, only match traces started after now; for --trace-only we
    # want the most recent matching trace regardless of age.
    since = ("2000-01-01T00:00:00+00:00" if trace_only
             else datetime.now(timezone.utc).isoformat())
    chat_res = {"ok": True, "skipped": "trace-only"}
    if not trace_only:
        session_id = open_session(session)
        print(f"  → sending [{task_id}] session={session} ({session_id})",
              file=sys.stderr)
        chat_res = send_chat(message, session_id, max_tokens, timeout=send_timeout,
                             molstar_base=molstar_base)
        if not chat_res.get("ok"):
            print(f"  ✗ transport error: {chat_res.get('transport_error')}",
                  file=sys.stderr)
    spans = find_trace(message, since)
    summary = summarise(spans) if spans else {"error": "trace_not_found"}
    record = {
        "ts": since, "task_id": task_id, "session": session,
        "session_id": _sessions.get(session),
        "message": message, "chat": chat_res, "trace": summary,
    }
    with open(os.path.join(outdir, "runs.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")
    if summary.get("tool_order"):
        update_coverage(os.path.join(outdir, "coverage.json"),
                        summary["tool_order"])
    # console line
    to = summary.get("tool_order", [])
    err = "ERR" if summary.get("any_tool_error") else "ok "
    print(f"  [{err}] {task_id}: model={summary.get('model')} "
          f"iters={summary.get('n_iterations')} tools={to}", file=sys.stderr)
    for t in summary.get("tools", []):
        if (t["success"] is False) or (t["status"] == "ERROR") or t["error"]:
            print(f"        ✗ {t['name']}: {(t['error'] or '')[:160]}",
                  file=sys.stderr)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", default="adhoc")
    ap.add_argument("--message")
    ap.add_argument("--suite", help="JSON file: list of {id,message,[session]}")
    ap.add_argument("--session",
                    help="label; tasks sharing one share a backend session")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--send-timeout", type=float, default=600.0,
                    help="SSE read timeout; raise for slow NIM tools")
    ap.add_argument("--trace-only", action="store_true")
    ap.add_argument("--tag", default="", help="suffix appended to session labels")
    ap.add_argument("--no-viewer", action="store_true",
                    help="disable the molstar bridge (structure_query will time out)")
    args = ap.parse_args()

    molstar_base = None if args.no_viewer else resolve_molstar_base()
    if not args.trace_only:
        if molstar_base:
            print(f"  viewer: molstar bridge -> {molstar_base}", file=sys.stderr)
        else:
            print("  viewer: NONE (structure_query will time out)", file=sys.stderr)

    tasks = []
    if args.suite:
        with open(args.suite) as f:
            tasks = json.load(f)
    elif args.message:
        tasks = [{"id": args.task_id, "message": args.message,
                  "session": args.session}]
    else:
        ap.error("need --message or --suite")

    stamp = datetime.now(timezone.utc).strftime("%H%M%S")
    for i, t in enumerate(tasks):
        sess = t.get("session") or f"camp-{t['id']}-{stamp}{args.tag}"
        run_one(t["id"], t["message"], sess, args.outdir,
                args.max_tokens, args.trace_only, args.send_timeout,
                molstar_base=molstar_base)
        if i < len(tasks) - 1:
            time.sleep(1.0)


if __name__ == "__main__":
    main()
