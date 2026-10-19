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
  useEffect,
  useRef,
  useCallback,
  FormEvent,
  DragEvent,
  ChangeEvent,
} from "react";

interface AddExtensionModalProps {
  isOpen: boolean;
  onClose: () => void;
  sessionId: string;
  /** Called after an MCP server, skill, or key is successfully added. */
  onAdded: () => void;
  /** Whether the agent has the generic `http_request` tool (`features.http_request_tool`). */
  httpRequestToolEnabled: boolean;
}

type Tab = "mcp" | "nim" | "skill" | "key";

/** What the backend learned about a NIM when it probed the URL. */
interface NimRegistration {
  id: string;
  name: string;
  model_id: string | null;
  model_version: string | null;
  description: string | null;
  ready: boolean;
  adapter: string | null;
  adapter_label: string | null;
  adapter_summary: string;
  tools: string[];
  repointed_tools: string[];
}

interface SkillFilePayload {
  path: string;
  content_base64: string;
}

interface SkillPreview {
  name?: string;
  description?: string;
  fileCount: number;
  paths: string[];
}

/**
 * A skill's data-type contract, mirroring `api.skills.contract`.
 *
 * The backend proposes one at install time (from the SKILL.md `endpoints:` block,
 * or by reading the prose with an LLM). Confirming it is what turns it into real
 * tools, and it also authorizes outbound calls to the endpoint hosts -- so the
 * user reviews it here rather than having it applied silently.
 */
interface SkillContractInput {
  name: string;
  data_type: string;
  json_path: string;
  required?: boolean;
  description?: string;
}

interface SkillContractOutput {
  name: string;
  json_path: string;
  data_type: string;
  description?: string;
  /** The path resolves to a list; store every element (e.g. ranked poses). */
  each?: boolean;
}

interface SkillContractEndpoint {
  tool_suffix: string;
  summary: string;
  url: string;
  method: string;
  secret_name?: string | null;
  inputs: SkillContractInput[];
  scalar_params: { name: string; type: string }[];
  outputs: SkillContractOutput[];
  capture_unmapped?: boolean;
}

interface SkillContract {
  endpoints: SkillContractEndpoint[];
  source: "declared" | "llm" | "none";
  confirmed: boolean;
}

interface SkillInstallResult {
  id: string;
  name: string;
  contract: SkillContract | null;
  tool_names: string[];
  contract_hosts: string[];
  blocked_hosts: string[];
  /**
   * Set when the skill documents an API but Aevom could not derive a contract
   * from it. The install succeeded, so without showing this the user cannot tell
   * a skill that gained a tool from one that has only its instructions.
   */
  contract_error?: string | null;
}

/** Data types the review step offers, matching the backend's DataType enum. */
const DATA_TYPE_OPTIONS = [
  "smiles",
  "safe",
  "sdf",
  "pdb",
  "mmcif",
  "sequence",
  "dna",
  "rna",
  "msa",
  "json",
  "array",
];

// Close icon
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

// Server icon
const ServerIcon = () => (
  <svg
    width="24"
    height="24"
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

// GPU/chip icon, for a model served on hardware the user runs
const NimIcon = () => (
  <svg
    width="24"
    height="24"
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
    width="24"
    height="24"
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

// Key icon
const KeyIcon = () => (
  <svg
    width="24"
    height="24"
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

// --- File helpers ----------------------------------------------------------

const readFileAsText = (file: File): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve((reader.result as string) ?? "");
    reader.onerror = () => reject(reader.error);
    reader.readAsText(file);
  });

const readFileAsBase64 = (file: File): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // result is a data URL: "data:...;base64,XXXX"
      const comma = result.indexOf(",");
      resolve(comma >= 0 ? result.slice(comma + 1) : result);
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });

// Recursively walk a dropped directory entry (webkitGetAsEntry API).
const readDirectoryEntry = (
  entry: FileSystemEntry,
  files: { path: string; file: File }[],
): Promise<void> => {
  return new Promise((resolve, reject) => {
    if (entry.isFile) {
      (entry as FileSystemFileEntry).file((file) => {
        files.push({ path: entry.fullPath.replace(/^\//, ""), file });
        resolve();
      }, reject);
    } else if (entry.isDirectory) {
      const reader = (entry as FileSystemDirectoryEntry).createReader();
      const entries: FileSystemEntry[] = [];
      const readBatch = () => {
        reader.readEntries((batch) => {
          if (batch.length === 0) {
            Promise.all(entries.map((e) => readDirectoryEntry(e, files)))
              .then(() => resolve())
              .catch(reject);
          } else {
            entries.push(...batch);
            readBatch();
          }
        }, reject);
      };
      readBatch();
    } else {
      resolve();
    }
  });
};

// Very small SKILL.md frontmatter reader for client-side preview only.
const previewFrontmatter = (
  text: string,
): { name?: string; description?: string } => {
  const match = text.match(/^\ufeff?---\s*\n([\s\S]*?)\n---/);
  if (!match) return {};
  const meta: { name?: string; description?: string } = {};
  for (const line of match[1].split("\n")) {
    const kv = line.match(/^(name|description)\s*:\s*(.*)$/);
    if (kv) {
      meta[kv[1] as "name" | "description"] = kv[2].trim().replace(/^["']|["']$/g, "");
    }
  }
  return meta;
};

// Shown until the first phase event arrives from the server.
const INSTALL_STARTING_MESSAGE = "Starting install...";

/** One event from the install progress stream. */
interface InstallStreamEvent {
  phase?: string;
  message?: string;
  detail?: string;
  skill?: SkillInstallResult;
}

/** Consume the install progress stream, returning the installed skill.
 *
 * Progress events carry a human-readable description of the phase the server
 * has just started; the last event carries either the skill or an error.
 */
const readInstallStream = async (
  response: Response,
  onPhase: (message: string) => void,
): Promise<SkillInstallResult> => {
  const reader = response.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";
  let installed: SkillInstallResult | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    // Keep any incomplete trailing line for the next chunk.
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const payload = line.slice(6).trim();
      if (!payload) continue;

      let event: InstallStreamEvent;
      try {
        event = JSON.parse(payload);
      } catch {
        continue;
      }

      if (event.phase === "error") {
        throw new Error(event.detail || "Failed to install skill");
      }
      if (event.phase === "installed" && event.skill) {
        installed = event.skill;
      } else if (event.message) {
        onPhase(event.message);
      }
    }
  }

  if (!installed) throw new Error("The install ended without a result.");
  return installed;
};

export default function AddExtensionModal({
  isOpen,
  onClose,
  sessionId,
  onAdded,
  httpRequestToolEnabled,
}: AddExtensionModalProps) {
  const [activeTab, setActiveTab] = useState<Tab>("mcp");

  // MCP form state
  const [name, setName] = useState("");
  const [url, setUrl] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [mcpSubmitting, setMcpSubmitting] = useState(false);
  const [mcpError, setMcpError] = useState<string | null>(null);
  const [toolsDiscovered, setToolsDiscovered] = useState<number | null>(null);

  // NIM form state. The form fields mirror MCP's, but the result does not: a
  // NIM reports a model rather than a toolset, so the success screen shows
  // what Aevom recognised the model as.
  const [nimName, setNimName] = useState("");
  const [nimUrl, setNimUrl] = useState("");
  const [nimApiKey, setNimApiKey] = useState("");
  const [nimSubmitting, setNimSubmitting] = useState(false);
  const [nimError, setNimError] = useState<string | null>(null);
  const [nimRegistered, setNimRegistered] = useState<NimRegistration | null>(
    null,
  );

  // Skill state
  const [skillFiles, setSkillFiles] = useState<SkillFilePayload[]>([]);
  const [skillPreview, setSkillPreview] = useState<SkillPreview | null>(null);
  const [githubUrl, setGithubUrl] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [skillSubmitting, setSkillSubmitting] = useState(false);
  const [skillError, setSkillError] = useState<string | null>(null);
  const [skillInstalled, setSkillInstalled] = useState<string | null>(null);

  // Contract review: set when an installed skill proposed an unconfirmed
  // contract. While non-null the skill tab shows the review step instead of
  // closing, because confirming authorizes outbound calls to `contract_hosts`.
  const [pendingSkill, setPendingSkill] = useState<SkillInstallResult | null>(
    null,
  );
  const [draftContract, setDraftContract] = useState<SkillContract | null>(null);
  // Why a skill that documents an API ended up with no synthesized tool. Shown
  // on the success screen, since the install itself did succeed.
  const [skillContractError, setSkillContractError] = useState<string | null>(
    null,
  );
  // Names of the session's stored API keys. The extractor can only guess an
  // endpoint's key name from whatever env var the docs happen to use, so the
  // review step checks the guess against these before the contract is
  // confirmed -- a wrong name yields a tool that fails on every call.
  const [storedKeyNames, setStoredKeyNames] = useState<string[]>([]);
  // The server's description of the install phase currently running. Deriving a
  // contract takes tens of seconds, and the server reports each phase as it
  // starts so this reflects real progress rather than a timed guess.
  const [installMessage, setInstallMessage] = useState(
    INSTALL_STARTING_MESSAGE,
  );

  // Key (secret) state
  const [keyName, setKeyName] = useState("");
  const [keyValue, setKeyValue] = useState("");
  const [keyFromFile, setKeyFromFile] = useState(false);
  const [keyDragging, setKeyDragging] = useState(false);
  const [keySubmitting, setKeySubmitting] = useState(false);
  const [keyError, setKeyError] = useState<string | null>(null);
  const [keySaved, setKeySaved] = useState<string | null>(null);

  const nameInputRef = useRef<HTMLInputElement>(null);
  const dirInputRef = useRef<HTMLInputElement>(null);
  const zipInputRef = useRef<HTMLInputElement>(null);
  const keyFileInputRef = useRef<HTMLInputElement>(null);

  // Reset all state when the modal closes
  useEffect(() => {
    if (!isOpen) {
      setActiveTab("mcp");
      setName("");
      setUrl("");
      setApiKey("");
      setMcpError(null);
      setToolsDiscovered(null);
      setNimName("");
      setNimUrl("");
      setNimApiKey("");
      setNimError(null);
      setNimRegistered(null);
      setSkillFiles([]);
      setSkillPreview(null);
      setGithubUrl("");
      setIsDragging(false);
      setSkillError(null);
      setSkillInstalled(null);
      setPendingSkill(null);
      setDraftContract(null);
      setSkillContractError(null);
      setStoredKeyNames([]);
      setInstallMessage(INSTALL_STARTING_MESSAGE);
      setKeyName("");
      setKeyValue("");
      setKeyFromFile(false);
      setKeyDragging(false);
      setKeyError(null);
      setKeySaved(null);
    }
  }, [isOpen]);

  // True only for the install request itself. Confirming a reviewed contract
  // also sets skillSubmitting, but that call is fast and needs no narration.
  const skillInstalling = skillSubmitting && pendingSkill === null;

  // Load stored key names once the review step opens, so a proposed key name
  // that is not actually stored can be corrected before it is confirmed.
  useEffect(() => {
    if (!pendingSkill || !sessionId) return;
    let cancelled = false;
    (async () => {
      try {
        const response = await fetch(
          `/api/secrets/?session_id=${encodeURIComponent(sessionId)}`,
        );
        if (!response.ok) return;
        const data = await response.json();
        if (cancelled) return;
        const names: string[] = (data.secrets ?? []).map(
          (s: { name: string }) => s.name,
        );
        setStoredKeyNames(names);
        // The proposed name is only whatever the docs called the key. If the
        // session already holds one, start from that so the common case needs
        // no edit; the proposal stays selectable in the dropdown.
        if (names.length > 0) {
          setDraftContract((prev) => {
            if (!prev) return prev;
            const endpoints = prev.endpoints.map((ep) =>
              ep.secret_name && !names.includes(ep.secret_name)
                ? { ...ep, secret_name: names[0] }
                : ep,
            );
            return { ...prev, endpoints };
          });
        }
      } catch {
        // Losing this lookup only costs the mismatch warning. The review step
        // must still be usable, so there is nothing to report here.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [pendingSkill, sessionId]);

  // Escape key closes
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (
        e.key === "Escape" &&
        isOpen &&
        !mcpSubmitting &&
        !skillSubmitting &&
        !keySubmitting
      ) {
        onClose();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, mcpSubmitting, skillSubmitting, keySubmitting, onClose]);

  // --- MCP handlers --------------------------------------------------------

  const normalizeUrl = (urlString: string): string => {
    const trimmed = urlString.trim();
    return /^https?:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`;
  };

  const handleMcpSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setMcpError(null);
    setToolsDiscovered(null);

    if (!name.trim()) return setMcpError("Name is required");
    if (!url.trim()) return setMcpError("URL is required");

    const normalizedUrl = normalizeUrl(url);
    setMcpSubmitting(true);
    try {
      const tryConnect = (targetUrl: string) =>
        fetch(`/api/mcp/servers?session_id=${encodeURIComponent(sessionId)}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: name.trim(),
            url: targetUrl,
            api_key: apiKey.trim() || undefined,
          }),
        });

      const userTypedProtocol = /^https?:\/\//i.test(url.trim());
      let response = await tryConnect(normalizedUrl);
      let data = await response.json();

      if (
        !userTypedProtocol &&
        !response.ok &&
        response.status === 502 &&
        normalizedUrl.startsWith("https://")
      ) {
        response = await tryConnect(normalizedUrl.replace(/^https:\/\//, "http://"));
        data = await response.json();
      }

      if (!response.ok) {
        throw new Error(data.detail || "Failed to connect to MCP server");
      }

      setToolsDiscovered(data.tools_count || 0);
      setTimeout(() => {
        onAdded();
        onClose();
      }, 1500);
    } catch (err) {
      setMcpError(err instanceof Error ? err.message : "Unknown error occurred");
    } finally {
      setMcpSubmitting(false);
    }
  };

  // --- NIM handlers --------------------------------------------------------

  const handleNimSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setNimError(null);
    setNimRegistered(null);

    if (!nimName.trim()) return setNimError("Name is required");
    if (!nimUrl.trim()) return setNimError("URL is required");

    // A self-hosted NIM is usually on a plain-HTTP host inside the network,
    // so unlike MCP the http:// attempt comes first rather than as a
    // fallback. normalizeUrl still handles a user who typed a scheme.
    const typed = nimUrl.trim();
    const hasScheme = /^https?:\/\//i.test(typed);
    const candidates = hasScheme
      ? [typed]
      : [`http://${typed}`, `https://${typed}`];

    setNimSubmitting(true);
    try {
      let data: { detail?: string } & Partial<NimRegistration> = {};
      let ok = false;

      for (const candidate of candidates) {
        const response = await fetch(
          `/api/nim/servers?session_id=${encodeURIComponent(sessionId)}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              name: nimName.trim(),
              url: candidate,
              api_key: nimApiKey.trim() || undefined,
            }),
          },
        );
        data = await response.json();
        ok = response.ok;
        // 502 means "nothing answered there", which is the only failure
        // worth retrying on the other scheme. A 401 or a 500 is about this
        // endpoint and would say the same thing twice.
        if (ok || response.status !== 502) break;
      }

      if (!ok) {
        throw new Error(data.detail || "Failed to reach the NIM");
      }

      setNimRegistered(data as NimRegistration);
    } catch (err) {
      setNimError(err instanceof Error ? err.message : "Unknown error occurred");
    } finally {
      setNimSubmitting(false);
    }
  };

  // --- Skill handlers ------------------------------------------------------

  const buildPayload = useCallback(
    async (collected: { path: string; file: File }[]) => {
      if (collected.length === 0) {
        setSkillError("No files found in the dropped item.");
        return;
      }
      const payload: SkillFilePayload[] = [];
      for (const { path, file } of collected) {
        payload.push({ path, content_base64: await readFileAsBase64(file) });
      }

      // Find SKILL.md for preview
      const skillEntry = collected
        .filter((c) => c.path.toLowerCase().endsWith("skill.md"))
        .sort((a, b) => a.path.split("/").length - b.path.split("/").length)[0];

      let meta: { name?: string; description?: string } = {};
      if (skillEntry) {
        meta = previewFrontmatter(await skillEntry.file.text());
      } else {
        setSkillError("No SKILL.md found in the dropped folder.");
        return;
      }

      setSkillError(null);
      setSkillFiles(payload);
      setSkillPreview({
        name: meta.name,
        description: meta.description,
        fileCount: payload.length,
        paths: payload.map((p) => p.path),
      });
    },
    [],
  );

  const handleDrop = async (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    setSkillError(null);

    const items = Array.from(e.dataTransfer.items);
    const zipItem = items.find((i) =>
      i.getAsFile()?.name.toLowerCase().endsWith(".zip"),
    );
    if (zipItem) {
      const file = zipItem.getAsFile();
      if (file) await handleZipFile(file);
      return;
    }

    const collected: { path: string; file: File }[] = [];
    const entries = items
      .map((i) => (i.webkitGetAsEntry ? i.webkitGetAsEntry() : null))
      .filter((entry): entry is FileSystemEntry => entry !== null);

    if (entries.length === 0) {
      setSkillError("Your browser did not provide directory contents. Try the folder picker.");
      return;
    }
    try {
      for (const entry of entries) {
        await readDirectoryEntry(entry, collected);
      }
      await buildPayload(collected);
    } catch {
      setSkillError("Failed to read the dropped folder.");
    }
  };

  const handleDirInput = async (e: ChangeEvent<HTMLInputElement>) => {
    const fileList = e.target.files;
    if (!fileList || fileList.length === 0) return;
    const collected = Array.from(fileList).map((file) => ({
      path: (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name,
      file,
    }));
    await buildPayload(collected);
  };

  const handleZipFile = async (file: File) => {
    try {
      const JSZip = (await import("jszip")).default;
      const zip = await JSZip.loadAsync(file);
      const collected: { path: string; file: File }[] = [];
      const entries = Object.values(zip.files).filter((f) => !f.dir);
      for (const entry of entries) {
        const blob = await entry.async("blob");
        collected.push({
          path: entry.name,
          file: new File([blob], entry.name.split("/").pop() || entry.name),
        });
      }
      await buildPayload(collected);
    } catch {
      setSkillError("Failed to read the .zip archive.");
    }
  };

  const handleZipInput = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) await handleZipFile(file);
  };

  const handleSkillInstall = async () => {
    setSkillError(null);
    const hasFiles = skillFiles.length > 0;
    const hasUrl = githubUrl.trim().length > 0;
    if (!hasFiles && !hasUrl) {
      setSkillError("Drop a skill folder, choose a .zip, or enter a GitHub URL.");
      return;
    }

    setSkillSubmitting(true);
    setInstallMessage(INSTALL_STARTING_MESSAGE);
    try {
      const body = hasFiles
        ? { files: skillFiles }
        : { github_url: githubUrl.trim() };
      const response = await fetch(
        `/api/skills/stream?session_id=${encodeURIComponent(sessionId)}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
      );
      if (!response.ok) {
        // A pre-stream rejection (bad session, say) answers with plain JSON.
        const failure = await response.json().catch(() => ({}));
        throw new Error(failure.detail || "Failed to install skill");
      }

      const data = await readInstallStream(response, setInstallMessage);

      // A declared contract arrives already confirmed, so there is nothing to
      // review. An unconfirmed one needs the user's approval before it becomes
      // tools that can call out to its hosts.
      if (data.contract && !data.contract.confirmed && data.contract.endpoints?.length) {
        setPendingSkill(data);
        setDraftContract(data.contract);
        onAdded();
        return;
      }

      setSkillInstalled(data.name || "skill");
      // Auto-dismissing would flash past the explanation, and it is the only
      // notice that this skill has no synthesized tool.
      if (data.contract_error) {
        setSkillContractError(data.contract_error);
        onAdded();
        return;
      }
      setTimeout(() => {
        onAdded();
        onClose();
      }, 1500);
    } catch (err) {
      setSkillError(err instanceof Error ? err.message : "Unknown error occurred");
    } finally {
      setSkillSubmitting(false);
    }
  };

  /** Send the reviewed contract, or null to install the skill without tools. */
  const submitContract = async (contract: SkillContract | null) => {
    if (!pendingSkill) return;
    setSkillError(null);
    setSkillSubmitting(true);
    try {
      const response = await fetch(
        `/api/skills/${encodeURIComponent(pendingSkill.id)}/contract?session_id=${encodeURIComponent(sessionId)}`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ contract }),
        },
      );
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Failed to save the contract");
      }
      setSkillInstalled(pendingSkill.name || "skill");
      setPendingSkill(null);
      setDraftContract(null);
      setTimeout(() => {
        onAdded();
        onClose();
      }, 1500);
    } catch (err) {
      setSkillError(err instanceof Error ? err.message : "Unknown error occurred");
    } finally {
      setSkillSubmitting(false);
    }
  };

  /** Key names to offer for an endpoint's auth dropdown.
   *
   * The stored keys plus the name originally proposed for this endpoint. The
   * proposal has to stay in the list after switching away from it, otherwise
   * choosing a stored key once makes it unreachable for the rest of the review.
   * `pendingSkill.contract` holds the untouched proposal, since edits go to
   * `draftContract`.
   */
  const keyNameOptions = (endpointIndex: number, current: string): string[] => {
    const proposed =
      pendingSkill?.contract?.endpoints?.[endpointIndex]?.secret_name ?? null;
    return Array.from(
      new Set(
        [...storedKeyNames, proposed, current].filter(
          (candidate): candidate is string => Boolean(candidate),
        ),
      ),
    );
  };

  /** Apply an edit to one endpoint of the draft contract. */
  const patchEndpoint = (
    index: number,
    patch: Partial<SkillContractEndpoint>,
  ) => {
    setDraftContract((prev) => {
      if (!prev) return prev;
      const endpoints = prev.endpoints.map((ep, i) =>
        i === index ? { ...ep, ...patch } : ep,
      );
      return { ...prev, endpoints };
    });
  };

  const patchInput = (
    endpointIndex: number,
    inputIndex: number,
    patch: Partial<SkillContractInput>,
  ) => {
    setDraftContract((prev) => {
      if (!prev) return prev;
      const endpoints = prev.endpoints.map((ep, i) => {
        if (i !== endpointIndex) return ep;
        const inputs = ep.inputs.map((inp, j) =>
          j === inputIndex ? { ...inp, ...patch } : inp,
        );
        return { ...ep, inputs };
      });
      return { ...prev, endpoints };
    });
  };

  const patchOutput = (
    endpointIndex: number,
    outputIndex: number,
    patch: Partial<SkillContractOutput>,
  ) => {
    setDraftContract((prev) => {
      if (!prev) return prev;
      const endpoints = prev.endpoints.map((ep, i) => {
        if (i !== endpointIndex) return ep;
        const outputs = ep.outputs.map((out, j) =>
          j === outputIndex ? { ...out, ...patch } : out,
        );
        return { ...ep, outputs };
      });
      return { ...prev, endpoints };
    });
  };

  const dropInput = (endpointIndex: number, inputIndex: number) => {
    setDraftContract((prev) => {
      if (!prev) return prev;
      const endpoints = prev.endpoints.map((ep, i) =>
        i === endpointIndex
          ? { ...ep, inputs: ep.inputs.filter((_, j) => j !== inputIndex) }
          : ep,
      );
      return { ...prev, endpoints };
    });
  };

  const dropOutput = (endpointIndex: number, outputIndex: number) => {
    setDraftContract((prev) => {
      if (!prev) return prev;
      const endpoints = prev.endpoints.map((ep, i) =>
        i === endpointIndex
          ? { ...ep, outputs: ep.outputs.filter((_, j) => j !== outputIndex) }
          : ep,
      );
      return { ...prev, endpoints };
    });
  };

  // --- Key (secret) handlers ----------------------------------------------

  const loadKeyFromFile = async (file: File) => {
    try {
      const text = (await readFileAsText(file)).trim();
      if (!text) {
        setKeyError("The dropped file is empty.");
        return;
      }
      setKeyError(null);
      setKeyValue(text);
      setKeyFromFile(true);
    } catch {
      setKeyError("Failed to read the dropped file.");
    }
  };

  const handleKeyDrop = async (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setKeyDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) await loadKeyFromFile(file);
  };

  const handleKeyFileInput = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) await loadKeyFromFile(file);
  };

  const clearKeyFile = () => {
    setKeyFromFile(false);
    setKeyValue("");
    if (keyFileInputRef.current) keyFileInputRef.current.value = "";
  };

  const handleKeySubmit = async (e: FormEvent) => {
    e.preventDefault();
    setKeyError(null);

    if (!keyName.trim()) return setKeyError("Name is required");
    if (!keyValue.trim()) return setKeyError("API key is required");

    setKeySubmitting(true);
    try {
      const response = await fetch(
        `/api/secrets/?session_id=${encodeURIComponent(sessionId)}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: keyName.trim(), value: keyValue.trim() }),
        },
      );
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Failed to save key");
      }
      setKeySaved(data.name || keyName.trim());
      setTimeout(() => {
        onAdded();
        onClose();
      }, 1200);
    } catch (err) {
      setKeyError(err instanceof Error ? err.message : "Unknown error occurred");
    } finally {
      setKeySubmitting(false);
    }
  };

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (
      e.target === e.currentTarget &&
      !mcpSubmitting &&
      !skillSubmitting &&
      !keySubmitting
    ) {
      onClose();
    }
  };

  const modalTitle =
    activeTab === "mcp"
      ? "Add MCP Server"
      : activeTab === "nim"
        ? "Add NIM"
        : activeTab === "skill"
          ? "Add Skill"
          : "Add Key";

  const modalIcon =
    activeTab === "mcp" ? (
      <ServerIcon />
    ) : activeTab === "nim" ? (
      <NimIcon />
    ) : activeTab === "skill" ? (
      <SkillIcon />
    ) : (
      <KeyIcon />
    );

  if (!isOpen) return null;

  return (
    <div className="modal-backdrop" onClick={handleBackdropClick}>
      <div className="modal-container">
        {/* Header */}
        <div className="modal-header">
          <div className="modal-title">
            {modalIcon}
            <h2>{modalTitle}</h2>
          </div>
          <button
            className="modal-close-btn"
            onClick={onClose}
            disabled={mcpSubmitting || skillSubmitting || keySubmitting}
            title="Close"
          >
            <CloseIcon />
          </button>
        </div>

        {/* Tabs */}
        <div className="modal-tabs">
          <button
            className={`modal-tab ${activeTab === "mcp" ? "active" : ""}`}
            onClick={() => setActiveTab("mcp")}
            type="button"
          >
            <ServerIcon />
            <span>MCP Server</span>
          </button>
          <button
            className={`modal-tab ${activeTab === "nim" ? "active" : ""}`}
            onClick={() => setActiveTab("nim")}
            type="button"
          >
            <NimIcon />
            <span>NIM</span>
          </button>
          <button
            className={`modal-tab ${activeTab === "skill" ? "active" : ""}`}
            onClick={() => setActiveTab("skill")}
            type="button"
          >
            <SkillIcon />
            <span>Skill</span>
          </button>
          <button
            className={`modal-tab ${activeTab === "key" ? "active" : ""}`}
            onClick={() => setActiveTab("key")}
            type="button"
          >
            <KeyIcon />
            <span>Key</span>
          </button>
        </div>

        {/* MCP tab */}
        {activeTab === "mcp" && (
          <form onSubmit={handleMcpSubmit}>
            <div className="modal-body">
              {toolsDiscovered !== null ? (
                <div className="modal-success">
                  <div className="success-icon">✓</div>
                  <p>Connected successfully!</p>
                  <p className="success-detail">
                    Discovered {toolsDiscovered} tool
                    {toolsDiscovered !== 1 ? "s" : ""}
                  </p>
                  {/* Connecting is not enabling: the backend withholds these
                      tools from the agent until they are reviewed. Saying so
                      here avoids the modal implying the server is live. */}
                  <p className="success-detail">
                    Review and enable {toolsDiscovered !== 1 ? "them" : "it"} in
                    the tools panel before the agent can use{" "}
                    {toolsDiscovered !== 1 ? "them" : "it"}.
                  </p>
                </div>
              ) : (
                <>
                  <div className="form-group">
                    <label htmlFor="server-name">Name</label>
                    <input
                      ref={nameInputRef}
                      id="server-name"
                      type="text"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      placeholder="My MCP Server"
                      disabled={mcpSubmitting}
                      autoComplete="off"
                    />
                    <span className="form-hint">
                      A friendly name to identify this server
                    </span>
                  </div>
                  <div className="form-group">
                    <label htmlFor="server-url">URL</label>
                    <input
                      id="server-url"
                      type="text"
                      value={url}
                      onChange={(e) => setUrl(e.target.value)}
                      placeholder="mcp.example.com"
                      disabled={mcpSubmitting}
                      autoComplete="off"
                    />
                    <span className="form-hint">
                      The MCP server endpoint (https:// added automatically)
                    </span>
                  </div>
                  <div className="form-group">
                    <label htmlFor="server-api-key">
                      API Key <span className="optional">(optional)</span>
                    </label>
                    <input
                      id="server-api-key"
                      type="password"
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      placeholder="Enter API key if required"
                      disabled={mcpSubmitting}
                      autoComplete="off"
                    />
                  </div>
                  {mcpError && (
                    <div className="modal-error">
                      <span className="error-icon">!</span>
                      <span>{mcpError}</span>
                    </div>
                  )}
                </>
              )}
            </div>
            {toolsDiscovered === null && (
              <div className="modal-footer">
                <button
                  type="button"
                  className="modal-btn secondary"
                  onClick={onClose}
                  disabled={mcpSubmitting}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="modal-btn primary"
                  disabled={mcpSubmitting || !name.trim() || !url.trim()}
                >
                  {mcpSubmitting ? (
                    <>
                      <span className="loading-spinner small"></span>
                      <span>Connecting...</span>
                    </>
                  ) : (
                    "Connect"
                  )}
                </button>
              </div>
            )}
          </form>
        )}

        {/* NIM tab */}
        {activeTab === "nim" && (
          <form onSubmit={handleNimSubmit}>
            <div className="modal-body">
              {nimRegistered !== null ? (
                <div className="modal-success">
                  <div className="success-icon">✓</div>
                  <p>
                    Found {nimRegistered.adapter_label ?? "a NIM"}
                    {nimRegistered.ready ? "" : " (not ready yet)"}
                  </p>
                  <p className="success-detail">
                    {nimRegistered.model_id}
                    {nimRegistered.model_version
                      ? ` v${nimRegistered.model_version}`
                      : ""}
                  </p>

                  {/* A NIM Aevom does not recognise is still registered. The
                      user learns their endpoint works, which is more useful
                      than refusing the URL outright. */}
                  {nimRegistered.adapter === null ? (
                    <p className="success-detail">
                      Aevom has no adapter for this model, so it contributes
                      no tools. It stays listed in the tools panel.
                    </p>
                  ) : (
                    <>
                      {nimRegistered.adapter_summary && (
                        <p className="success-detail">
                          {nimRegistered.adapter_summary}
                        </p>
                      )}
                      {nimRegistered.tools.length > 0 && (
                        <p className="success-detail">
                          Adds: {nimRegistered.tools.join(", ")}
                        </p>
                      )}
                      {nimRegistered.repointed_tools.length > 0 && (
                        <p className="success-detail">
                          Will serve{" "}
                          {nimRegistered.repointed_tools.join(", ")} in place
                          of the hosted NVIDIA endpoint.
                        </p>
                      )}
                      {/* Registering is not enabling, exactly as with MCP. */}
                      <p className="success-detail">
                        Review and enable it in the tools panel before the
                        agent can use it.
                      </p>
                    </>
                  )}

                  <div className="modal-footer">
                    <button
                      type="button"
                      className="modal-btn primary"
                      onClick={() => {
                        onAdded();
                        onClose();
                      }}
                    >
                      Done
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  <div className="form-group">
                    <label htmlFor="nim-name">Name</label>
                    <input
                      id="nim-name"
                      type="text"
                      value={nimName}
                      onChange={(e) => setNimName(e.target.value)}
                      placeholder="My RFD4-Proteina"
                      disabled={nimSubmitting}
                      autoComplete="off"
                    />
                    <span className="form-hint">
                      A friendly name to identify this NIM
                    </span>
                  </div>
                  <div className="form-group">
                    <label htmlFor="nim-url">URL</label>
                    <input
                      id="nim-url"
                      type="text"
                      value={nimUrl}
                      onChange={(e) => setNimUrl(e.target.value)}
                      placeholder="localhost:8000"
                      disabled={nimSubmitting}
                      autoComplete="off"
                    />
                    <span className="form-hint">
                      The base URL of the NIM container, not a specific
                      endpoint
                    </span>
                  </div>
                  <div className="form-group">
                    <label htmlFor="nim-api-key">
                      API Key <span className="optional">(optional)</span>
                    </label>
                    <input
                      id="nim-api-key"
                      type="password"
                      value={nimApiKey}
                      onChange={(e) => setNimApiKey(e.target.value)}
                      placeholder="Only if the NIM is behind an auth proxy"
                      disabled={nimSubmitting}
                      autoComplete="off"
                    />
                    <span className="form-hint">
                      A self-hosted NIM usually needs none
                    </span>
                  </div>
                  {nimError && (
                    <div className="modal-error">
                      <span className="error-icon">!</span>
                      <span>{nimError}</span>
                    </div>
                  )}
                </>
              )}
            </div>
            {nimRegistered === null && (
              <div className="modal-footer">
                <button
                  type="button"
                  className="modal-btn secondary"
                  onClick={onClose}
                  disabled={nimSubmitting}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="modal-btn primary"
                  disabled={
                    nimSubmitting || !nimName.trim() || !nimUrl.trim()
                  }
                >
                  {nimSubmitting ? (
                    <>
                      <span className="loading-spinner small"></span>
                      <span>Probing...</span>
                    </>
                  ) : (
                    "Connect"
                  )}
                </button>
              </div>
            )}
          </form>
        )}

        {/* Skill tab */}
        {activeTab === "skill" && (
          <>
            <div className="modal-body">
              {skillInstalled !== null ? (
                <div className="modal-success">
                  <div className="success-icon">✓</div>
                  <p>Skill installed!</p>
                  <p className="success-detail">{skillInstalled}</p>
                  {skillContractError && (
                    <div className="contract-warning">
                      <p className="contract-warning-title">
                        No tool was created for this skill
                      </p>
                      <p className="contract-warning-detail">
                        {skillContractError}
                      </p>
                      {httpRequestToolEnabled ? (
                        <p className="contract-warning-detail">
                          The skill still works: aevom will read its
                          instructions and call the API with{" "}
                          <code>http_request</code>. But large inputs like PDB
                          and SDF files cannot be passed that way, so endpoints
                          that need them may fail.
                        </p>
                      ) : (
                        <p className="contract-warning-detail">
                          aevom can read the skill's instructions, but nothing
                          will call its API: the generic HTTP tool is disabled
                          in this deployment.
                        </p>
                      )}
                      <button
                        type="button"
                        className="btn-secondary"
                        onClick={onClose}
                      >
                        Got it
                      </button>
                    </div>
                  )}
                </div>
              ) : pendingSkill && draftContract ? (
                <div className="contract-review">
                  <p className="contract-review-intro">
                    <strong>{pendingSkill.name}</strong> describes an API. aevom
                    read its instructions and proposed the tools below. Review
                    them, since approving also lets this skill send requests
                    (and your API key) to these hosts.
                  </p>

                  <div className="contract-hosts">
                    <span className="contract-hosts-label">
                      Will be authorized to call:
                    </span>
                    {pendingSkill.contract_hosts.map((host) => (
                      <code key={host} className="contract-host">
                        {host}
                      </code>
                    ))}
                  </div>

                  {pendingSkill.blocked_hosts.length > 0 && (
                    <div className="modal-error">
                      <span className="error-icon">!</span>
                      <span>
                        This deployment's allowlist does not permit{" "}
                        {pendingSkill.blocked_hosts.join(", ")}. The tools will
                        be created but their calls will be refused until an
                        administrator adds the host to
                        HTTP_TOOL_ALLOWED_HOSTS.
                      </span>
                    </div>
                  )}

                  {draftContract.endpoints.map((endpoint, ei) => (
                    <div className="contract-endpoint" key={ei}>
                      <div className="contract-endpoint-head">
                        <code className="contract-tool-name">
                          {pendingSkill.id}_{endpoint.tool_suffix}
                        </code>
                        <span className="contract-method">
                          {endpoint.method}
                        </span>
                      </div>
                      <div className="contract-url">{endpoint.url}</div>

                      <div className="form-group">
                        <label htmlFor={`ct-summary-${ei}`}>
                          What it does (the model reads this)
                        </label>
                        <input
                          id={`ct-summary-${ei}`}
                          type="text"
                          value={endpoint.summary}
                          onChange={(e) =>
                            patchEndpoint(ei, { summary: e.target.value })
                          }
                          disabled={skillSubmitting}
                        />
                      </div>

                      <div className="form-group">
                        <label htmlFor={`ct-suffix-${ei}`}>Tool name</label>
                        <input
                          id={`ct-suffix-${ei}`}
                          type="text"
                          value={endpoint.tool_suffix}
                          onChange={(e) =>
                            patchEndpoint(ei, { tool_suffix: e.target.value })
                          }
                          disabled={skillSubmitting}
                        />
                      </div>

                      <div className="contract-section-title">
                        Inputs — passed as data references, never raw text
                      </div>
                      {endpoint.inputs.length === 0 ? (
                        <p className="contract-empty">
                          No molecular inputs detected.
                        </p>
                      ) : (
                        endpoint.inputs.map((input, ii) => (
                          <div className="contract-row" key={ii}>
                            <input
                              type="text"
                              className="contract-field-name"
                              value={input.name}
                              onChange={(e) =>
                                patchInput(ei, ii, { name: e.target.value })
                              }
                              disabled={skillSubmitting}
                              aria-label="Input name"
                            />
                            <select
                              value={input.data_type}
                              onChange={(e) =>
                                patchInput(ei, ii, {
                                  data_type: e.target.value,
                                })
                              }
                              disabled={skillSubmitting}
                              aria-label="Input data type"
                            >
                              {DATA_TYPE_OPTIONS.map((dt) => (
                                <option key={dt} value={dt}>
                                  {dt}
                                </option>
                              ))}
                            </select>
                            <input
                              type="text"
                              className="contract-field-path"
                              value={input.json_path}
                              onChange={(e) =>
                                patchInput(ei, ii, {
                                  json_path: e.target.value,
                                })
                              }
                              disabled={skillSubmitting}
                              aria-label="Request body path"
                              placeholder="$.field.path"
                            />
                            <button
                              type="button"
                              className="contract-drop"
                              onClick={() => dropInput(ei, ii)}
                              disabled={skillSubmitting}
                              aria-label={`Remove input ${input.name}`}
                            >
                              ×
                            </button>
                          </div>
                        ))
                      )}

                      <div className="contract-section-title">
                        Outputs — stored and returned as references
                      </div>
                      {endpoint.outputs.length === 0 ? (
                        <p className="contract-empty">
                          No outputs mapped. Numeric response fields are still
                          saved as arrays you can load in python_execute.
                        </p>
                      ) : (
                        endpoint.outputs.map((output, oi) => (
                          <div className="contract-row" key={oi}>
                            <input
                              type="text"
                              className="contract-field-name"
                              value={output.name}
                              onChange={(e) =>
                                patchOutput(ei, oi, { name: e.target.value })
                              }
                              disabled={skillSubmitting}
                              aria-label="Output name"
                            />
                            <select
                              value={output.data_type}
                              onChange={(e) =>
                                patchOutput(ei, oi, {
                                  data_type: e.target.value,
                                })
                              }
                              disabled={skillSubmitting}
                              aria-label="Output data type"
                            >
                              {DATA_TYPE_OPTIONS.map((dt) => (
                                <option key={dt} value={dt}>
                                  {dt}
                                </option>
                              ))}
                            </select>
                            <input
                              type="text"
                              className="contract-field-path"
                              value={output.json_path}
                              onChange={(e) =>
                                patchOutput(ei, oi, {
                                  json_path: e.target.value,
                                })
                              }
                              disabled={skillSubmitting}
                              aria-label="Response body path"
                              placeholder="$.field.path"
                            />
                            <button
                              type="button"
                              className="contract-drop"
                              onClick={() => dropOutput(ei, oi)}
                              disabled={skillSubmitting}
                              aria-label={`Remove output ${output.name}`}
                            >
                              ×
                            </button>
                          </div>
                        ))
                      )}

                      {endpoint.secret_name && (
                        <div className="contract-secret">
                          <label className="contract-secret-label">
                            Authenticates with key
                            <select
                              value={endpoint.secret_name}
                              onChange={(e) =>
                                patchEndpoint(ei, {
                                  secret_name: e.target.value,
                                })
                              }
                              disabled={skillSubmitting}
                              aria-label="Stored API key to authenticate with"
                            >
                              {keyNameOptions(ei, endpoint.secret_name).map(
                                (keyOption) => (
                                  <option key={keyOption} value={keyOption}>
                                    {storedKeyNames.includes(keyOption)
                                      ? keyOption
                                      : `${keyOption} (not stored)`}
                                  </option>
                                ),
                              )}
                            </select>
                          </label>
                          {!storedKeyNames.includes(endpoint.secret_name) && (
                            <p className="contract-secret-warning">
                              {storedKeyNames.length > 0
                                ? `You have no key named ${endpoint.secret_name}. This name came from the skill's documentation, not from your keys — pick one above or every call will fail.`
                                : `This name came from the skill's documentation. You have no keys stored yet, so add one named ${endpoint.secret_name} in the Add Key tab.`}
                            </p>
                          )}
                        </div>
                      )}
                    </div>
                  ))}

                  {skillError && (
                    <div className="modal-error">
                      <span className="error-icon">!</span>
                      <span>{skillError}</span>
                    </div>
                  )}
                </div>
              ) : (
                <>
                  <div
                    className={`skill-dropzone ${isDragging ? "dragging" : ""}`}
                    onDragOver={(e) => {
                      e.preventDefault();
                      setIsDragging(true);
                    }}
                    onDragLeave={() => setIsDragging(false)}
                    onDrop={handleDrop}
                  >
                    <SkillIcon />
                    {skillFiles.length > 0 ? (
                      <>
                        <p className="skill-dropzone-title">
                          Files added
                          <br />
                          Ready to install?
                        </p>
                        <p className="skill-dropzone-hint">
                          Or pick a different folder to replace them
                        </p>
                      </>
                    ) : (
                      <>
                        <p className="skill-dropzone-title">
                          Drag &amp; drop a skill folder or .zip here
                        </p>
                        <p className="skill-dropzone-hint">
                          The folder must contain a SKILL.md file
                        </p>
                      </>
                    )}
                    <div className="skill-dropzone-actions">
                      <button
                        type="button"
                        className="modal-btn secondary"
                        onClick={() => dirInputRef.current?.click()}
                        disabled={skillSubmitting}
                      >
                        Choose folder
                      </button>
                      <button
                        type="button"
                        className="modal-btn secondary"
                        onClick={() => zipInputRef.current?.click()}
                        disabled={skillSubmitting}
                      >
                        Choose .zip
                      </button>
                    </div>
                    <input
                      ref={dirInputRef}
                      type="file"
                      // @ts-expect-error non-standard directory-picker attributes
                      webkitdirectory=""
                      directory=""
                      multiple
                      style={{ display: "none" }}
                      onChange={handleDirInput}
                    />
                    <input
                      ref={zipInputRef}
                      type="file"
                      accept=".zip"
                      style={{ display: "none" }}
                      onChange={handleZipInput}
                    />
                  </div>

                  <div className="skill-or-divider">
                    <span>or install from GitHub</span>
                  </div>
                  <div className="form-group">
                    <label htmlFor="skill-github-url">GitHub URL</label>
                    <input
                      id="skill-github-url"
                      type="text"
                      value={githubUrl}
                      onChange={(e) => {
                        setGithubUrl(e.target.value);
                        if (e.target.value.trim()) {
                          setSkillFiles([]);
                          setSkillPreview(null);
                        }
                      }}
                      placeholder="https://github.com/org/repo/tree/main/path/to/skill"
                      disabled={skillSubmitting}
                      autoComplete="off"
                    />
                    <span className="form-hint">
                      Point to a directory containing a SKILL.md
                    </span>
                  </div>

                  {(skillPreview || skillInstalling) && (
                    <div className="skill-preview">
                      {skillPreview && (
                        <>
                          <div className="skill-preview-name">
                            {skillPreview.name || "(unnamed skill)"}
                          </div>
                          {skillPreview.description && (
                            <div className="skill-preview-desc">
                              {skillPreview.description}
                            </div>
                          )}
                        </>
                      )}
                      {skillInstalling ? (
                        <div className="skill-preview-progress" aria-live="polite">
                          <span className="loading-spinner small"></span>
                          <span>{installMessage}</span>
                        </div>
                      ) : (
                        skillPreview && (
                          <div className="skill-preview-files">
                            {skillPreview.fileCount} file
                            {skillPreview.fileCount !== 1 ? "s" : ""}
                          </div>
                        )
                      )}
                    </div>
                  )}

                  {skillError && (
                    <div className="modal-error">
                      <span className="error-icon">!</span>
                      <span>{skillError}</span>
                    </div>
                  )}
                </>
              )}
            </div>
            {skillInstalled === null && pendingSkill && (
              <div className="modal-footer">
                <button
                  type="button"
                  className="modal-btn secondary"
                  onClick={() => submitContract(null)}
                  disabled={skillSubmitting}
                  title={
                    httpRequestToolEnabled
                      ? "Keep the skill's instructions but create no tools; the model will fall back to http_request."
                      : "Keep the skill's instructions but create no tools; nothing will call its API."
                  }
                >
                  Skip tools
                </button>
                <button
                  type="button"
                  className="modal-btn primary"
                  onClick={() => submitContract(draftContract)}
                  disabled={skillSubmitting || !draftContract}
                >
                  {skillSubmitting ? (
                    <>
                      <span className="loading-spinner small"></span>
                      <span>Creating tools...</span>
                    </>
                  ) : (
                    "Approve & create tools"
                  )}
                </button>
              </div>
            )}
            {skillInstalled === null && !pendingSkill && (
              <div className="modal-footer">
                <button
                  type="button"
                  className="modal-btn secondary"
                  onClick={onClose}
                  disabled={skillSubmitting}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  className="modal-btn primary"
                  onClick={handleSkillInstall}
                  disabled={
                    skillSubmitting ||
                    (skillFiles.length === 0 && !githubUrl.trim())
                  }
                >
                  {skillSubmitting ? (
                    <>
                      <span className="loading-spinner small"></span>
                      <span>Installing...</span>
                    </>
                  ) : (
                    "Install skill"
                  )}
                </button>
              </div>
            )}
          </>
        )}

        {/* Key tab */}
        {activeTab === "key" && (
          <form onSubmit={handleKeySubmit}>
            <div className="modal-body">
              {keySaved !== null ? (
                <div className="modal-success">
                  <div className="success-icon">✓</div>
                  <p>Key saved!</p>
                  <p className="success-detail">{keySaved}</p>
                </div>
              ) : (
                <>
                  <div className="form-group">
                    <label htmlFor="key-name">Name</label>
                    <input
                      id="key-name"
                      type="text"
                      value={keyName}
                      onChange={(e) => setKeyName(e.target.value)}
                      placeholder="NVIDIA_API_KEY"
                      disabled={keySubmitting}
                      autoComplete="off"
                    />
                    <span className="form-hint">
                      The name a skill uses to reference this key
                    </span>
                  </div>
                  <div className="form-group">
                    <label htmlFor="key-value">API Key</label>
                    <input
                      id="key-value"
                      type="password"
                      value={keyValue}
                      onChange={(e) => setKeyValue(e.target.value)}
                      placeholder={
                        keyFromFile ? "Loaded from file" : "Paste your API key"
                      }
                      disabled={keySubmitting || keyFromFile}
                      autoComplete="off"
                    />
                    {keyFromFile ? (
                      <span className="form-hint">
                        Loaded from a dropped file.{" "}
                        <button
                          type="button"
                          className="link-btn"
                          onClick={clearKeyFile}
                          disabled={keySubmitting}
                        >
                          Enter manually instead
                        </button>
                      </span>
                    ) : (
                      <span className="form-hint">
                        Stored only for this session; never shown again
                      </span>
                    )}
                  </div>

                  <div className="skill-or-divider">
                    <span>or drop a key file</span>
                  </div>
                  <div
                    className={`skill-dropzone compact ${
                      keyDragging ? "dragging" : ""
                    }`}
                    onDragOver={(e) => {
                      e.preventDefault();
                      setKeyDragging(true);
                    }}
                    onDragLeave={() => setKeyDragging(false)}
                    onDrop={handleKeyDrop}
                  >
                    <KeyIcon />
                    <p className="skill-dropzone-title">
                      Drag &amp; drop a text file containing the key
                    </p>
                    <div className="skill-dropzone-actions">
                      <button
                        type="button"
                        className="modal-btn secondary"
                        onClick={() => keyFileInputRef.current?.click()}
                        disabled={keySubmitting}
                      >
                        Choose file
                      </button>
                    </div>
                    <input
                      ref={keyFileInputRef}
                      type="file"
                      accept=".txt,.key,.pem,text/plain"
                      style={{ display: "none" }}
                      onChange={handleKeyFileInput}
                    />
                  </div>

                  {keyError && (
                    <div className="modal-error">
                      <span className="error-icon">!</span>
                      <span>{keyError}</span>
                    </div>
                  )}
                </>
              )}
            </div>
            {keySaved === null && (
              <div className="modal-footer">
                <button
                  type="button"
                  className="modal-btn secondary"
                  onClick={onClose}
                  disabled={keySubmitting}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="modal-btn primary"
                  disabled={keySubmitting || !keyName.trim() || !keyValue.trim()}
                >
                  {keySubmitting ? (
                    <>
                      <span className="loading-spinner small"></span>
                      <span>Saving...</span>
                    </>
                  ) : (
                    "Save key"
                  )}
                </button>
              </div>
            )}
          </form>
        )}
      </div>
    </div>
  );
}
