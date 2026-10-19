#!/usr/bin/env bash
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

#
# Aevom Prerequisites Check
# Validates system requirements before building/running
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Minimum versions
MIN_DOCKER_VERSION="20.10"
MIN_DISK_GB=5

# Track errors
ERRORS=0

print_header() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Aevom Prerequisites Check${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

check_pass() {
    echo -e "  ${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "  ${RED}✗${NC} $1"
    ERRORS=$((ERRORS + 1))
}

check_warn() {
    echo -e "  ${YELLOW}⚠${NC} $1"
}

print_install_help() {
    echo ""
    echo -e "${YELLOW}Installation Instructions:${NC}"
    echo "$1"
    echo ""
}

# Compare dotted numeric version strings (returns 0 if $1 >= $2).
# awk rather than `sort -V`, which older macOS releases do not have.
version_gte() {
    awk -v a="$1" -v b="$2" 'BEGIN {
        na = split(a, x, "."); nb = split(b, y, ".")
        n = (na > nb) ? na : nb
        for (i = 1; i <= n; i++) {
            if ((x[i] + 0) > (y[i] + 0)) exit 0
            if ((x[i] + 0) < (y[i] + 0)) exit 1
        }
        exit 0
    }'
}

HOST_OS="$(uname -s)"

# Docker VM / engine memory below which the RDKit source build slows to one or
# two compile jobs and the full stack (ADMET's model ensemble included) is
# close to the limit.
MIN_DOCKER_MEM_GB=8

# ─────────────────────────────────────────────────────────────────────────────
# Check: Docker installed
# ─────────────────────────────────────────────────────────────────────────────
check_docker_installed() {
    echo -e "${BLUE}Checking Docker...${NC}"
    
    if ! command -v docker &> /dev/null; then
        check_fail "Docker is not installed"
        print_install_help "  Install Docker Desktop from: https://docs.docker.com/get-docker/
  
  On Linux, you can also install Docker Engine:
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker \$USER
    # Log out and back in for group changes to take effect"
        return 1
    fi
    
    check_pass "Docker is installed"
    return 0
}

# ─────────────────────────────────────────────────────────────────────────────
# Check: Docker daemon running
# ─────────────────────────────────────────────────────────────────────────────
check_docker_running() {
    if ! docker info &> /dev/null; then
        check_fail "Docker daemon is not running"
        print_install_help "  Start Docker:
    - Docker Desktop: Open the Docker Desktop application
    - Linux: sudo systemctl start docker"
        return 1
    fi
    
    check_pass "Docker daemon is running"
    return 0
}

# ─────────────────────────────────────────────────────────────────────────────
# Check: Docker version
# ─────────────────────────────────────────────────────────────────────────────
check_docker_version() {
    local version
    version=$(docker version --format '{{.Server.Version}}' 2>/dev/null | cut -d'-' -f1)
    
    if [ -z "$version" ]; then
        check_warn "Could not determine Docker version"
        return 0
    fi
    
    if version_gte "$version" "$MIN_DOCKER_VERSION"; then
        check_pass "Docker version $version (>= $MIN_DOCKER_VERSION required)"
    else
        check_fail "Docker version $version is too old (>= $MIN_DOCKER_VERSION required)"
        print_install_help "  Update Docker Desktop from: https://docs.docker.com/get-docker/
  Or update Docker Engine: sudo apt-get update && sudo apt-get install docker-ce"
        return 1
    fi
    return 0
}

# ─────────────────────────────────────────────────────────────────────────────
# Check: Docker Compose available
# ─────────────────────────────────────────────────────────────────────────────
check_docker_compose() {
    echo ""
    echo -e "${BLUE}Checking Docker Compose...${NC}"
    
    # Try 'docker compose' first (modern plugin)
    if docker compose version &> /dev/null; then
        local version
        version=$(docker compose version --short 2>/dev/null || docker compose version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
        check_pass "Docker Compose plugin v$version"
        echo "DOCKER_COMPOSE=docker compose" # For caller to capture
        return 0
    fi
    
    # Fall back to 'docker-compose' (standalone)
    if command -v docker-compose &> /dev/null; then
        local version
        version=$(docker-compose version --short 2>/dev/null || docker-compose version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
        check_pass "Docker Compose standalone v$version"
        echo "DOCKER_COMPOSE=docker-compose"
        return 0
    fi
    
    check_fail "Docker Compose is not available"
    print_install_help "  Docker Compose is included with Docker Desktop.
  
  If using Docker Engine on Linux, install the Compose plugin:
    sudo apt-get update
    sudo apt-get install docker-compose-plugin
  
  Or install standalone docker-compose:
    sudo curl -L \"https://github.com/docker/compose/releases/latest/download/docker-compose-\$(uname -s)-\$(uname -m)\" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose"
    return 1
}

# ─────────────────────────────────────────────────────────────────────────────
# Check: macOS host tools (Darwin only)
# ─────────────────────────────────────────────────────────────────────────────

# git, make and python3 all come from the Xcode Command Line Tools. Before they
# are installed /usr/bin/python3 and friends still exist -- as stubs that fail
# when run -- so `command -v` proves nothing and each tool has to be executed.
check_macos_host() {
    echo ""
    echo -e "${BLUE}Checking macOS host tools...${NC}"

    local product_version
    product_version="$(sw_vers -productVersion 2>/dev/null || echo unknown)"
    check_pass "macOS $product_version ($(uname -m))"

    if ! xcode-select -p &> /dev/null; then
        check_fail "Xcode Command Line Tools are not installed (needed for git, make and python3)"
        print_install_help "  Install them, then re-run 'make check':
    xcode-select --install"
        return 1
    fi

    local tool missing=""
    for tool in git make python3; do
        case "$tool" in
            python3) python3 -c 'import sys' &> /dev/null || missing="$missing $tool" ;;
            *) "$tool" --version &> /dev/null || missing="$missing $tool" ;;
        esac
    done
    if [ -n "$missing" ]; then
        check_fail "Command Line Tools are installed but these do not run:$missing"
        print_install_help "  Reinstall the Command Line Tools:
    sudo rm -rf \"\$(xcode-select -p)\"
    xcode-select --install"
        return 1
    fi

    check_pass "Xcode Command Line Tools present (git, make, python3 run)"
    return 0
}

# Docker Desktop only bind-mounts paths under its shared directories; a project
# elsewhere fails at 'up' with "mounts denied", after the images have built.
check_macos_file_sharing() {
    local project_root="$1"
    case "$project_root" in
        /Users/*|/Volumes/*|/private/*|/tmp/*|/var/folders/*)
            check_pass "Project is under a Docker Desktop shared path ($project_root)"
            ;;
        *)
            check_warn "Project is outside Docker Desktop's default shared paths: $project_root"
            echo "    Move it under your home directory, or add its parent in"
            echo "    Docker Desktop > Settings > Resources > File sharing."
            ;;
    esac
}

# ─────────────────────────────────────────────────────────────────────────────
# Check: Docker engine architecture and resources
# ─────────────────────────────────────────────────────────────────────────────
check_docker_resources() {
    echo ""
    echo -e "${BLUE}Checking Docker resources...${NC}"

    local info arch ncpu mem_bytes mem_gb
    info="$(docker info --format '{{.Architecture}} {{.NCPU}} {{.MemTotal}}' 2>/dev/null)" || info=""
    if [ -z "$info" ]; then
        check_warn "Could not read the Docker engine's architecture and resources"
        return 0
    fi
    # shellcheck disable=SC2086
    set -- $info
    arch="$1"; ncpu="$2"; mem_bytes="$3"
    mem_gb=$(awk -v b="$mem_bytes" 'BEGIN { printf "%.1f", b / 1024 / 1024 / 1024 }')

    case "$arch" in
        x86_64|amd64)
            check_pass "Docker engine architecture: $arch"
            ;;
        aarch64|arm64)
            check_pass "Docker engine architecture: $arch (images build natively)"
            ;;
        *)
            check_warn "Docker engine architecture '$arch' is untested"
            ;;
    esac

    if awk -v m="$mem_gb" -v min="$MIN_DOCKER_MEM_GB" 'BEGIN { exit !(m + 0.5 >= min) }'; then
        check_pass "Docker has ${mem_gb}GB memory and $ncpu CPUs"
    else
        check_warn "Docker has only ${mem_gb}GB memory ($ncpu CPUs); ${MIN_DOCKER_MEM_GB}GB or more is recommended"
        if [ "$HOST_OS" = "Darwin" ]; then
            echo "    Raise it in Docker Desktop > Settings > Resources > Memory limit."
        fi
        echo "    The RDKit build still succeeds with less, but compiles fewer files in"
        echo "    parallel and takes longer."
    fi
    return 0
}

# ─────────────────────────────────────────────────────────────────────────────
# Check: Host ports the stack publishes
# ─────────────────────────────────────────────────────────────────────────────

# A port held by another process makes 'up' fail only after every image has
# built. Ports Docker itself publishes are not conflicts -- that is a previous
# run of this stack.
check_host_ports() {
    # macOS always has lsof; minimal Linux installs often do not, and there the
    # check is skipped rather than reported.
    command -v lsof &> /dev/null || return 0

    echo ""
    echo -e "${BLUE}Checking host ports...${NC}"

    local port listener busy=""
    for port in 5180 6080 8080 "${REDIS_HOST_PORT:-6379}"; do
        listener="$(lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | awk 'NR == 2 { print $1 }')"
        [ -n "$listener" ] || continue
        case "$listener" in
            com.docke*|docker*|vpnkit*|orbstack*|limactl*) continue ;;
        esac
        busy="$busy $port($listener)"
    done

    if [ -n "$busy" ]; then
        check_warn "Ports already in use by other processes:$busy"
        echo "    Stop those processes before 'make up' / 'make dev'. For redis on 6379"
        echo "    (often Homebrew's: brew services stop redis), you can instead set"
        echo "    REDIS_HOST_PORT to a free port in .env."
    else
        check_pass "Ports 5180, 6080, 8080 and ${REDIS_HOST_PORT:-6379} are free"
    fi
    return 0
}

# ─────────────────────────────────────────────────────────────────────────────
# Check: LLM API key files (all optional, at least one required)
# ─────────────────────────────────────────────────────────────────────────────

# Portal URL and human description per key, keyed by label.
key_url() {
    case "$1" in
        API_KEY_NVINF) echo "https://inference.nvidia.com/" ;;
        API_KEY_ANTHROPIC) echo "https://console.anthropic.com/settings/keys" ;;
        API_KEY_OPENAI) echo "https://platform.openai.com/api-keys" ;;
        *)             echo "https://build.nvidia.com/" ;;
    esac
}

key_description() {
    case "$1" in
        API_KEY_NVINF) echo "inference.nvidia.com (inference-api.nvidia.com endpoint)" ;;
        API_KEY_ANTHROPIC) echo "Anthropic (api.anthropic.com endpoint)" ;;
        API_KEY_OPENAI) echo "OpenAI (api.openai.com endpoint)" ;;
        *)             echo "build.nvidia.com (integrate.api.nvidia.com endpoint)" ;;
    esac
}

# Issued-by-portal prefixes. Distinct and stable, which is what makes a swapped
# pair of key files detectable without calling either endpoint.
key_prefix() {
    case "$1" in
        API_KEY_NVINF) echo "sk-" ;;
        API_KEY_ANTHROPIC) echo "sk-ant-" ;;
        API_KEY_OPENAI) echo "sk-" ;;
        *)             echo "nvapi-" ;;
    esac
}

# The file must exist, and must not be the empty directory Docker's short-form
# bind mount silently creates when the file was missing during a prior
# 'docker compose up'.
# Args: $1=label, $2=path, $3=help text shown when the file is absent
check_key_file_present() {
    local label="$1"
    local path="$2"
    local missing_help="$3"
    local url description
    url="$(key_url "$label")"
    description="$(key_description "$label")"

    if [ -d "$path" ]; then
        check_fail "$label exists but is a DIRECTORY (likely auto-created by Docker)"
        print_install_help "  Remove the stub directory and replace it with your key file:
    rm -rf $path
    echo 'your-${label}-key' > $path

  This usually happens when '$label' was missing during 'docker compose up'.
  Get a key for $description from: $url"
        return 1
    fi

    if [ ! -f "$path" ]; then
        check_fail "$label file not found ($description)"
        print_install_help "$missing_help"
        return 1
    fi

    return 0
}

# Full validation, for the one key the active model will actually read.
# Args: $1=label, $2=path
check_key_file() {
    local label="$1"
    local path="$2"
    local url description prefix contents
    url="$(key_url "$label")"
    description="$(key_description "$label")"
    prefix="$(key_prefix "$label")"

    check_key_file_present "$label" "$path" "  Create the API key file:
    1. Get an API key from: $url
    2. Save it to: $path

    echo 'your-${label}-key' > $path" || return 1

    if [ ! -s "$path" ] || [ -z "$(tr -d '[:space:]' < "$path")" ]; then
        check_fail "$label file is empty ($description)"
        print_install_help "  Add your API key to: $path
    1. Get an API key from: $url
    2. Paste the key into the file (no trailing newline required)"
        return 1
    fi

    # A warning rather than a failure: a prefix that stops matching one day is
    # a format change, and that should not block a deployment whose key works.
    contents="$(tr -d '[:space:]' < "$path")"
    if [ "${contents#"$prefix"}" = "$contents" ]; then
        check_warn "$label does not start with '$prefix' — are API_KEY_NV and API_KEY_NVINF swapped?"
        return 0
    fi

    check_pass "$label file exists and has content"
    return 0
}

# Resolve AEVOM_MODEL the way Docker Compose will resolve it.
#
# Compose auto-loads .env and lets the shell environment override it. Make does
# not export .env at all, so a preflight that reads only the shell validates the
# key for a model the stack is not going to run -- which passed a build.nvidia.com
# check and then failed at runtime against inference-api. Echoes an empty string
# when neither source sets it, leaving the default to the registry.
#
# Parsed, not sourced: .env carries secrets and arbitrary values, and none of it
# should execute here.
resolve_active_model() {
    local project_root="$1"
    local from_env

    if [ -n "${AEVOM_MODEL:-}" ]; then
        printf '%s' "$AEVOM_MODEL"
        return 0
    fi

    if [ -f "$project_root/.env" ]; then
        from_env="$(sed -n 's/^[[:space:]]*AEVOM_MODEL[[:space:]]*=[[:space:]]*\([^#]*\).*/\1/p' \
            "$project_root/.env" | tail -1)"
        from_env="$(printf '%s' "$from_env" | sed -e 's/[[:space:]]*$//' -e 's/^["'\'']//' -e 's/["'\'']$//')"
        printf '%s' "$from_env"
    fi
}

# Ask the registry which key file a model needs, instead of keeping a second
# copy of the mapping here that nothing keeps honest.
#
# backend/api/llm/config.py imports only the standard library, so it loads on the
# host with nothing installed and no container running. Taking DEFAULT_MODEL_ID
# from the same place means the fallback cannot drift from the code either.
# Prints the resolved model id and its api_key_file, one per line.
resolve_model_key_file() {
    local project_root="$1"
    local model_id="$2"

    # config.py is loaded by path, not imported as api.llm.config. This runs on
    # the host before anything is built, where the backend dependencies do not
    # exist, and importing the package would execute api/llm/__init__.py and
    # fail on httpx. The file itself imports only the standard library.
    (cd "$project_root" && python3 -c '
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("aevom_llm_config", "backend/api/llm/config.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

model_id = sys.argv[1] or config.DEFAULT_MODEL_ID
if model_id not in config.MODELS:
    sys.exit("Unknown AEVOM_MODEL=%r. Registered models: %s" % (model_id, ", ".join(config.MODELS)))
print(model_id)
print(config.MODELS[model_id].api_key_file)
' "$model_id")
}

# Every key file the compose stacks bind-mount. All are optional -- an empty file
# means that provider is not configured -- but at least one LLM key must hold a
# key, or no model can be called.
LLM_KEY_LABELS="API_KEY_NV API_KEY_NVINF API_KEY_ANTHROPIC API_KEY_OPENAI"

key_has_content() {
    [ -f "$1" ] && [ -n "$(tr -d '[:space:]' < "$1")" ]
}

# Ask api.anthropic.com whether the key works. The key travels on stdin as a
# header file, never in argv, so it does not show up in `ps` or in this output.
# Prints the HTTP status, or 000 when the host could not be reached.
anthropic_key_status() {
    local path="$1"
    command -v curl &> /dev/null || { echo "000"; return 0; }
    printf 'x-api-key: %s\nanthropic-version: 2023-06-01\n' "$(tr -d '[:space:]' < "$path")" \
        | curl -sS -o /dev/null -w '%{http_code}' --max-time 10 -H @- \
            "https://api.anthropic.com/v1/models?limit=1" 2>/dev/null \
        || echo "000"
}

check_anthropic_key() {
    local path="$1" contents status
    contents="$(tr -d '[:space:]' < "$path")"
    if [ "${contents#sk-ant-}" = "$contents" ]; then
        check_warn "API_KEY_ANTHROPIC does not start with 'sk-ant-'"
    fi
    status="$(anthropic_key_status "$path")"
    case "$status" in
        200)
            check_pass "API_KEY_ANTHROPIC accepted by api.anthropic.com (Claude models will be listed)"
            ;;
        401|403)
            check_fail "API_KEY_ANTHROPIC was rejected by api.anthropic.com (HTTP $status)"
            print_install_help "  Replace the key in: $path
  Get a key from: $(key_url API_KEY_ANTHROPIC)"
            return 1
            ;;
        *)
            # Not fatal: the backend lists models at runtime and retries.
            check_warn "Could not verify API_KEY_ANTHROPIC (HTTP $status); network or proxy issue?"
            ;;
    esac
    return 0
}

# Same as anthropic_key_status, against api.openai.com.
openai_key_status() {
    local path="$1"
    command -v curl &> /dev/null || { echo "000"; return 0; }
    printf 'Authorization: Bearer %s\n' "$(tr -d '[:space:]' < "$path")" \
        | curl -sS -o /dev/null -w '%{http_code}' --max-time 10 -H @- \
            "https://api.openai.com/v1/models" 2>/dev/null \
        || echo "000"
}

check_openai_key() {
    local path="$1" contents status
    contents="$(tr -d '[:space:]' < "$path")"
    if [ "${contents#sk-ant-}" != "$contents" ]; then
        check_warn "API_KEY_OPENAI looks like an Anthropic key — should it be in API_KEY_ANTHROPIC?"
    elif [ "${contents#sk-}" = "$contents" ]; then
        check_warn "API_KEY_OPENAI does not start with 'sk-'"
    fi
    status="$(openai_key_status "$path")"
    case "$status" in
        200)
            check_pass "API_KEY_OPENAI accepted by api.openai.com (GPT models will be listed)"
            ;;
        401|403)
            check_fail "API_KEY_OPENAI was rejected by api.openai.com (HTTP $status)"
            print_install_help "  Replace the key in: $path
  Get a key from: $(key_url API_KEY_OPENAI)"
            return 1
            ;;
        *)
            check_warn "Could not verify API_KEY_OPENAI (HTTP $status); network or proxy issue?"
            ;;
    esac
    return 0
}

# Validate every configured key, require at least one, and report which
# provider the active model resolves to.
check_api_keys() {
    echo ""
    echo -e "${BLUE}Checking LLM API keys...${NC}"

    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local project_root
    project_root="$(dirname "$script_dir")"

    if ! command -v python3 &> /dev/null; then
        check_fail "python3 not found — needed to read the model registry"
        print_install_help "  Install python3 (3.10+), then re-run. It is used only to read
  backend/api/llm/config.py and learn which API key the active model needs."
        return 1
    fi

    local rc=0 label path configured=""

    # Compose bind-mounts every key file with create_host_path:false, so each
    # path has to exist even for a provider nobody uses. An empty placeholder
    # (gitignored, like the keys) satisfies the mount and reads as "absent".
    for label in $LLM_KEY_LABELS; do
        path="$project_root/$label"
        if [ ! -e "$path" ]; then
            : > "$path"
            echo -e "  ${BLUE}•${NC} Created empty placeholder $label ($(key_description "$label") not configured)"
        fi
        check_key_file_present "$label" "$path" "" || { rc=1; continue; }
        if key_has_content "$path"; then
            configured="$configured $label"
        fi
    done
    [ "$rc" -eq 0 ] || return 1

    if [ -z "$configured" ]; then
        check_fail "No LLM API key configured"
        print_install_help "  Put at least one key in its file at the repo root:
    API_KEY_ANTHROPIC  $(key_description API_KEY_ANTHROPIC)  $(key_url API_KEY_ANTHROPIC)
    API_KEY_OPENAI     $(key_description API_KEY_OPENAI)  $(key_url API_KEY_OPENAI)
    API_KEY_NV         $(key_description API_KEY_NV)  $(key_url API_KEY_NV)
    API_KEY_NVINF      $(key_description API_KEY_NVINF)  $(key_url API_KEY_NVINF)

  API_KEY_NV also serves the NIM-backed tools (OpenFold3, Boltz-2, ...)."
        return 1
    fi

    for label in $configured; do
        path="$project_root/$label"
        case "$label" in
            API_KEY_ANTHROPIC) check_anthropic_key "$path" || rc=1 ;;
            API_KEY_OPENAI) check_openai_key "$path" || rc=1 ;;
            *)
                if [ "$label" = "API_KEY_NVINF" ] \
                    && [ "$(tr -d '[:space:]' < "$path" | cut -c1-7)" = "sk-ant-" ]; then
                    check_warn "API_KEY_NVINF looks like an Anthropic key — should it be in API_KEY_ANTHROPIC?"
                fi
                check_key_file "$label" "$path" || rc=1
                ;;
        esac
    done

    if ! key_has_content "$project_root/API_KEY_NV"; then
        check_warn "API_KEY_NV is empty: NIM tools (OpenFold3, Boltz-2, MSA search, GenMol, RFdiffusion, ProteinMPNN) are disabled"
    fi

    # Which provider the active model needs. A model whose key is absent is not
    # an error: the backend falls back to one that has a key.
    local active_model lookup rc_lookup=0 key_file active_label
    active_model="$(resolve_active_model "$project_root")"
    case "$active_model" in
        anthropic/*)
            key_file="/run/secrets/anthropic_api_key"
            ;;
        openai/*)
            key_file="/run/secrets/openai_api_key"
            ;;
        *)
            lookup="$(resolve_model_key_file "$project_root" "$active_model" 2>&1)" || rc_lookup=$?
            if [ "$rc_lookup" -ne 0 ]; then
                check_fail "Could not resolve the active model"
                print_install_help "  $lookup

  Set AEVOM_MODEL to a registered model, either in your shell or in .env.
  Claude models are named anthropic/<model-id>, e.g. anthropic/claude-opus-5-5,
  and OpenAI models openai/<model-id>, e.g. openai/gpt-6.1-sol."
                return 1
            fi
            active_model="$(printf '%s\n' "$lookup" | sed -n '1p')"
            key_file="$(printf '%s\n' "$lookup" | sed -n '2p')"
            ;;
    esac

    case "$key_file" in
        */nvidia_api_key_nvinf) active_label="API_KEY_NVINF" ;;
        */nvidia_api_key) active_label="API_KEY_NV" ;;
        */anthropic_api_key) active_label="API_KEY_ANTHROPIC" ;;
        */openai_api_key) active_label="API_KEY_OPENAI" ;;
        *)
            check_fail "Model '$active_model' declares an unrecognised api_key_file: $key_file"
            print_install_help "  Every ModelConfig in backend/api/llm/config.py must point at one of
  /run/secrets/nvidia_api_key, /run/secrets/nvidia_api_key_nvinf,
  /run/secrets/anthropic_api_key or /run/secrets/openai_api_key."
            return 1
            ;;
    esac

    if key_has_content "$project_root/$active_label"; then
        echo -e "  ${BLUE}Active model:${NC} $active_model → $active_label [$(key_description "$active_label")]"
    else
        check_warn "Active model '$active_model' needs $active_label, which is empty; the backend will fall back to a model with a key"
    fi

    return $rc
}

# ─────────────────────────────────────────────────────────────────────────────
# Check: Git submodules populated
# ─────────────────────────────────────────────────────────────────────────────

# third_party/bionemo-agent-toolkit carries the NIM SKILL.md prose that
# api/skills/binding.py loads at startup. A fresh `git clone` without --recursive
# leaves the directory empty, and the failure mode is silent: the skills simply
# do not appear in the catalog, so the model never learns the NIM tools exist.
check_submodules() {
    echo ""
    echo -e "${BLUE}Checking git submodules...${NC}"

    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local project_root
    project_root="$(dirname "$script_dir")"

    local nim_skills="$project_root/third_party/bionemo-agent-toolkit/nim-skills"

    if [ ! -d "$nim_skills" ]; then
        check_fail "bionemo-agent-toolkit submodule is not checked out"
        print_install_help "  Populate the submodule (works for a git clone and for a source tarball):
    make submodules

  Without it the NIM skills (rfdiffusion, proteinmpnn, genmol, msa-search) are missing
  from the skill catalog and the model will not know the NIM tools exist."
        return 1
    fi

    check_pass "bionemo-agent-toolkit submodule is checked out"
    return 0
}

# ─────────────────────────────────────────────────────────────────────────────
# Check: Disk space
# ─────────────────────────────────────────────────────────────────────────────
check_disk_space() {
    echo ""
    echo -e "${BLUE}Checking disk space...${NC}"
    
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local project_root
    project_root="$(dirname "$script_dir")"
    
    # Get available space in GB
    local available_gb
    if command -v df &> /dev/null; then
        # Works on Linux and macOS
        available_gb=$(df -BG "$project_root" 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')
        
        if [ -z "$available_gb" ]; then
            # macOS df output format differs
            available_gb=$(df -g "$project_root" 2>/dev/null | tail -1 | awk '{print $4}')
        fi
    fi
    
    if [ -z "$available_gb" ]; then
        check_warn "Could not determine available disk space"
        return 0
    fi
    
    if [ "$available_gb" -ge "$MIN_DISK_GB" ]; then
        check_pass "${available_gb}GB available (>= ${MIN_DISK_GB}GB required for RDKit build)"
    else
        check_fail "Only ${available_gb}GB available (need >= ${MIN_DISK_GB}GB for RDKit build)"
        print_install_help "  Free up disk space before building.
  The RDKit container build requires approximately 5GB."
        return 1
    fi
    return 0
}

# ─────────────────────────────────────────────────────────────────────────────
# Check: GPU Support (optional - for Vina GPU and DevContainer)
# ─────────────────────────────────────────────────────────────────────────────
check_gpu_support() {
    echo ""
    echo -e "${BLUE}Checking GPU support (optional)...${NC}"

    if [ "$HOST_OS" = "Darwin" ]; then
        echo -e "  ${BLUE}•${NC} Not applicable on macOS: Docker has no NVIDIA GPU access there, so"
        echo "    Aevom runs CPU-only (make up / make dev). The *-gpu targets are Linux-only."
        return 0
    fi

    local engine_arch
    engine_arch="$(docker info --format '{{.Architecture}}' 2>/dev/null || true)"

    local gpu_ready=true
    
    # Check 1: NVIDIA driver installed (nvidia-smi available)
    if ! command -v nvidia-smi &> /dev/null; then
        check_warn "NVIDIA driver not found (nvidia-smi not available)"
        gpu_ready=false
    else
        local driver_version
        driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        if [ -n "$driver_version" ]; then
            check_pass "NVIDIA driver $driver_version installed"
            # The arm64 GPU image is built with CUDA 12.9 (GB10 needs it).
            case "$engine_arch" in
                aarch64|arm64)
                    if ! version_gte "$driver_version" "575"; then
                        check_warn "The arm64 GPU image uses CUDA 12.9, which needs driver 575 or newer"
                        gpu_ready=false
                    fi
                    ;;
            esac
        else
            check_warn "NVIDIA driver installed but could not query version"
            gpu_ready=false
        fi
    fi
    
    # Check 2: nvidia-container-toolkit (test with Docker)
    if [ "$gpu_ready" = true ] && command -v docker &> /dev/null && docker info &> /dev/null; then
        echo -e "  ${BLUE}Testing Docker GPU access...${NC}"
        if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
            check_pass "nvidia-container-toolkit is working"
        else
            check_warn "nvidia-container-toolkit not configured"
            gpu_ready=false
        fi
    fi
    
    # Summary
    if [ "$gpu_ready" = true ]; then
        check_pass "GPU acceleration available (Vina GPU, DevContainer)"
    else
        check_warn "GPU acceleration not available"
        echo ""
        echo -e "  ${YELLOW}GPU is optional but recommended for:${NC}"
        echo "    - Vina GPU molecular docking (make dev-gpu)"
        echo "    - DevContainer development environment"
        echo ""
        echo -e "  ${YELLOW}To enable GPU support:${NC}"
        echo "    1. Install NVIDIA driver: https://www.nvidia.com/Download/index.aspx"
        echo "    2. Install nvidia-container-toolkit:"
        echo "       curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
        echo "       curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \\"
        echo "         sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \\"
        echo "         sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
        echo "       sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
        echo "       sudo nvidia-ctk runtime configure --runtime=docker"
        echo "       sudo systemctl restart docker"
        echo ""
        echo -e "  ${YELLOW}Verify with:${NC}"
        echo "       docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi"
    fi
    
    # GPU is optional, so don't fail the check
    return 0
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
main() {
    print_header

    local project_root
    project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

    # Run all checks
    if [ "$HOST_OS" = "Darwin" ]; then
        check_macos_host || true
        echo ""
    fi
    check_docker_installed || true
    if command -v docker &> /dev/null; then
        check_docker_running || true
        check_docker_version || true
    fi
    check_docker_compose || true
    if command -v docker &> /dev/null && docker info &> /dev/null; then
        check_docker_resources || true
        if [ "$HOST_OS" = "Darwin" ]; then
            check_macos_file_sharing "$project_root"
        fi
    fi
    check_host_ports || true
    check_api_keys || true
    check_submodules || true
    check_disk_space || true
    check_gpu_support || true
    
    # Summary
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    if [ $ERRORS -eq 0 ]; then
        echo -e "${GREEN}  ✓ All prerequisites satisfied!${NC}"
        echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
        exit 0
    else
        echo -e "${RED}  ✗ $ERRORS prerequisite(s) not satisfied${NC}"
        echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
        echo "Please fix the issues above and try again."
        echo ""
        exit 1
    fi
}

main "$@"
