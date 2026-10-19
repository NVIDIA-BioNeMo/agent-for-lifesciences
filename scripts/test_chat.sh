#!/bin/bash
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
# test_chat.sh - Quick test and trace workflow for Aevom
#
# Usage:
#   ./scripts/test_chat.sh "Your test message here"
#   ./scripts/test_chat.sh "Generate 3D coordinates for benzene" --show-reasoning
#
# This script:
#   1. Ensures network connectivity to Aevom services
#   2. Sends a chat request to the backend
#   3. Queries Phoenix for the resulting trace
#   4. Displays formatted trace output
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BACKEND_URL="${BACKEND_URL:-http://aevom-backend-1:8000}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
MESSAGE=""
SHOW_REASONING=""
SPAN_LIMIT=15

while [[ $# -gt 0 ]]; do
    case $1 in
        --show-reasoning|-r)
            SHOW_REASONING="--show-reasoning"
            shift
            ;;
        --limit|-l)
            SPAN_LIMIT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 \"message\" [options]"
            echo ""
            echo "Options:"
            echo "  --show-reasoning, -r  Show full LLM reasoning content"
            echo "  --limit N, -l N       Number of spans to fetch (default: 15)"
            echo "  --help, -h            Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 \"What is the molecular weight of aspirin?\""
            echo "  $0 \"Generate 3D coordinates for caffeine\" --show-reasoning"
            exit 0
            ;;
        *)
            MESSAGE="$1"
            shift
            ;;
    esac
done

if [ -z "$MESSAGE" ]; then
    echo -e "${RED}Error: No message provided${NC}"
    echo "Usage: $0 \"Your test message here\""
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Aevom Test Chat Workflow${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Ensure network connectivity
echo -e "${YELLOW}[1/4] Ensuring network connectivity...${NC}"
docker network connect aevom_default "$(hostname)" 2>/dev/null || true

# Verify backend is reachable
if ! curl -s "${BACKEND_URL}/api/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: Cannot reach backend at ${BACKEND_URL}${NC}"
    echo "Make sure services are running with: make dev"
    exit 1
fi
echo -e "${GREEN}✓ Connected to backend${NC}"

# Step 2: Send chat request
echo ""
echo -e "${YELLOW}[2/4] Sending chat request...${NC}"
echo -e "   Message: ${MESSAGE}"
echo ""

# Capture start time for trace filtering
START_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Chat needs a backend-minted session and the owner token it is bound to
SESSION_JSON=$(curl -s -X POST "${BACKEND_URL}/api/session/create")
SESSION_ID=$(echo "$SESSION_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['session_id'])")
OWNER_TOKEN=$(echo "$SESSION_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['owner_token'])")

# Send request and capture output
RESPONSE=$(curl -s -X POST "${BACKEND_URL}/api/chat" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${OWNER_TOKEN}" \
    -d "{\"messages\": [{\"role\": \"user\", \"content\": \"${MESSAGE}\"}], \"max_tokens\": ${MAX_TOKENS}, \"session_id\": \"${SESSION_ID}\"}" \
    2>&1)

# Brief pause to ensure traces are flushed to Phoenix
sleep 1

echo -e "${GREEN}✓ Request completed${NC}"

# Step 3: Show SSE response summary
echo ""
echo -e "${YELLOW}[3/4] Response summary:${NC}"

# Parse SSE response and show key events
echo "$RESPONSE" | while IFS= read -r line; do
    # Skip empty lines and "data: " prefix
    line="${line#data: }"
    [ -z "$line" ] && continue
    
    # Try to parse as JSON
    if echo "$line" | python3 -c "import json,sys; json.load(sys.stdin)" 2>/dev/null; then
        # Extract event type
        event_type=$(echo "$line" | python3 -c "
import json, sys
data = json.load(sys.stdin)
if 'reasoning' in data:
    print(f\"   [Iteration {data.get('iteration', '?')}] Reasoning ({len(data['reasoning'])} chars)\")
elif 'tool_call_start' in data:
    tc = data['tool_call_start']
    print(f\"   → Tool: {tc['name']} - {tc.get('summary', '')}\")
elif 'tool_call_end' in data:
    tc = data['tool_call_end']
    status = '✓' if tc.get('success', False) else '✗'
    print(f\"   {status} Tool: {tc['name']} completed\")
elif 'content' in data:
    content = data['content'][:100]
    if len(data['content']) > 100:
        content += '...'
    print(f\"   Content: {content}\")
" 2>/dev/null) && echo "$event_type"
    fi
done

# Step 4: Query and display traces
echo ""
echo -e "${YELLOW}[4/4] Querying traces from Phoenix...${NC}"

if [ -f "${SCRIPT_DIR}/phoenix_query.py" ]; then
    python3 "${SCRIPT_DIR}/phoenix_query.py" recent "$SPAN_LIMIT" $SHOW_REASONING
else
    echo -e "${RED}Warning: phoenix_query.py not found${NC}"
    echo "Run the raw GraphQL query manually or check scripts/ directory"
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Test complete!${NC}"
echo -e "${BLUE}========================================${NC}"
