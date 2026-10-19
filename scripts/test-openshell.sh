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

# Escape suite for python_execute under OpenShell. See
# scripts/openshell_escape_suite.py for what is checked and why.
#
# Needs the stack running with the OpenShell overlay (`make up` or `make dev`,
# which use it by default); it uses the backend container because that is where
# the SDK, the gateway route and the policy environment already are.
set -euo pipefail

cd "$(dirname "$0")/.."

backend=$(docker ps -q --filter label=com.docker.compose.service=backend | head -1)
if [ -z "$backend" ]; then
    echo "No backend container is running. Start the stack with make up or make dev." >&2
    exit 1
fi
if [ "$(docker exec "$backend" printenv PYEXEC_MODE || true)" != "openshell" ]; then
    echo "The backend is not in PYEXEC_MODE=openshell. Restart it with make up or make dev, without PYEXEC_MODE=direct." >&2
    exit 1
fi

ip_of() {
    local id
    id=$(docker ps -q --filter "label=com.docker.compose.service=$1" | head -1)
    [ -n "$id" ] && docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$id" || true
}

targets=$(printf '{"backend": "%s:%s", "redis": "%s", "postgres": "%s", "rdkit": "%s", "gateway": "%s:8090"}' \
    "$(docker exec "$backend" printenv OPENSHELL_BACKEND_HOST)" \
    "$(docker exec "$backend" printenv OPENSHELL_BACKEND_PORT)" \
    "$(ip_of redis)" "$(ip_of postgres)" "$(ip_of rdkit)" "$(ip_of openshell-gateway)")

# Run detached and collect a log: the gRPC client detaches the process from
# `docker exec`'s stdio, so its output would otherwise be lost.
docker cp scripts/openshell_escape_suite.py "$backend":/tmp/openshell_escape_suite.py >/dev/null
docker exec "$backend" rm -f /tmp/openshell_escape_suite.log
docker exec -d -e ESCAPE_TARGETS="$targets" -w /app "$backend" sh -c \
    'python -u /tmp/openshell_escape_suite.py > /tmp/openshell_escape_suite.log 2>&1; echo "exit=$?" >> /tmp/openshell_escape_suite.log'

for _ in $(seq 1 120); do
    docker exec "$backend" grep -q '^exit=' /tmp/openshell_escape_suite.log 2>/dev/null && break
    sleep 2
done
docker exec "$backend" cat /tmp/openshell_escape_suite.log
docker exec "$backend" grep -q '^exit=0$' /tmp/openshell_escape_suite.log
