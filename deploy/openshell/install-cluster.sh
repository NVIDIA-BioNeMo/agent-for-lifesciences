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
# Install the cluster prerequisites for pyexec.mode=openshell: everything the
# Aevom chart expects but does not install. Run it once per cluster, as a
# cluster admin, before the first deploy in openshell mode.
#
# What it does
# ------------
#   1. Applies the Agent Sandbox CRDs and controller, pinned by version and
#      sha256. OpenShell creates sandboxes through this API.
#   2. Creates the OpenShell namespace.
#   3. Copies the gitlab-registry pull secret from the Aevom namespace, so
#      sandbox pods can pull pyexec-sandbox. The deploy job refreshes it.
#   4. Installs the OpenShell chart. Its certgen hook writes the client
#      certificate openshell-client-tls that the deploy job copies. The
#      gateway requires that certificate at the TLS handshake and, with
#      allowUnauthenticatedUsers, treats any caller presenting it as a user:
#      chart 0.1.2 has no other API identity short of OIDC.
#   5. Lets the CI deploy identity read that certificate and refresh the pull
#      secret, and nothing else in the namespace.
#
# Every step is idempotent. Sandboxes need nodes on Linux 6.2 or newer, which
# the script checks but cannot fix. User namespaces stay off: idmapped tmpfs
# mounts, which service account tokens need, arrived in 6.3.
#
# Usage
# -----
#     deploy/openshell/install-cluster.sh            # dry run, changes nothing
#     deploy/openshell/install-cluster.sh --apply
#
# Needs kubectl, helm 3.8+, jq and curl. The alpine/k8s image the deploy job
# uses has all four.

set -euo pipefail

OPENSHELL_NAMESPACE="${OPENSHELL_NAMESPACE:-openshell}"
AEVOM_NAMESPACE="${AEVOM_NAMESPACE:-aevom}"
DEPLOY_SA_NAMESPACE="${DEPLOY_SA_NAMESPACE:-gitlab-ci}"
DEPLOY_SA_NAME="${DEPLOY_SA_NAME:-gitlab-deployer}"

# The OpenShell chart must match the openshell SDK pinned in the backend.
OPENSHELL_CHART="oci://ghcr.io/nvidia/openshell/helm-chart"
OPENSHELL_CHART_VERSION="0.1.2"
AGENT_SANDBOX_VERSION="v1.0.6"
AGENT_SANDBOX_SHA256="b995db5839f2bdca3a24b8bbefdec405df662d6dd223eec8f5dc74098c974231"

APPLY=false
for argument in "$@"; do
    case "$argument" in
        --apply) APPLY=true ;;
        -h|--help)
            sed -n '17,44p' "$0" | sed 's|^# \{0,1\}||'
            exit 0
            ;;
        *)
            echo "Unknown argument: $argument (try --help)" >&2
            exit 2
            ;;
    esac
done

if $APPLY; then
    DRY_RUN=()
    echo "Applying to $(kubectl config current-context)."
else
    DRY_RUN=(--dry-run=server)
    echo "Dry run against $(kubectl config current-context); nothing is changed. Pass --apply to install."
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo
echo "== 0. Node kernels"
# The sandbox refuses to start below Landlock ABI v3, whatever the policy's
# landlock.compatibility says: "sandbox self-protection requires Landlock ABI
# v3 or newer". ABI v3 arrived in Linux 6.2. The gateway runs on older nodes;
# sandboxes scheduled there fail at startup.
old_nodes=$(kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name} {.status.nodeInfo.kernelVersion}{"\n"}{end}' \
    | awk '{ split($2, v, /[.-]/); if (v[1] < 6 || (v[1] == 6 && v[2] < 2)) print "    " $1 " " $2 }')
if [ -n "$old_nodes" ]; then
    echo "WARNING: sandboxes need Linux 6.2 or newer (Landlock ABI v3). These nodes are older:"
    echo "$old_nodes"
else
    echo "Every node runs Linux 6.2 or newer."
fi

echo
echo "== 1. Agent Sandbox $AGENT_SANDBOX_VERSION"
curl -fsSL -o "$work/agent-sandbox.yaml" \
    "https://github.com/kubernetes-sigs/agent-sandbox/releases/download/$AGENT_SANDBOX_VERSION/sandbox.yaml"
actual_sha256="$(sha256sum "$work/agent-sandbox.yaml" | cut -d' ' -f1)"
if [ "$actual_sha256" != "$AGENT_SANDBOX_SHA256" ]; then
    echo "ERROR: agent-sandbox $AGENT_SANDBOX_VERSION manifest has sha256 $actual_sha256," >&2
    echo "       expected $AGENT_SANDBOX_SHA256." >&2
    exit 1
fi
if $APPLY; then
    kubectl apply --server-side --field-manager=aevom-openshell-install -f "$work/agent-sandbox.yaml"
    kubectl wait --for=condition=Established --timeout=60s crd/sandboxes.agents.x-k8s.io
    kubectl -n agent-sandbox-system rollout status deploy/agent-sandbox-controller --timeout=5m
else
    # A server dry run rejects the controller's namespaced objects because
    # their namespace does not exist yet.
    kubectl apply --dry-run=client -f "$work/agent-sandbox.yaml"
fi

echo
echo "== 2. Namespace $OPENSHELL_NAMESPACE"
kubectl create namespace "$OPENSHELL_NAMESPACE" --dry-run=client -o yaml \
    | kubectl apply "${DRY_RUN[@]}" -f -

echo
echo "== 3. Pull secret gitlab-registry, from $AEVOM_NAMESPACE"
pull_secret="$(kubectl -n "$AEVOM_NAMESPACE" get secret gitlab-registry -o json \
    | jq --arg ns "$OPENSHELL_NAMESPACE" \
        '{apiVersion, kind, type, data, metadata: {name: .metadata.name, namespace: $ns}}')"
if $APPLY; then
    echo "$pull_secret" | kubectl apply -f -
else
    echo "would copy secret/gitlab-registry into $OPENSHELL_NAMESPACE"
fi

echo
echo "== 4. OpenShell chart $OPENSHELL_CHART_VERSION"
openshell_values=(
    --set server.telemetryEnabled=false
    --set server.auth.allowUnauthenticatedUsers=true
    --set "server.sandboxImagePullSecrets[0].name=gitlab-registry"
)
if $APPLY; then
    helm upgrade --install openshell "$OPENSHELL_CHART" --version "$OPENSHELL_CHART_VERSION" \
        --namespace "$OPENSHELL_NAMESPACE" "${openshell_values[@]}" --wait --timeout 10m
else
    # The chart's preflight looks for the Agent Sandbox API, which a dry run
    # has not installed, so render offline and list what would be created.
    helm template openshell "$OPENSHELL_CHART" --version "$OPENSHELL_CHART_VERSION" \
        --namespace "$OPENSHELL_NAMESPACE" "${openshell_values[@]}" \
        --set agentSandbox.preflight.enabled=false \
        | awk '/^kind:/ {kind=$2} /^  name:/ && kind {print "would create " kind "/" $2; kind=""}'
fi

echo
echo "== 5. Access for $DEPLOY_SA_NAMESPACE/$DEPLOY_SA_NAME"
cat > "$work/deploy-access.yaml" <<EOF
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: aevom-deploy
  namespace: $OPENSHELL_NAMESPACE
rules:
  - apiGroups: [""]
    resources: [secrets]
    resourceNames: [openshell-client-tls]
    verbs: [get]
  - apiGroups: [""]
    resources: [secrets]
    resourceNames: [gitlab-registry]
    verbs: [get, patch]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: aevom-deploy
  namespace: $OPENSHELL_NAMESPACE
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: aevom-deploy
subjects:
  - kind: ServiceAccount
    name: $DEPLOY_SA_NAME
    namespace: $DEPLOY_SA_NAMESPACE
EOF
if $APPLY; then
    kubectl apply -f "$work/deploy-access.yaml"
else
    echo "would create Role/aevom-deploy and RoleBinding/aevom-deploy in $OPENSHELL_NAMESPACE"
fi

if $APPLY; then
    echo
    echo "== Checks"
    kubectl -n "$OPENSHELL_NAMESPACE" get secret openshell-client-tls \
        -o jsonpath='{.data}' | jq -e 'has("ca.crt") and has("tls.crt") and has("tls.key")' >/dev/null
    echo "openshell-client-tls has ca.crt, tls.crt and tls.key"
    # A SubjectAccessReview rather than `kubectl auth can-i --as`, which needs
    # impersonation rights that admins behind the Rancher proxy do not have.
    deploy_sa="system:serviceaccount:$DEPLOY_SA_NAMESPACE:$DEPLOY_SA_NAME"
    allowed="$(kubectl create -o jsonpath='{.status.allowed}' -f - <<EOF
apiVersion: authorization.k8s.io/v1
kind: SubjectAccessReview
spec:
  user: $deploy_sa
  groups: [system:serviceaccounts, "system:serviceaccounts:$DEPLOY_SA_NAMESPACE", system:authenticated]
  resourceAttributes:
    namespace: $OPENSHELL_NAMESPACE
    verb: get
    resource: secrets
    name: openshell-client-tls
EOF
)"
    if [ "$allowed" != "true" ]; then
        echo "ERROR: $deploy_sa cannot read openshell-client-tls." >&2
        exit 1
    fi
    echo "$deploy_sa can read it"
    kubectl -n "$OPENSHELL_NAMESPACE" get pods
    echo
    echo "Done. Rerun the deploy-production job."
fi
