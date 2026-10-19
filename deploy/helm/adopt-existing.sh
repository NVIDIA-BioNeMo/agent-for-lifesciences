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
# One-time cutover from the kustomize manifests in k8s/ to the Helm chart in
# deploy/helm/aevom.
#
# Why this is needed
# ------------------
# Helm 3 refuses to take ownership of an object it did not create. Running
# `helm upgrade --install` against a namespace that kustomize already populated
# fails with:
#
#     Error: ... exists and cannot be imported into the current release:
#     invalid ownership metadata
#
# The tempting workarounds are both wrong. Deleting the namespace first destroys
# the postgres, redis and phoenix PersistentVolumeClaims, which is production's
# entire chat history and telemetry. Renaming the chart's objects so they do not
# collide leaves the old ones running and the new ones bound to fresh, empty
# volumes -- the same data loss, arrived at more slowly.
#
# What it actually takes is three pieces of metadata per object, which is all
# Helm looks at:
#
#     annotation meta.helm.sh/release-name
#     annotation meta.helm.sh/release-namespace
#     label      app.kubernetes.io/managed-by=Helm
#
# This script writes them onto every object the chart will render, then leaves
# the first `helm upgrade --install` to whoever is doing the cutover (the CI job
# on the next push to main, normally).
#
# Safety
# ------
# Only metadata is touched. No spec field changes, so nothing restarts and no
# volume is rebound. It is idempotent: running it twice is a no-op, and running
# it against a namespace where a resource is missing skips that resource with a
# note rather than failing. Nothing here deletes anything.
#
# Usage
# -----
#     deploy/helm/adopt-existing.sh                    # dry run, prints the plan
#     deploy/helm/adopt-existing.sh --apply            # write the metadata
#     RELEASE=aevom NAMESPACE=aevom deploy/helm/adopt-existing.sh --apply
#
# The full runbook, including what to check before and after, is in
# deploy/helm/README.md under "Cutover from kustomize".

set -euo pipefail

RELEASE="${RELEASE:-aevom}"
NAMESPACE="${NAMESPACE:-aevom}"
APPLY=false

for argument in "$@"; do
    case "$argument" in
        --apply) APPLY=true ;;
        -h|--help)
            sed -n '2,45p' "$0" | sed 's|^# \{0,1\}||'
            exit 0
            ;;
        *)
            echo "Unknown argument: $argument (try --help)" >&2
            exit 2
            ;;
    esac
done

# Every namespaced object the chart renders for production, as kubectl
# type/name. Kept in the same order the chart creates them, purely for readable
# output. aevom-prompts is included because the old deploy job created it
# imperatively with `kubectl create configmap`, so it exists and is unowned just
# like the rest.
#
# Deliberately absent:
#   - namespace/aevom, which the chart does not render (helm --create-namespace
#     handles it, and a namespace inside a release is a footgun: `helm uninstall`
#     would take the whole thing with it).
#   - the letsencrypt ClusterIssuers, cluster-scoped and unreferenced while the
#     internal ingress is HTTP-only. See certManager in values-production.yaml.
#   - persistentvolumeclaim/shared-images, which the chart intentionally no
#     longer renders. Adopting it would only hand the release something to
#     delete. See the runbook for retiring it.
#   - secret/aevom-secrets, secret/nvidia-api-key, secret/nvidia-api-key-nvinf
#     and secret/gitlab-registry. The chart references but never creates these,
#     so they must stay outside the release -- key material never passes through
#     a values file or a release revision.
RESOURCES=(
    configmap/aevom-config
    configmap/aevom-prompts
    deployment/aevom-api
    service/aevom-api
    poddisruptionbudget/aevom-api
    horizontalpodautoscaler/aevom-api
    deployment/rdkit
    service/rdkit
    horizontalpodautoscaler/rdkit
    deployment/pyexec
    service/pyexec
    horizontalpodautoscaler/pyexec
    deployment/vina
    service/vina
    horizontalpodautoscaler/vina
    deployment/postgres
    service/postgres
    persistentvolumeclaim/postgres-data
    deployment/redis
    service/redis
    persistentvolumeclaim/redis-data
    deployment/phoenix
    service/phoenix
    persistentvolumeclaim/phoenix-data
    ingress/aevom-ingress
)

if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
    echo "Namespace '$NAMESPACE' does not exist. Nothing to adopt -- this is a"
    echo "fresh cluster, so just run helm upgrade --install directly."
    exit 0
fi

# A release that already exists means the cutover happened. Re-running would be
# harmless but it is worth saying so rather than looking like it did something.
if helm -n "$NAMESPACE" status "$RELEASE" >/dev/null 2>&1; then
    echo "WARNING: release '$RELEASE' already exists in namespace '$NAMESPACE'."
    echo "         The cutover is already done; this script is for the first one."
    echo
fi

echo "Release:   $RELEASE"
echo "Namespace: $NAMESPACE"
if $APPLY; then
    echo "Mode:      APPLY -- metadata will be written"
else
    echo "Mode:      dry run -- pass --apply to write metadata"
fi
echo

adopted=0
skipped=0
already=0

for resource in "${RESOURCES[@]}"; do
    if ! kubectl -n "$NAMESPACE" get "$resource" >/dev/null 2>&1; then
        printf '  %-45s absent, skipping\n' "$resource"
        skipped=$((skipped + 1))
        continue
    fi

    owner=$(kubectl -n "$NAMESPACE" get "$resource" \
        -o 'jsonpath={.metadata.annotations.meta\.helm\.sh/release-name}' 2>/dev/null || true)
    if [ "$owner" = "$RELEASE" ]; then
        printf '  %-45s already adopted\n' "$resource"
        already=$((already + 1))
        continue
    fi
    if [ -n "$owner" ]; then
        echo "ERROR: $resource is owned by release '$owner', not '$RELEASE'."
        echo "       Adopting it would make two releases fight over one object."
        exit 1
    fi

    if $APPLY; then
        # --overwrite because a partially-adopted object (an interrupted earlier
        # run) has some of these set already, and annotate fails on a conflict.
        kubectl -n "$NAMESPACE" annotate "$resource" --overwrite \
            "meta.helm.sh/release-name=$RELEASE" \
            "meta.helm.sh/release-namespace=$NAMESPACE" >/dev/null
        kubectl -n "$NAMESPACE" label "$resource" --overwrite \
            app.kubernetes.io/managed-by=Helm >/dev/null
        printf '  %-45s adopted\n' "$resource"
    else
        printf '  %-45s would adopt\n' "$resource"
    fi
    adopted=$((adopted + 1))
done

echo
if $APPLY; then
    echo "Adopted $adopted, already adopted $already, absent $skipped."
    echo
    echo "Next: confirm the chart agrees with what is running before releasing."
    echo
    echo "  helm template $RELEASE deploy/helm/aevom --namespace $NAMESPACE \\"
    echo "    -f deploy/helm/aevom/values-production.yaml \\"
    echo "    --set image.tag=\$(kubectl -n $NAMESPACE get deployment/aevom-api \\"
    echo "      -o jsonpath='{.spec.template.spec.containers[0].image}' | sed 's/.*://') \\"
    echo "    --set-file prompts.content=prompts.md | kubectl -n $NAMESPACE diff -f - || true"
    echo
    echo "Expect only the changes this migration makes on purpose (the dropped"
    echo "IMAGES_DIR and shared-images mounts, phoenix's Recreate strategy, the"
    echo "chart's labels and checksum annotations). A PVC spec or selector change"
    echo "means stop: both are immutable."
    echo
    echo "Note that the first release will not carry out the removals that diff"
    echo "reports. Helm has no prior manifest for an adopted object, so its patch"
    echo "can only add and change. deploy/helm/README.md lists the full expected"
    echo "diff and the post-release steps -- patching the leftovers out, verifying"
    echo "the three volumes kept their data, and deleting the shared-images claim."
else
    echo "Would adopt $adopted, already adopted $already, absent $skipped."
    echo "Re-run with --apply to write the metadata."
fi
