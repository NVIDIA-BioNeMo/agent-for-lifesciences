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

{{/*
Standard metadata labels for a component.

Deliberately NOT used in spec.selector.matchLabels or in pod template labels.
Deployment and PodDisruptionBudget selectors are immutable, and the live cluster's
objects select on a bare `app: <name>`; adding labels to a selector would make
every subsequent `helm upgrade` fail with no fix short of deleting the Deployment.
The `app` label is carried here too so the existing selectors keep matching.
*/}}
{{- define "aevom.labels" -}}
app: {{ .name }}
app.kubernetes.io/name: {{ .name }}
app.kubernetes.io/instance: {{ .root.Release.Name }}
app.kubernetes.io/part-of: aevom
app.kubernetes.io/managed-by: {{ .root.Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .root.Chart.Name .root.Chart.Version | replace "+" "_" }}
{{- end -}}

{{/*
Image reference for one of the four Aevom images.

There is no default registry because there are no published Aevom images yet, so
this fails the render rather than producing an unpullable reference that only
shows up as an ImagePullBackOff minutes later.
*/}}
{{- define "aevom.image" -}}
{{- $registry := required (printf "image.registry is not set, so the %s image reference cannot be built. Aevom has no public images yet: build them with `make build`, push them to a registry you control, and set image.registry to it (see deploy/helm/README.md)." .component) .root.Values.image.registry -}}
{{- printf "%s/%s:%s" (trimSuffix "/" $registry) .component (.root.Values.image.tag | toString) -}}
{{- end -}}