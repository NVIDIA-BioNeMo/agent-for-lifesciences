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
 * Keep the whole scene visible when the camera focuses on something, while
 * leaving manual clipping alone.
 *
 * Mol* ties its near and far clipping planes to the camera's `radius`, the
 * size of whatever was last focused. A left click on a residue focuses it, the
 * radius drops to a few angstroms, and everything nearer the camera or deeper
 * behind the residue than that radius is cut away; depth fog then fades what
 * is left towards the background. That is useful for looking into a cavity and
 * unwelcome for reading a design, where the rest of the structure is the
 * context.
 *
 * Two things are turned off:
 *
 * - **Depth fog**, a plain canvas setting.
 * - **The automatic clip on focus.** Mol* exposes no setting for it, but it can
 *   be told apart from the user's own clipping by what is passed to
 *   `camera.setState`. A focus, a reset or a camera transition hands over a
 *   whole view (a target or a position, with the radius); Shift + scroll and
 *   the drag-focus gestures change the radius alone. So a radius that arrives
 *   with a view is raised to the scene's size, and a radius that arrives alone
 *   is the user's and is left as it is. Their clip then lasts until the next
 *   focus or reset, which clears it.
 */

import type { PluginUIContext } from 'molstar/lib/mol-plugin-ui/context';
import type { Camera } from 'molstar/lib/mol-canvas3d/camera';

type CameraSnapshot = Partial<Camera.Snapshot>;

/** True when a snapshot moves the view, as opposed to adjusting one setting. */
function isView(snapshot: CameraSnapshot): boolean {
  return snapshot.target !== undefined || snapshot.position !== undefined;
}

/**
 * Wrap `camera.setState` so a focus never narrows the clipping radius below
 * the scene's. Returns a function that undoes the wrap.
 */
export function keepFullDepth(camera: Camera): () => void {
  const original = camera.setState.bind(camera);
  camera.setState = (snapshot: CameraSnapshot, durationMs?: number) => {
    if (isView(snapshot) && (snapshot.radius !== undefined || snapshot.radiusMax !== undefined)) {
      // A focus, reset or transition frame: the radius belongs to the view.
      const radiusMax = snapshot.radiusMax ?? camera.state.radiusMax;
      const radius = snapshot.radius ?? camera.state.radius;
      snapshot = { ...snapshot, radius: Math.max(radius, radiusMax) };
    } else if (snapshot.radius === undefined && snapshot.radiusMax !== undefined) {
      // The scene changed size. Follow it unless the user has narrowed the clip.
      const { radius, radiusMax } = camera.state;
      if (radius >= radiusMax) snapshot = { ...snapshot, radius: snapshot.radiusMax };
    }
    return original(snapshot, durationMs);
  };
  return () => {
    camera.setState = original;
  };
}

/** Turn off depth fog and focus-driven clipping for this viewer. */
export function disableDepthClipping(plugin: PluginUIContext): () => void {
  const canvas = plugin.canvas3d;
  if (!canvas) return () => undefined;
  canvas.setProps({ cameraFog: { name: 'off', params: {} } });
  return keepFullDepth(canvas.camera);
}
