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

import test from 'node:test';
import assert from 'node:assert/strict';
import { Camera } from 'molstar/lib/mol-canvas3d/camera';
import { Vec3 } from 'molstar/lib/mol-math/linear-algebra';
import type { PluginUIContext } from 'molstar/lib/mol-plugin-ui/context';

import { disableDepthClipping, keepFullDepth } from '../src/lib/molstar/clipping';

/** A camera looking at a scene 100 A across, as Mol* sets one up. */
function sceneCamera() {
  const camera = new Camera({
    ...Camera.createDefaultSnapshot(),
    position: Vec3.create(0, 0, 200),
    target: Vec3.create(0, 0, 0),
    radius: 100,
    radiusMax: 100,
  });
  camera.update();
  return camera;
}

/** What a left click on a residue does: focus a 6 A sphere off-centre. */
function focusResidue(camera: Camera) {
  camera.setState(camera.getFocus(Vec3.create(10, 0, 0), 6), 0);
  camera.update();
}

test('without the guard, focusing a residue clips close to it', () => {
  const camera = sceneCamera();
  focusResidue(camera);
  assert.equal(camera.state.radius, 6);
  assert.ok(camera.far - Vec3.distance(camera.position, camera.target) < 10, 'far plane hugs the residue');
});

test('with the guard, focus still moves the camera but the whole scene stays in view', () => {
  const camera = sceneCamera();
  keepFullDepth(camera);
  focusResidue(camera);
  assert.deepEqual(Array.from(camera.target), [10, 0, 0]);
  assert.ok(Vec3.distance(camera.position, camera.target) < 30, 'zoomed in on the residue');
  assert.equal(camera.state.radius, 100);
  assert.equal(camera.near, camera.state.minNear, 'nothing in front is cut away');
  assert.ok(camera.far >= Vec3.distance(camera.position, camera.target) + 100, 'nothing behind is cut away');
});

test('a larger scene raises the radius with it', () => {
  const camera = sceneCamera();
  keepFullDepth(camera);
  focusResidue(camera);
  camera.setState({ radiusMax: 250 }, 0);
  assert.equal(camera.state.radius, 250);
});

test("Shift + scroll still narrows the clip, because it changes only the radius", () => {
  const camera = sceneCamera();
  keepFullDepth(camera);
  focusResidue(camera);
  camera.setState({ radius: 20 }, 0);
  camera.update();
  assert.equal(camera.state.radius, 20);
  assert.ok(camera.far < Vec3.distance(camera.position, camera.target) + 25, 'the far plane followed it');
});

test('a manual clip survives the scene changing size, and the next focus clears it', () => {
  const camera = sceneCamera();
  keepFullDepth(camera);
  focusResidue(camera);
  camera.setState({ radius: 20 }, 0);
  camera.setState({ radiusMax: 120 }, 0); // e.g. atoms appear around the focused residue
  assert.equal(camera.state.radius, 20);
  focusResidue(camera);
  assert.equal(camera.state.radius, 120);
});

test('a camera transition never clips either', () => {
  const camera = sceneCamera();
  keepFullDepth(camera);
  camera.setState(camera.getFocus(Vec3.create(10, 0, 0), 6), 200);
  // Frames of the move arrive as whole views.
  camera.setState({ ...camera.getSnapshot(), radius: 30 }, 0);
  assert.equal(camera.state.radius, 100);
});

test('the guard can be removed', () => {
  const camera = sceneCamera();
  const restore = keepFullDepth(camera);
  restore();
  focusResidue(camera);
  assert.equal(camera.state.radius, 6);
});

test('disabling depth clipping also turns the fog off', () => {
  const camera = sceneCamera();
  const calls: unknown[] = [];
  const plugin = { canvas3d: { camera, setProps: (p: unknown) => calls.push(p) } } as unknown as PluginUIContext;
  disableDepthClipping(plugin);
  assert.deepEqual(calls, [{ cameraFog: { name: 'off', params: {} } }]);
  focusResidue(camera);
  assert.equal(camera.state.radius, 100);
});

test('with no canvas there is nothing to do', () => {
  assert.doesNotThrow(() => disableDepthClipping({ canvas3d: undefined } as unknown as PluginUIContext)());
});
