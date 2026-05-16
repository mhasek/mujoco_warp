# Copyright 2026 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for render functions."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data

try:
  mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco/>"))
  _HAS_RENDERER = True
except Exception:
  _HAS_RENDERER = False


def _assert_eq(a, b, name):
  tol = 5e-4
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


def _unpack_rgb(packed):
  r = ((packed >> 16) & 0xFF).astype(np.uint8)
  g = ((packed >> 8) & 0xFF).astype(np.uint8)
  b = (packed & 0xFF).astype(np.uint8)
  return np.stack([r, g, b], axis=-1)


class RenderTest(parameterized.TestCase):
  @parameterized.parameters(2, 512)
  def test_render(self, nworld: int):
    mjm, mjd, m, d = test_data.fixture("primitives.xml", nworld=nworld)

    rc = mjw.create_render_context(
      mjm,
      nworld=nworld,
      cam_res=(32, 32),
      render_rgb=True,
      render_depth=True,
    )

    mjw.render(m, d, rc)

    rgb = rc.rgb_data.numpy()
    depth = rc.depth_data.numpy()

    self.assertGreater(np.count_nonzero(rgb), 0)
    self.assertGreater(np.count_nonzero(depth), 0)

    self.assertNotEqual(np.unique(rgb).shape[0], 1)
    self.assertNotEqual(np.unique(depth).shape[0], 1)

  def test_render_humanoid(self):
    mjm, mjd, m, d = test_data.fixture("humanoid/humanoid.xml")
    rc = mjw.create_render_context(
      mjm,
      cam_res=(32, 32),
      render_rgb=True,
      render_depth=True,
    )
    mjw.render(m, d, rc)
    rgb = rc.rgb_data.numpy()

    self.assertNotEqual(np.unique(rgb).shape[0], 1)

  @absltest.skipIf(not wp.get_device().is_cuda, "Skipping test that requires CUDA.")
  def test_render_graph_capture(self):
    mjm, mjd, m, d = test_data.fixture("humanoid/humanoid.xml")
    rc = mjw.create_render_context(
      mjm,
      cam_res=(32, 32),
      render_rgb=True,
      render_depth=True,
    )

    mjw.render(m, d, rc)
    rgb_np = rc.rgb_data.numpy()

    with wp.ScopedCapture() as capture:
      mjw.render(m, d, rc)

    wp.capture_launch(capture.graph)

    _assert_eq(rgb_np, rc.rgb_data.numpy(), "rgb_data")

  @parameterized.parameters(2, 512)
  def test_render_segmentation(self, nworld: int):
    mjm, mjd, m, d = test_data.fixture("primitives.xml", nworld=nworld)

    rc = mjw.create_render_context(
      mjm,
      nworld=nworld,
      cam_res=(32, 32),
      render_rgb=False,
      render_depth=False,
      render_seg=True,
    )

    mjw.render(m, d, rc)

    seg = rc.seg_data.numpy()

    geom_mask = seg[..., 1] == int(mjw.ObjType.GEOM)
    self.assertTrue(np.any(geom_mask), "Expected at least one geom hit")
    self.assertGreater(np.unique(seg[..., 0][geom_mask]).shape[0], 1)

  def test_render_rgb_and_segmentation(self):
    mjm, mjd, m, d = test_data.fixture("primitives.xml", nworld=2)

    rc = mjw.create_render_context(
      mjm,
      nworld=2,
      cam_res=(32, 32),
      render_rgb=True,
      render_seg=True,
    )

    mjw.render(m, d, rc)

    rgb = rc.rgb_data.numpy()
    seg = rc.seg_data.numpy()

    self.assertGreater(np.count_nonzero(rgb), 0)
    self.assertTrue(np.any(seg[..., 1] == int(mjw.ObjType.GEOM)))

  def test_disable_ambient_lighting(self):
    xml = """
    <mujoco>
      <visual>
        <headlight active="0" ambient="0 0 0" diffuse="0 0 0" specular="0 0 0"/>
      </visual>
      <worldbody>
        <camera name="cam" pos="0 -3 1" xyaxes="1 0 0 0 0.25 1" resolution="32 32" output="rgb"/>
        <geom type="sphere" pos="0 0 0.5" size="0.5" rgba="1 0 0 1"/>
      </worldbody>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)
    self.assertEqual(mjm.nlight, 0)
    self.assertEqual(m.nlight, 0)

    rc = mjw.create_render_context(
      mjm,
      cam_res=(32, 32),
      render_rgb=True,
      render_seg=True,
    )
    mjw.render(m, d, rc)

    seg = rc.seg_data.numpy()[0]
    geom_mask = seg[:, 1] == int(mjw.ObjType.GEOM)
    self.assertTrue(np.any(geom_mask), "Expected at least one geom hit")

    rgb = _unpack_rgb(rc.rgb_data.numpy()[0])
    self.assertGreater(np.count_nonzero(rgb[geom_mask]), 0)

    rc = mjw.create_render_context(
      mjm,
      cam_res=(32, 32),
      render_rgb=True,
      render_seg=True,
      use_ambient_lighting=False,
    )
    mjw.render(m, d, rc)

    seg = rc.seg_data.numpy()[0]
    geom_mask = seg[:, 1] == int(mjw.ObjType.GEOM)
    rgb = _unpack_rgb(rc.rgb_data.numpy()[0])
    np.testing.assert_array_equal(rgb[geom_mask], 0)

  @absltest.skipIf(not _HAS_RENDERER, "MuJoCo rendering requires OpenGL")
  def test_segmentation_matches_mujoco(self):
    """Segmentation should match native MuJoCo's `(object_id, object_type)` output."""
    mjm, mjd, m, d = test_data.fixture("primitives.xml", nworld=1)
    cam_w, cam_h = 32, 32

    rc = mjw.create_render_context(
      mjm,
      nworld=1,
      cam_res=(cam_w, cam_h),
      render_seg=[True],
    )
    mjw.render(m, d, rc)

    warp_seg_np = rc.seg_data.numpy()[0].reshape(-1, 2)

    with mujoco.Renderer(mjm, height=cam_h, width=cam_w) as renderer:
      renderer.update_scene(mjd, camera=0)
      renderer.enable_segmentation_rendering()
      mj_seg = renderer.render().reshape(-1, 2)

    np.testing.assert_array_equal(warp_seg_np, mj_seg)

  @absltest.skipIf(not _HAS_RENDERER, "MuJoCo rendering requires OpenGL")
  def test_depth_matches_mujoco(self):
    """Depth values should match native MuJoCo (planar depth, not Euclidean)."""
    mjm, mjd, m, d = test_data.fixture("primitives.xml", nworld=1)
    cam_w, cam_h = 32, 32

    # mjwarp depth
    rc = mjw.create_render_context(
      mjm,
      nworld=1,
      cam_res=(cam_w, cam_h),
      render_rgb=[False],
      render_depth=[True],
    )
    mjw.render(m, d, rc)
    warp_depth = rc.depth_data.numpy()[0]  # flat array for world 0

    # Native MuJoCo depth
    with mujoco.Renderer(mjm, height=cam_h, width=cam_w) as renderer:
      renderer.update_scene(mjd, camera=0)
      renderer.enable_depth_rendering()
      mj_depth = renderer.render().flatten()

    # Compare only pixels that hit geometry (non-zero in both)
    valid = (warp_depth > 0) & (mj_depth > 0)
    np.testing.assert_allclose(
      warp_depth[valid],
      mj_depth[valid],
      atol=1e-2,
      rtol=1e-2,
    )

  # Each scene places the camera at the origin fully enclosed by a geom (a
  # primitive or a convex mesh), with a marker box at +Y (in front of the
  # camera) well outside the enclosure. A correctly backface-culling renderer
  # must drop the far exit-face hit on the enclosure and "see through" to the
  # marker.
  _BACKFACE_CULL_SCENE = """
    <mujoco>
      <visual>
        <map znear="0.001" />
      </visual>{asset}
      <worldbody>
        <camera xyaxes="1 0 0 0 0 1" />
        <geom name="enclosure" {enclosure} />
        <geom name="marker" type="box" size="0.5 0.5 0.5" pos="0 5 0" />
      </worldbody>
    </mujoco>"""

  _MESH_ASSET = """
  <asset>
    <mesh name="tetra" vertex="1 1 1  1 -1 -1  -1 1 -1  -1 -1 1" />
  </asset>"""

  _BACKFACE_CULL_PRIMITIVES = (
    ("sphere", "", 'type="sphere" size="1"'),
    ("ellipsoid", "", 'type="ellipsoid" size="1 1 1"'),
    ("capsule", "", 'type="capsule" size="0.5 0.5"'),
    ("cylinder", "", 'type="cylinder" size="1 1"'),
    ("box", "", 'type="box" size="1 1 1"'),
    ("mesh", _MESH_ASSET, 'type="mesh" mesh="tetra"'),
  )

  @parameterized.named_parameters(*_BACKFACE_CULL_PRIMITIVES)
  def test_backface_cull_camera_inside_primitive(self, asset: str, enclosure: str):
    """Camera inside a geom must not render that geom's back face."""
    xml = self._BACKFACE_CULL_SCENE.format(asset=asset, enclosure=enclosure)
    mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=1)

    cam_w, cam_h = 16, 16
    rc = mjw.create_render_context(
      mjm,
      nworld=1,
      cam_res=(cam_w, cam_h),
      render_rgb=True,
      render_depth=True,
      render_seg=True,
    )
    mjw.render(m, d, rc)

    seg = rc.seg_data.numpy()[0]
    depth = rc.depth_data.numpy()[0]

    geom_mask = seg[..., 1] == int(mjw.ObjType.GEOM)
    hit_ids = seg[..., 0][geom_mask]
    enclosure_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "enclosure")
    marker_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "marker")

    self.assertFalse(
      np.any(hit_ids == enclosure_id),
      "enclosing geom should be backface-culled but appeared in segmentation",
    )

    self.assertTrue(
      np.any(hit_ids == marker_id),
      "camera should see through the enclosing geom to the marker box",
    )

    # Considering the inner surface of the enclosure is culled, the depth of the marker should
    # be ~5.0 i.e. the distance to the box surface.
    marker_depth = depth.reshape(cam_h, cam_w)[seg[..., 0].reshape(cam_h, cam_w) == marker_id]
    if marker_depth.size > 0:
      self.assertGreater(float(np.min(marker_depth)), 1.0)

  @absltest.skipIf(not _HAS_RENDERER, "MuJoCo rendering requires OpenGL")
  @parameterized.named_parameters(*_BACKFACE_CULL_PRIMITIVES)
  def test_backface_cull_matches_mujoco(self, asset: str, enclosure: str):
    """Backface-cull behavior must match native MuJoCo for every geom type."""
    xml = self._BACKFACE_CULL_SCENE.format(asset=asset, enclosure=enclosure)
    mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=1)

    cam_w, cam_h = 16, 16
    rc = mjw.create_render_context(
      mjm,
      nworld=1,
      cam_res=(cam_w, cam_h),
      render_seg=[True],
    )
    mjw.render(m, d, rc)
    warp_seg = rc.seg_data.numpy()[0].reshape(-1, 2)

    with mujoco.Renderer(mjm, height=cam_h, width=cam_w) as renderer:
      renderer.update_scene(mjd, camera=0)
      renderer.enable_segmentation_rendering()
      mj_seg = renderer.render().reshape(-1, 2)

    np.testing.assert_array_equal(warp_seg, mj_seg)

  @parameterized.named_parameters(*_BACKFACE_CULL_PRIMITIVES)
  def test_backface_cull_disabled_keeps_enclosure(self, asset: str, enclosure: str):
    """When `enable_backface_culling=False`, the enclosure must reappear."""
    xml = self._BACKFACE_CULL_SCENE.format(asset=asset, enclosure=enclosure)
    mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=1)

    cam_w, cam_h = 16, 16
    rc = mjw.create_render_context(
      mjm,
      nworld=1,
      cam_res=(cam_w, cam_h),
      render_seg=True,
      enable_backface_culling=False,
    )
    mjw.render(m, d, rc)

    seg = rc.seg_data.numpy()[0]
    geom_mask = seg[..., 1] == int(mjw.ObjType.GEOM)
    hit_ids = seg[..., 0][geom_mask]

    self.assertTrue(
      np.any(hit_ids == mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "enclosure")),
      "with cull disabled, enclosing geom should appear in segmentation",
    )

  def test_msaa_smooths_silhouette(self):
    """samples_per_pixel > 1 should anti-alias geom silhouettes."""
    xml = """
    <mujoco>
      <visual>
        <headlight active="1" ambient="0.3 0.3 0.3" diffuse="0.8 0.8 0.8" specular="0.5 0.5 0.5"/>
        <map znear="0.01"/>
      </visual>
      <worldbody>
        <camera name="cam" pos="0 -2 0.5" xyaxes="1 0 0 0 0.3 1" resolution="32 32"/>
        <geom type="box" pos="0 0 0.3" size="0.4 0.4 0.4" euler="0 0 30" rgba="0.9 0.2 0.2 1"/>
      </worldbody>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    def partial_edge_count(n_samples: int) -> int:
      rc = mjw.create_render_context(mjm, cam_res=(32, 32), render_rgb=True, samples_per_pixel=n_samples)
      mjw.render(m, d, rc)
      r = _unpack_rgb(rc.rgb_data.numpy()[0])[..., 0].reshape(32, 32)
      # Pixels that are neither pure background (R==0) nor pure box (R>=200).
      return int(((r > 0) & (r < 200)).sum())

    n1 = partial_edge_count(1)
    n16 = partial_edge_count(16)
    self.assertGreater(n16, n1, f"MSAA should produce more anti-aliased edge pixels (n1={n1}, n16={n16})")

  def test_msaa_n1_matches_baseline(self):
    """samples_per_pixel == 1 must be bit-identical to the default render."""
    mjm, mjd, m, d = test_data.fixture("primitives.xml", nworld=1)
    rc_default = mjw.create_render_context(mjm, nworld=1, cam_res=(32, 32), render_rgb=True)
    rc_n1 = mjw.create_render_context(mjm, nworld=1, cam_res=(32, 32), render_rgb=True, samples_per_pixel=1)

    mjw.render(m, d, rc_default)
    mjw.render(m, d, rc_n1)
    np.testing.assert_array_equal(rc_default.rgb_data.numpy(), rc_n1.rgb_data.numpy())

  def test_msaa_rejects_zero(self):
    mjm, _, _, _ = test_data.fixture("primitives.xml")
    with self.assertRaises(ValueError):
      mjw.create_render_context(mjm, cam_res=(32, 32), render_rgb=True, samples_per_pixel=0)

  # ---- opt-out flags ----
  # A scene that exercises every feature: headlight on, one user light with
  # nonzero ambient, a material with specular + emission, default shadows.
  _FLAGS_XML = """
  <mujoco>
    <visual>
      <headlight active="1" ambient="0.1 0.1 0.1" diffuse="0.8 0.8 0.8" specular="0.5 0.5 0.5"/>
      <map znear="0.01"/>
    </visual>
    <asset>
      <material name="m" specular="0.7" shininess="0.5" emission="0.3" rgba="0.2 0.4 0.8 1"/>
    </asset>
    <worldbody>
      <camera name="cam" pos="0 -2 0.5" xyaxes="1 0 0 0 0.3 1" resolution="32 32"/>
      <light pos="0 0 3" dir="0 0 -1" directional="true"
             diffuse="0.6 0.6 0.6" specular="0.4 0.4 0.4" ambient="0.15 0.15 0.15"
             attenuation="1 0 0"/>
      <geom name="floor" type="plane" size="2 2 0.1" rgba="0.6 0.6 0.6 1"/>
      <geom name="ball" type="sphere" pos="0 0 0.3" size="0.3" material="m"/>
    </worldbody>
  </mujoco>
  """

  def _render_with(self, **kwargs) -> np.ndarray:
    mjm, _, m, d = test_data.fixture(xml=self._FLAGS_XML)
    rc = mjw.create_render_context(mjm, cam_res=(32, 32), render_rgb=True, **kwargs)
    mjw.render(m, d, rc)
    return _unpack_rgb(rc.rgb_data.numpy()[0]).reshape(32, 32, 3)

  def test_enable_headlight_off_changes_render(self):
    """Disabling the headlight must remove its contribution from the image."""
    on = self._render_with()
    off = self._render_with(enable_headlight=False)
    # Headlight is by far the brightest source in this scene; disabling it
    # must dim the lit ball substantially.
    self.assertGreater(
      int(on.max()), int(off.max()) + 30, f"headlight-off max ({int(off.max())}) too close to on ({int(on.max())})"
    )

  def test_enable_specular_off_removes_highlight(self):
    """Disabling specular must dim the specular hotspot on the ball."""
    on = self._render_with()
    off = self._render_with(enable_specular=False)
    # Specular highlights show as elevated green channel (the ball is blue
    # so its base diffuse-only green is low; specular adds neutral white).
    # The maximum green pixel on the ball drops measurably without specular.
    self.assertGreater(int(on[..., 1].max()), int(off[..., 1].max()) + 20)

  def test_enable_emission_off_dims_geom(self):
    """Disabling emission must dim the geom by its mat_emission * base_color."""
    on = self._render_with()
    off = self._render_with(enable_emission=False)
    # The ball's blue channel includes 0.3 * 0.8 * 255 = 61 of emission.
    # Disabling emission removes that even on pixels in shadow.
    self.assertGreater(int(on[..., 2].mean()), int(off[..., 2].mean()))

  def test_enable_per_light_ambient_off_dims_unlit_floor(self):
    """Disabling per-light ambient must dim regions outside the light cone."""
    on = self._render_with()
    off = self._render_with(enable_per_light_ambient=False)
    # The light_ambient term contributes to every pixel regardless of N.L,
    # so disabling it must reduce overall brightness.
    self.assertGreater(float(on.mean()), float(off.mean()))

  def test_all_flags_off_still_renders(self):
    """Turning every Phase 1 feature flag off must still produce a valid image."""
    img = self._render_with(
      enable_headlight=False,
      enable_specular=False,
      enable_emission=False,
      enable_per_light_ambient=False,
    )
    # Should still see *something* (the directional light's diffuse term).
    self.assertGreater(int(img.max()), 10)


if __name__ == "__main__":
  wp.init()
  absltest.main()
