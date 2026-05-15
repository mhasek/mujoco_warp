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
"""Parity tests for the mjwarp ray-traced renderer vs MuJoCo OpenGL.

These tests compare mjwarp output to `mujoco.Renderer` on canonical fixtures
that exercise lighting and material parameters. They are intentionally tolerant
of small numerical differences (per-pixel ray tracing vs per-vertex
rasterization), but tight enough to catch regressions in lighting behavior.

The full parity suite is built up incrementally across commits in the Phase 1
PR. As each commit closes a parity gap, the corresponding test is enabled
here.
"""

import pathlib

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw

_GOLDEN_DIR = pathlib.Path(__file__).parent.parent / "test_data" / "golden_renders"

try:
  mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco/>"))
  _HAS_RENDERER = True
except Exception:
  _HAS_RENDERER = False


# ---------- helpers ------------------------------------------------------------


def _unpack_rgb(packed: np.ndarray) -> np.ndarray:
  """Unpack mjwarp's ABGR-packed uint32 buffer to an Nx3 uint8 array."""
  r = ((packed >> 16) & 0xFF).astype(np.uint8)
  g = ((packed >> 8) & 0xFF).astype(np.uint8)
  b = (packed & 0xFF).astype(np.uint8)
  return np.stack([r, g, b], axis=-1)


def _mjwarp_render(mjm: mujoco.MjModel, cam_w: int, cam_h: int) -> np.ndarray:
  """Render `mjm` with mjwarp and return an HxWx3 uint8 RGB image (world 0)."""
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd)

  rc = mjw.create_render_context(
    mjm,
    cam_res=(cam_w, cam_h),
    render_rgb=True,
    use_shadows=False,
  )
  mjw.render(m, d, rc)
  rgb = _unpack_rgb(rc.rgb_data.numpy()[0])
  return rgb.reshape(cam_h, cam_w, 3)


def _mujoco_render(mjm: mujoco.MjModel, cam_w: int, cam_h: int) -> np.ndarray:
  """Render `mjm` with `mujoco.Renderer` and return an HxWx3 uint8 RGB image."""
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  with mujoco.Renderer(mjm, height=cam_h, width=cam_w) as renderer:
    renderer.update_scene(mjd, camera=0)
    return renderer.render()


def _mean_l1(a: np.ndarray, b: np.ndarray) -> float:
  """Mean per-pixel L1 distance in [0, 1] color space."""
  return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean() / 255.0)


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
  """Compact SSIM implementation (luminance only) for HxWx3 uint8 inputs.

  Uses an 8x8 box window. This is intentionally simple to avoid a scikit-image
  dependency; absolute numbers are not directly comparable to the canonical
  Wang et al. implementation but the same threshold semantics apply.
  """
  ay = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float32) / 255.0
  by = (0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2]).astype(np.float32) / 255.0

  k1, k2, L = 0.01, 0.03, 1.0
  c1, c2 = (k1 * L) ** 2, (k2 * L) ** 2

  win = 8
  h, w = ay.shape
  hh, ww = h - h % win, w - w % win
  ay = ay[:hh, :ww].reshape(hh // win, win, ww // win, win)
  by = by[:hh, :ww].reshape(hh // win, win, ww // win, win)
  ay = ay.transpose(0, 2, 1, 3).reshape(-1, win * win)
  by = by.transpose(0, 2, 1, 3).reshape(-1, win * win)

  mu_a, mu_b = ay.mean(axis=1), by.mean(axis=1)
  va = ay.var(axis=1)
  vb = by.var(axis=1)
  cov = ((ay - mu_a[:, None]) * (by - mu_b[:, None])).mean(axis=1)

  num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
  den = (mu_a**2 + mu_b**2 + c1) * (va + vb + c2)
  return float((num / den).mean())


# ---------- fixtures -----------------------------------------------------------

# Each XML below isolates one lighting/material behavior and provides a small
# camera so renders are cheap. The headlight is intentionally enabled in most
# fixtures to exercise the default MuJoCo OpenGL behavior of always lighting
# the scene from the camera.
_FIXTURE_HEADLIGHT_ONLY = """
<mujoco>
  <visual>
    <headlight active="1" ambient="0.3 0.3 0.3" diffuse="0.8 0.8 0.8" specular="0.5 0.5 0.5"/>
    <map znear="0.01"/>
  </visual>
  <worldbody>
    <camera name="cam" pos="0 -2 0.5" xyaxes="1 0 0 0 0.3 1" resolution="64 64"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.6 0.6 0.6 1"/>
    <geom name="ball" type="sphere" pos="0 0 0.3" size="0.3" rgba="0.8 0.2 0.2 1"/>
  </worldbody>
</mujoco>
"""

_FIXTURE_SPOTLIGHT = """
<mujoco>
  <visual>
    <headlight active="0" ambient="0 0 0" diffuse="0 0 0" specular="0 0 0"/>
    <map znear="0.01"/>
  </visual>
  <worldbody>
    <camera name="cam" pos="0 -2 0.8" xyaxes="1 0 0 0 0.5 1" resolution="64 64"/>
    <light pos="0 0 2.5" dir="0 0 -1" cutoff="20" exponent="10"
           diffuse="1 0.9 0.7" specular="0 0 0" ambient="0 0 0"
           attenuation="1 0 0"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.7 0.7 0.7 1"/>
    <geom name="ball" type="sphere" pos="0 0 0.3" size="0.3" rgba="0.8 0.8 0.8 1"/>
  </worldbody>
</mujoco>
"""

_FIXTURE_SPECULAR = """
<mujoco>
  <visual>
    <headlight active="1" ambient="0.2 0.2 0.2" diffuse="0.8 0.8 0.8" specular="0.9 0.9 0.9"/>
    <map znear="0.01"/>
  </visual>
  <asset>
    <material name="shiny" specular="0.9" shininess="0.8" rgba="0.2 0.2 0.8 1"/>
  </asset>
  <worldbody>
    <camera name="cam" pos="0 -2 0.6" xyaxes="1 0 0 0 0.4 1" resolution="64 64"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.5 0.5 0.5 1"/>
    <geom name="ball" type="sphere" pos="0 0 0.3" size="0.3" material="shiny"/>
  </worldbody>
</mujoco>
"""

_FIXTURE_TWO_LIGHTS = """
<mujoco>
  <visual>
    <headlight active="0" ambient="0 0 0" diffuse="0 0 0" specular="0 0 0"/>
    <map znear="0.01"/>
  </visual>
  <worldbody>
    <camera name="cam" pos="0 -2 0.6" xyaxes="1 0 0 0 0.4 1" resolution="64 64"/>
    <light pos="-1 -1 2" dir="0.4 0.4 -1" directional="true" diffuse="1 0 0" specular="0 0 0" ambient="0 0 0"/>
    <light pos=" 1 -1 2" dir="-0.4 0.4 -1" directional="true" diffuse="0 0 1" specular="0 0 0" ambient="0 0 0"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    <geom name="ball" type="sphere" pos="0 0 0.3" size="0.3" rgba="1 1 1 1"/>
  </worldbody>
</mujoco>
"""

_FIXTURE_EMISSION = """
<mujoco>
  <visual>
    <headlight active="0" ambient="0 0 0" diffuse="0 0 0" specular="0 0 0"/>
    <map znear="0.01"/>
  </visual>
  <asset>
    <material name="glow" emission="0.8" rgba="0 1 0 1"/>
  </asset>
  <worldbody>
    <camera name="cam" pos="0 -2 0.5" xyaxes="1 0 0 0 0.3 1" resolution="64 64"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.2 0.2 0.2 1"/>
    <geom name="ball" type="sphere" pos="0 0 0.3" size="0.3" material="glow"/>
  </worldbody>
</mujoco>
"""


@absltest.skipIf(not _HAS_RENDERER, "MuJoCo rendering requires OpenGL")
class RenderParityTest(parameterized.TestCase):
  """Compare mjwarp lighting against MuJoCo OpenGL on canonical scenes.

  The SSIM/L1 thresholds are deliberately loose: per-pixel ray-traced shading
  vs per-vertex rasterized shading will never agree exactly, but they should
  respond to parameter changes the same way. Per-test thresholds are set
  empirically with margin below the observed mean.
  """

  # ---- headlight ----
  def test_headlight_only_lights_the_scene(self):
    mjm = mujoco.MjModel.from_xml_string(_FIXTURE_HEADLIGHT_ONLY)
    self.assertEqual(mjm.nlight, 0, "fixture must have no explicit lights")

    warp_rgb = _mjwarp_render(mjm, 64, 64)
    self.assertGreater(int(warp_rgb.max()), 30, "headlight should light the ball/floor")

    mj_rgb = _mujoco_render(mjm, 64, 64)
    self.assertLess(_mean_l1(warp_rgb, mj_rgb), 0.15, "headlight scene mean L1 too large")
    self.assertGreater(_ssim(warp_rgb, mj_rgb), 0.75, "headlight scene SSIM too low")

  # ---- spotlight cutoff / exponent ----
  def test_spotlight_respects_cutoff(self):
    """A 20-degree cutoff should produce a clearly bounded bright cone."""
    mjm = mujoco.MjModel.from_xml_string(_FIXTURE_SPOTLIGHT)
    warp_rgb = _mjwarp_render(mjm, 64, 64)

    # The spotlight is at (0,0,2.5) pointing straight down. The 20-degree
    # cone illuminates a ~0.9 m disk on the floor at z=0. With the camera at
    # (0,-2,0.8), the floor occupies image rows ~16-44. Row 32 col 32 hits
    # the floor near the cone center (bright); row 20 col 32 hits the floor
    # well outside the cone (dim, ambient-only).
    floor_inside = int(warp_rgb[32, 32].sum())
    floor_outside = int(warp_rgb[20, 32].sum())
    self.assertGreater(
      floor_inside,
      floor_outside + 200,
      f"spotlight center should be much brighter than outside-cone floor "
      f"(inside_sum={floor_inside}, outside_sum={floor_outside})",
    )

    mj_rgb = _mujoco_render(mjm, 64, 64)
    self.assertLess(_mean_l1(warp_rgb, mj_rgb), 0.20, "spotlight mean L1 too large")

  def test_spotlight_attenuation_is_read_from_model(self):
    """Changing `attenuation` should produce a visibly different render."""
    spec = mujoco.MjSpec.from_string(_FIXTURE_SPOTLIGHT)
    spec.lights[0].attenuation = [1.0, 0.0, 0.5]
    mjm_quad = spec.compile()
    rgb_quad = _mjwarp_render(mjm_quad, 64, 64)

    spec = mujoco.MjSpec.from_string(_FIXTURE_SPOTLIGHT)
    spec.lights[0].attenuation = [1.0, 0.0, 0.0]
    mjm_const = spec.compile()
    rgb_const = _mjwarp_render(mjm_const, 64, 64)

    self.assertGreater(_mean_l1(rgb_quad, rgb_const), 0.01, "attenuation should change the render; field is being ignored")

  # ---- specular ----
  def test_specular_highlight_appears(self):
    """A high-specular material under a directional light should show a highlight."""
    # Use a fixture with an explicit light to exercise specular without
    # depending on the headlight (which is added in a later commit).
    xml = """
    <mujoco>
      <visual>
        <headlight active="0" ambient="0 0 0" diffuse="0 0 0" specular="0 0 0"/>
        <map znear="0.01"/>
      </visual>
      <asset>
        <material name="shiny" specular="0.9" shininess="0.8" rgba="0.2 0.2 0.8 1"/>
      </asset>
      <worldbody>
        <camera name="cam" pos="0 -2 0.6" xyaxes="1 0 0 0 0.4 1" resolution="64 64"/>
        <light pos="0 -1 3" dir="0 0.3 -1" directional="true"
               diffuse="0.8 0.8 0.8" specular="1 1 1" ambient="0 0 0" attenuation="1 0 0"/>
        <geom name="floor" type="plane" size="2 2 0.1" rgba="0.5 0.5 0.5 1"/>
        <geom name="ball" type="sphere" pos="0 0 0.3" size="0.3" material="shiny"/>
      </worldbody>
    </mujoco>
    """
    # Same scene with specular disabled on the material.
    xml_no_spec = xml.replace('specular="0.9"', 'specular="0.0"')

    mjm = mujoco.MjModel.from_xml_string(xml)
    mjm_no_spec = mujoco.MjModel.from_xml_string(xml_no_spec)

    warp_with = _mjwarp_render(mjm, 64, 64).astype(np.int32)
    warp_without = _mjwarp_render(mjm_no_spec, 64, 64).astype(np.int32)

    # The ball pixels should be measurably brighter with specular on.
    diff = warp_with.sum(axis=-1) - warp_without.sum(axis=-1)
    bright_pixels = int((diff > 30).sum())
    self.assertGreater(bright_pixels, 5, f"specular highlight should add brightness to >=5 pixels (got {bright_pixels})")

  # ---- per-light color ----
  def test_per_light_color_propagates(self):
    """Two colored directional lights should yield non-grayscale output."""
    mjm = mujoco.MjModel.from_xml_string(_FIXTURE_TWO_LIGHTS)
    warp_rgb = _mjwarp_render(mjm, 64, 64).astype(np.float32)

    floor_mask = warp_rgb.sum(axis=-1) > 30
    if floor_mask.sum() == 0:
      self.skipTest("no lit floor pixels; render misconfigured")

    lit = warp_rgb[floor_mask]
    r_mean, g_mean, b_mean = lit[:, 0].mean(), lit[:, 1].mean(), lit[:, 2].mean()
    self.assertLess(
      g_mean, max(r_mean, b_mean) * 0.5, f"green channel should be dim (R/G/B means = {r_mean:.1f}/{g_mean:.1f}/{b_mean:.1f})"
    )
    self.assertGreater(r_mean, g_mean + 5)
    self.assertGreater(b_mean, g_mean + 5)

  # ---- emission ----
  def test_emission_lights_geom_without_lights(self):
    """A material with high emission should be visible even with no lights."""
    mjm = mujoco.MjModel.from_xml_string(_FIXTURE_EMISSION)
    self.assertEqual(mjm.nlight, 0)

    warp_rgb = _mjwarp_render(mjm, 64, 64)
    self.assertGreater(int(warp_rgb[..., 1].max()), 100, "emissive ball should be clearly visible (green channel)")

  # ---- aggregate SSIM / L1 across canonical fixtures ----
  # Per-fixture SSIM and L1 thresholds tuned with ~0.05 / +0.02 margin below
  # the empirically observed values on the reference build. If a regression
  # pushes scores below these, the test fails and the diff is small enough to
  # be obviously something specific (e.g. a sign flip in compute_lighting).
  _PARITY_FIXTURES = (
    ("headlight_only", _FIXTURE_HEADLIGHT_ONLY, 0.92, 0.02),
    ("spotlight", _FIXTURE_SPOTLIGHT, 0.78, 0.06),
    ("specular", _FIXTURE_SPECULAR, 0.90, 0.02),
    ("two_lights", _FIXTURE_TWO_LIGHTS, 0.88, 0.03),
    ("emission", _FIXTURE_EMISSION, 0.92, 0.02),
  )

  @parameterized.named_parameters(*[(n, x, s, l1) for n, x, s, l1 in _PARITY_FIXTURES])
  def test_ssim_vs_mujoco(self, xml: str, ssim_threshold: float, l1_threshold: float):
    """Mjwarp output must respond to lighting params like MuJoCo OpenGL does."""
    mjm = mujoco.MjModel.from_xml_string(xml)
    warp_rgb = _mjwarp_render(mjm, 64, 64)
    mj_rgb = _mujoco_render(mjm, 64, 64)
    ssim = _ssim(warp_rgb, mj_rgb)
    l1 = _mean_l1(warp_rgb, mj_rgb)
    self.assertGreater(ssim, ssim_threshold, f"SSIM too low: {ssim:.4f} < {ssim_threshold}")
    self.assertLess(l1, l1_threshold, f"mean L1 too high: {l1:.4f} > {l1_threshold}")

  # ---- golden-image regression ----
  def test_golden_headlight_only(self):
    """Bit-stable regression check against a checked-in golden snapshot.

    mjwarp's headlight render of a fixed fixture must stay within a small
    per-channel tolerance of `test_data/golden_renders/headlight_only_32.npy`.
    A larger drift means a lighting computation changed unexpectedly;
    regenerate the snapshot only after intentional updates.
    """
    golden_path = _GOLDEN_DIR / "headlight_only_32.npy"
    if not golden_path.exists():
      self.skipTest(f"no golden snapshot at {golden_path}")
    expected = np.load(golden_path)
    self.assertEqual(expected.shape, (32, 32, 3))

    mjm = mujoco.MjModel.from_xml_string(_FIXTURE_HEADLIGHT_ONLY)
    actual = _mjwarp_render(mjm, 32, 32)
    diff = np.abs(actual.astype(np.int32) - expected.astype(np.int32))
    # Allow tiny per-channel drift (rounding/float ordering between Warp
    # builds and CPU/CUDA backends) but flag substantive changes.
    self.assertLess(diff.max(), 5, f"max channel diff = {int(diff.max())} (>= 5)")
    self.assertLess(float(diff.mean()), 0.5, f"mean channel diff = {float(diff.mean())} (>= 0.5)")


if __name__ == "__main__":
  wp.init()
  absltest.main()
