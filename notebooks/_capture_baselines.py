"""Capture renderer baseline artifacts at a specific mjwarp commit.

Renders the canonical SSIM-parity fixtures and times the resolution / world /
shadow benchmarks, writing the results as `.npy` images and a `perf.json`
file under a target directory. The notebook loads these artifacts to plot
before/after comparisons without re-rendering at notebook-execution time.

Run via the convenience wrapper `notebooks/_capture_baselines.sh`, or
manually:

    python notebooks/_capture_baselines.py \\
      --label before_phase1 \\
      --out mujoco_warp/test_data/baselines/before_phase1

The script imports mjwarp from whatever Python path it's invoked under, so
to capture a *historical* baseline you must run it from a worktree of the
target commit. The wrapper script handles that.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import mujoco
import numpy as np
import warp as wp

import mujoco_warp as mjw


# --- Canonical fixtures (intentionally inlined here so the script is hermetic
#     and produces identical inputs across baselines, even if the in-tree
#     parity-test fixtures change later). ---

FIXTURES: dict[str, str] = {
  "headlight_only": """
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
  """,
  "spotlight": """
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
  """,
  "specular": """
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
  """,
  "two_lights": """
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
  """,
  "emission": """
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
  """,
}


BENCH_XML = """
<mujoco>
  <visual>
    <headlight active="1" ambient="0.2 0.2 0.2" diffuse="0.7 0.7 0.7" specular="0.5 0.5 0.5"/>
    <map znear="0.01"/>
  </visual>
  <asset>
    <material name="shiny" specular="0.7" shininess="0.6" rgba="0.4 0.5 0.8 1"/>
  </asset>
  <worldbody>
    <camera name="cam" pos="0 -2 0.8" xyaxes="1 0 0 0 0.4 1" resolution="64 64"/>
    <light pos="1 -1 2" dir="-0.4 0.5 -1" diffuse="0.8 0.8 0.7" specular="0.5 0.5 0.5"
           ambient="0.05 0.05 0.05" attenuation="1 0 0" castshadow="false"/>
    <geom type="plane" size="3 3 0.1" rgba="0.8 0.8 0.8 1"/>
    <geom type="box"     pos="-0.6 0 0.25" size="0.2 0.2 0.25" material="shiny"/>
    <geom type="sphere"  pos=" 0   0 0.30" size="0.30" material="shiny"/>
    <geom type="capsule" pos=" 0.6 0 0.30" size="0.15 0.15" material="shiny"/>
  </worldbody>
</mujoco>
"""

SHADOW_BENCH_XML = BENCH_XML.replace('castshadow="false"', 'castshadow="true"')

# Mirror notebook constants so all baselines line up.
MSAA_NS = (1, 2, 4, 8, 16)
RESOLUTIONS = (32, 64, 96, 128, 192, 256)
WORLD_COUNTS_CPU = (1, 4, 16, 64)
WORLD_COUNTS_CUDA = (1, 4, 16, 64, 256, 1024)
SHADOW_RESOLUTIONS = (64, 128, 256)


# --- helpers ----------------------------------------------------------------


def _unpack_rgb(packed: np.ndarray) -> np.ndarray:
  r = ((packed >> 16) & 0xFF).astype(np.uint8)
  g = ((packed >> 8) & 0xFF).astype(np.uint8)
  b = (packed & 0xFF).astype(np.uint8)
  return np.stack([r, g, b], axis=-1)


def _create_rc(mjm, w, h, *, nworld=1, use_shadows=True, samples_per_pixel=1):
  """Build a render context, tolerating absence of features added later.

  Older baselines don't have `samples_per_pixel` on `create_render_context`,
  so we fall back to the legacy signature and emulate by ignoring the value.
  Similarly for `render_skybox` and other kwargs.
  """
  kwargs = dict(cam_res=(w, h), nworld=nworld, render_rgb=True, use_shadows=use_shadows)
  try:
    return mjw.create_render_context(mjm, samples_per_pixel=samples_per_pixel, **kwargs)
  except TypeError:
    return mjw.create_render_context(mjm, **kwargs)


def render_warp(mjm, w, h, *, use_shadows=True, samples_per_pixel=1) -> np.ndarray:
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd)
  rc = _create_rc(mjm, w, h, use_shadows=use_shadows, samples_per_pixel=samples_per_pixel)
  mjw.render(m, d, rc)
  return _unpack_rgb(rc.rgb_data.numpy()[0]).reshape(h, w, 3)


def render_mujoco(mjm, w, h, *, use_shadows=True) -> np.ndarray:
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  with mujoco.Renderer(mjm, height=h, width=w) as r:
    r.update_scene(mjd, camera=0)
    r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1 if use_shadows else 0
    return r.render()


def time_call(fn, *, warmup: int = 2, repeats: int = 5) -> float:
  for _ in range(warmup):
    fn()
  ts = []
  for _ in range(repeats):
    t0 = time.perf_counter()
    fn()
    ts.append(time.perf_counter() - t0)
  return float(np.median(ts))


def time_warp(mjm, w, h, *, nworld=1, use_shadows=False, samples_per_pixel=1) -> float:
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd, nworld=nworld)
  rc = _create_rc(mjm, w, h, nworld=nworld, use_shadows=use_shadows, samples_per_pixel=samples_per_pixel)

  def step():
    mjw.render(m, d, rc)
    wp.synchronize()

  return time_call(step)


def time_mujoco(mjm, w, h) -> float:
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  with mujoco.Renderer(mjm, height=h, width=w) as r:

    def step():
      r.update_scene(mjd, camera=0)
      r.render()

    return time_call(step)


# --- main capture -----------------------------------------------------------


def capture(out_dir: pathlib.Path, *, label: str) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  print(f"capturing baseline '{label}' -> {out_dir}")

  # Visual fixtures: render at each MSAA N. Older baselines that don't
  # support MSAA will silently fall back to N=1 inside _create_rc, so all
  # five files will be identical for those baselines.
  for name, xml in FIXTURES.items():
    mjm = mujoco.MjModel.from_xml_string(xml)
    for n in MSAA_NS:
      rgb = render_warp(mjm, 64, 64, use_shadows=True, samples_per_pixel=n)
      np.save(out_dir / f"fixture_{name}_n{n}.npy", rgb)
    print(f"  rendered fixture {name}")

  # Perf benchmarks.
  perf = {
    "label": label,
    "has_cuda": wp.get_cuda_device_count() > 0,
    "msaa_ns": list(MSAA_NS),
    "resolutions": list(RESOLUTIONS),
    "world_counts": list(WORLD_COUNTS_CUDA if wp.get_cuda_device_count() > 0 else WORLD_COUNTS_CPU),
    "shadow_resolutions": list(SHADOW_RESOLUTIONS),
    "mujoco_time_by_res_ms": {},
    "warp_time_by_n_by_res_ms": {},
    "warp_time_by_n_by_world_ms": {},
    "shadow_ratio_by_n_by_res": {},
  }

  mjm_bench = mujoco.MjModel.from_xml_string(BENCH_XML)
  mjm_shadow = mujoco.MjModel.from_xml_string(SHADOW_BENCH_XML)

  print("  resolution sweep…")
  for r in RESOLUTIONS:
    perf["mujoco_time_by_res_ms"][str(r)] = time_mujoco(mjm_bench, r, r) * 1000
  for n in MSAA_NS:
    perf["warp_time_by_n_by_res_ms"][str(n)] = {
      str(r): time_warp(mjm_bench, r, r, samples_per_pixel=n) * 1000 for r in RESOLUTIONS
    }
    print(f"    N={n} done")

  print("  world scaling…")
  world_counts = perf["world_counts"]
  for n in MSAA_NS:
    perf["warp_time_by_n_by_world_ms"][str(n)] = {
      str(w): time_warp(mjm_bench, 64, 64, nworld=w, samples_per_pixel=n) * 1000 for w in world_counts
    }
    print(f"    N={n} done")

  print("  shadow cost…")
  for n in MSAA_NS:
    perf["shadow_ratio_by_n_by_res"][str(n)] = {}
    for r in SHADOW_RESOLUTIONS:
      t_off = time_warp(mjm_shadow, r, r, use_shadows=False, samples_per_pixel=n)
      t_on = time_warp(mjm_shadow, r, r, use_shadows=True, samples_per_pixel=n)
      perf["shadow_ratio_by_n_by_res"][str(n)][str(r)] = {
        "off_ms": t_off * 1000,
        "on_ms": t_on * 1000,
        "ratio": t_on / t_off,
      }
    print(f"    N={n} done")

  with open(out_dir / "perf.json", "w") as f:
    json.dump(perf, f, indent=2)
  print(f"  wrote {out_dir / 'perf.json'}")


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--label", required=True, help="human-readable baseline name (e.g. before_phase1)")
  ap.add_argument("--out", required=True, type=pathlib.Path, help="output directory")
  args = ap.parse_args()
  wp.init()
  capture(args.out, label=args.label)


if __name__ == "__main__":
  main()
