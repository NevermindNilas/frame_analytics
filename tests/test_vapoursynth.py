"""VapourSynth binding contract tests.

The metrics themselves are gated in ``tests/validate.py``; what is checked here
is the glue -- that a frame arrives at the kernel unchanged whatever the
format's bit depth, subsampling or colour family says, that the frame
properties are named what ``VapourSynth-VMAF`` names them, and that the
argument surface rejects what it cannot compute instead of guessing.

Every score is compared against the same call made directly on the numpy array
the clip was built from, so a conversion bug shows up as a number, not as a
crash.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import frame_analytics as fa

vs = pytest.importorskip("vapoursynth")
fa_vs = pytest.importorskip("frame_analytics.vapoursynth")

core = vs.core

H, W = 224, 288          # >= 176 px, so MS-SSIM gets its five scales
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _pair(h=H, w=W, planes=3, peak=255, quantise=True, seed=5):
    """A reference/distorted pair as ``(planes, h, w)`` float64 arrays."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    base = 0.5 + 0.35 * np.sin(xx / 9.0) * np.cos(yy / 11.0)
    a = np.stack([np.clip(base + 0.05 * c, 0, 1) for c in range(planes)])
    b = np.clip(a + 0.03 * rng.standard_normal(a.shape), 0, 1)
    a, b = a * peak, b * peak
    return (a.round(), b.round()) if quantise else (a, b)


def _clip(arrays, fmt, length=2):
    """A clip whose every frame is ``arrays`` (one 2-D array per plane).

    Dimensions come from the first plane, so a test can hand this any size.
    """
    h, w = np.shape(arrays[0])
    blank = core.std.BlankClip(width=w, height=h, format=fmt, length=length)
    dtype = np.float32 if blank.format.sample_type == vs.FLOAT else \
        (np.uint8 if blank.format.bits_per_sample == 8 else np.uint16)
    planes = [np.ascontiguousarray(a, dtype=dtype) for a in arrays]

    def sel(n, f):
        out = f.copy()
        for p, plane in enumerate(planes):
            np.asarray(out[p])[:] = plane
        return out

    return core.std.ModifyFrame(blank, blank, sel)


def _sub(arr, fmt_id):
    """Subsample chroma the way the format wants it, by decimation."""
    fmt = core.get_video_format(fmt_id)
    out = [arr[0]]
    for a in arr[1:]:
        out.append(a[::1 << fmt.subsampling_h, ::1 << fmt.subsampling_w])
    return out


def _t(a, device="cpu", dtype=torch.float64):
    """One plane as ``(1, 1, H, W)``; a stack of planes as ``(1, C, H, W)``."""
    t = torch.as_tensor(np.asarray(a), dtype=dtype, device=device)
    return t[None, None] if t.ndim == 2 else t[None]


def _props(clip, n=0):
    frame = clip.get_frame(n)
    return {k: v for k, v in frame.props.items() if not k.startswith("_")}


# --------------------------------------------------------------------------- #
# property names and values
# --------------------------------------------------------------------------- #

def test_prop_names_match_the_vmaf_plugin():
    ref, dist = _pair()
    r, d = _clip(_sub(ref, vs.YUV420P8), vs.YUV420P8), \
        _clip(_sub(dist, vs.YUV420P8), vs.YUV420P8)
    props = _props(fa_vs.Metric(r, d, [0, 2, 3]))
    assert set(props) == {"psnr_y", "psnr_cb", "psnr_cr",
                          "float_ssim", "float_ms_ssim"}


def test_rgb_planes_are_named_rgb():
    ref, dist = _pair()
    props = _props(fa_vs.Metric(_clip(ref, vs.RGB24), _clip(dist, vs.RGB24), "psnr"))
    assert set(props) == {"psnr_r", "psnr_g", "psnr_b"}


def test_gray_has_one_plane():
    ref, dist = _pair(planes=1)
    props = _props(fa_vs.Metric(_clip(ref, vs.GRAY8), _clip(dist, vs.GRAY8),
                                ["psnr", "ssim"]))
    assert set(props) == {"psnr_y", "float_ssim"}


def test_names_and_ids_agree():
    ref, dist = _pair()
    r, d = _clip(ref, vs.RGB24), _clip(dist, vs.RGB24)
    by_id = _props(fa_vs.Metric(r, d, [0, 2, 3, 5, 6, 7, 8]))
    by_name = _props(fa_vs.Metric(r, d, ["psnr", "ssim", "ms_ssim", "gmsd",
                                         "gms", "mse", "l1"]))
    assert by_id == by_name


@pytest.mark.parametrize("device", DEVICES)
def test_values_match_a_direct_call_yuv(device):
    """Y/Cb/Cr at their own resolutions, luma-only SSIM -- as libvmaf does it."""
    ref, dist = _pair()
    rp, dp = _sub(ref, vs.YUV420P8), _sub(dist, vs.YUV420P8)
    scored = fa_vs.Metric(_clip(rp, vs.YUV420P8), _clip(dp, vs.YUV420P8),
                          [0, 2, 3, 5], device=device)
    props = _props(scored)

    for name, i in (("psnr_y", 0), ("psnr_cb", 1), ("psnr_cr", 2)):
        want = float(fa.psnr(_t(rp[i]), _t(dp[i]), data_range=255.0))
        assert props[name] == pytest.approx(want, rel=1e-6)

    y_r, y_d = _t(rp[0]), _t(dp[0])
    assert props["float_ssim"] == pytest.approx(
        float(fa.ssim(y_r, y_d, data_range=255.0)), rel=1e-6)
    assert props["float_ms_ssim"] == pytest.approx(
        float(fa.ms_ssim(y_r, y_d, data_range=255.0)), rel=1e-6)
    assert props["gmsd"] == pytest.approx(
        float(fa.gmsd(y_r, y_d, data_range=255.0)), rel=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_values_match_a_direct_call_rgb(device):
    """RGB feeds all three channels to the single-value metrics."""
    ref, dist = _pair()
    scored = fa_vs.Metric(_clip(ref, vs.RGB24), _clip(dist, vs.RGB24),
                          ["ssim", "lpips", "ssimulacra2"], device=device)
    props = _props(scored)
    r = torch.as_tensor(ref, dtype=torch.float32, device=device)[None]
    d = torch.as_tensor(dist, dtype=torch.float32, device=device)[None]
    assert props["float_ssim"] == pytest.approx(
        float(fa.ssim(r, d, data_range=255.0)), rel=1e-5)
    assert props["lpips"] == pytest.approx(
        float(fa.lpips(r, d, data_range=255.0)), rel=1e-4)
    assert props["ssimulacra2"] == pytest.approx(
        float(fa.ssimulacra2(r, d, data_range=255.0)), rel=1e-4)


@pytest.mark.parametrize("fmt,bits", [(vs.YUV420P10, 10), (vs.YUV444P12, 12),
                                      (vs.YUV420P16, 16)])
def test_high_bit_depth(fmt, bits):
    """data_range follows the bit depth, and uint16 survives the trip."""
    peak = (1 << bits) - 1
    ref, dist = _pair(peak=peak)
    rp, dp = _sub(ref, fmt), _sub(dist, fmt)
    props = _props(fa_vs.Metric(_clip(rp, fmt), _clip(dp, fmt), ["psnr", "ssim"]))
    assert props["psnr_y"] == pytest.approx(
        float(fa.psnr(_t(rp[0]), _t(dp[0]), data_range=float(peak))), rel=1e-6)
    assert props["float_ssim"] == pytest.approx(
        float(fa.ssim(_t(rp[0]), _t(dp[0]), data_range=float(peak))), rel=1e-6)


def test_float_clip():
    """RGBS is 0..1, so data_range is 1.0 and nothing is quantised."""
    ref, dist = _pair(peak=1, quantise=False)
    props = _props(fa_vs.Metric(_clip(ref, vs.RGBS), _clip(dist, vs.RGBS), "psnr"))
    want = float(fa.psnr(_t(ref[0], dtype=torch.float32),
                         _t(dist[0], dtype=torch.float32), data_range=1.0))
    assert props["psnr_r"] == pytest.approx(want, rel=1e-5)


def test_odd_width_padded_stride():
    """A width VapourSynth has to pad still reaches the kernel unpadded."""
    ref, dist = _pair(w=W - 5, planes=1)
    props = _props(fa_vs.Metric(_clip(ref, vs.GRAY8), _clip(dist, vs.GRAY8), "psnr"))
    want = float(fa.psnr(_t(ref[0]), _t(dist[0]), data_range=255.0))
    assert props["psnr_y"] == pytest.approx(want, rel=1e-6)


def test_identical_pair():
    ref, _ = _pair()
    r = _clip(ref, vs.RGB24)
    props = _props(fa_vs.Metric(r, r, ["psnr", "ssim", "ssimulacra2"]))
    assert props["psnr_r"] == float("inf")
    assert props["float_ssim"] == pytest.approx(1.0, abs=1e-6)   # float32 rounding
    assert props["ssimulacra2"] == pytest.approx(100.0, abs=1e-6)


def test_existing_props_survive():
    ref, dist = _pair()
    d = core.std.SetFrameProp(_clip(dist, vs.RGB24), prop="_Matrix", intval=0)
    props = fa_vs.Metric(_clip(ref, vs.RGB24), d, "psnr").get_frame(0).props
    assert props["_Matrix"] == 0
    assert "psnr_r" in props


# --------------------------------------------------------------------------- #
# knobs
# --------------------------------------------------------------------------- #

def test_options_reach_the_metric():
    ref, dist = _pair()
    r, d = _clip(ref, vs.RGB24), _clip(dist, vs.RGB24)
    default = _props(fa_vs.Metric(r, d, "ssim"))["float_ssim"]
    win7 = _props(fa_vs.Metric(r, d, "ssim",
                               options={"ssim": {"win_size": 7}}))["float_ssim"]
    assert default != win7
    want = float(fa.ssim(_t(ref, dtype=torch.float32), _t(dist, dtype=torch.float32),
                         data_range=255.0, win_size=7))
    assert win7 == pytest.approx(want, rel=1e-5)


def test_crop_border_and_data_range():
    ref, dist = _pair(planes=1)
    r, d = _clip(ref, vs.GRAY8), _clip(dist, vs.GRAY8)
    props = _props(fa_vs.Metric(r, d, "psnr", crop_border=8, data_range=219.0))
    want = float(fa.psnr(_t(ref[0]), _t(dist[0]), data_range=219.0, crop_border=8))
    assert props["psnr_y"] == pytest.approx(want, rel=1e-6)


def test_single_metric_wrappers():
    ref, dist = _pair()
    r, d = _clip(ref, vs.RGB24), _clip(dist, vs.RGB24)
    assert _props(fa_vs.SSIM(r, d)) == _props(fa_vs.Metric(r, d, "ssim"))
    assert set(_props(fa_vs.PSNR(r, d))) == {"psnr_r", "psnr_g", "psnr_b"}


def test_numpy_integer_feature():
    """A numpy scalar is not a Python int; it is accepted either way."""
    ref, dist = _pair()
    r, d = _clip(ref, vs.RGB24), _clip(dist, vs.RGB24)
    assert _props(fa_vs.Metric(r, d, np.int64(2))) == _props(fa_vs.Metric(r, d, 2))
    assert _props(fa_vs.Metric(r, d, [np.int64(2)])) == _props(fa_vs.Metric(r, d, 2))


def test_available_features():
    feats = fa_vs.available_features()
    assert feats["psnr"] == 0 and feats["ssim"] == 2 and feats["ms_ssim"] == 3
    assert 1 not in feats.values() and 4 not in feats.values()


# --------------------------------------------------------------------------- #
# rejections
# --------------------------------------------------------------------------- #

def test_libvmaf_only_features_are_named_in_the_error():
    ref, dist = _pair()
    r, d = _clip(ref, vs.RGB24), _clip(dist, vs.RGB24)
    for bad, name in ((1, "psnr_hvs"), (4, "ciede2000")):
        with pytest.raises(ValueError, match=name):
            fa_vs.Metric(r, d, bad)


def test_rgb_only_features_reject_yuv():
    ref, dist = _pair()
    r = _clip(_sub(ref, vs.YUV420P8), vs.YUV420P8)
    d = _clip(_sub(dist, vs.YUV420P8), vs.YUV420P8)
    for feat in ("lpips", "ssimulacra2"):
        with pytest.raises(vs.Error, match="needs RGB"):
            fa_vs.Metric(r, d, feat)


def test_mismatched_clips():
    ref, dist = _pair()
    r = _clip(ref, vs.RGB24)
    with pytest.raises(vs.Error, match="same format"):
        fa_vs.Metric(r, _clip(_sub(dist, vs.YUV444P8), vs.YUV444P8), "psnr")
    with pytest.raises(vs.Error, match="same number of frames"):
        fa_vs.Metric(r, _clip(dist, vs.RGB24, length=3), "psnr")
    with pytest.raises(vs.Error, match="constant"):
        fa_vs.Metric(r, _variable_clip(), "psnr")


def _variable_clip():
    """Two sizes spliced together -- ``width`` reads 0 on the result."""
    a = core.std.BlankClip(width=W, height=H, format=vs.RGB24, length=1)
    b = core.std.BlankClip(width=W // 2, height=H, format=vs.RGB24, length=1)
    return core.std.Splice([a, b], mismatch=True)


def test_bad_feature_arguments():
    ref, dist = _pair()
    r, d = _clip(ref, vs.RGB24), _clip(dist, vs.RGB24)
    with pytest.raises(ValueError, match="feature is required"):
        fa_vs.Metric(r, d, None)
    with pytest.raises(ValueError, match="unknown feature"):
        fa_vs.Metric(r, d, "vmaf")
    with pytest.raises(ValueError, match="unknown feature id"):
        fa_vs.Metric(r, d, 99)
    with pytest.raises(TypeError):
        fa_vs.Metric(r, d, [1.5])


def test_dimension_mismatch():
    ref, dist = _pair()
    r = _clip(ref, vs.RGB24)
    small = core.std.BlankClip(width=W // 2, height=H, format=vs.RGB24, length=2)
    with pytest.raises(vs.Error, match="same dimensions"):
        fa_vs.Metric(r, small, "psnr")


# --------------------------------------------------------------------------- #
# logging and pooling
# --------------------------------------------------------------------------- #

def _scored(length=3):
    ref, dist = _pair()
    return (_clip(ref, vs.RGB24, length=length),
            _clip(dist, vs.RGB24, length=length))


def _drain(clip):
    for _ in clip.frames(close=True):
        pass


@pytest.mark.parametrize("fmt,ext", [(0, "xml"), (1, "json"), (2, "csv"), (3, "sub")])
def test_log_written(tmp_path, fmt, ext):
    r, d = _scored()
    path = tmp_path / f"scores.{ext}"
    _drain(fa_vs.Metric(r, d, ["psnr", "ssim"], log_path=str(path), log_format=fmt))
    text = path.read_text(encoding="utf-8")
    assert "float_ssim" in text
    if fmt == 0:
        assert text.count("<frame ") == 3 and "pooled_metrics" in text
    if fmt == 1:
        doc = json.loads(text)                       # i.e. it is valid JSON
        assert len(doc["frames"]) == 3
        assert doc["frames"][0]["metrics"]["psnr_r"] == pytest.approx(
            doc["pooled_metrics"]["psnr_r"]["mean"])
    if fmt == 2:
        assert text.splitlines()[0].startswith("frameNum,psnr_r")
        assert len(text.splitlines()) == 4
    if fmt == 3:                                 # MicroDVD, as libvmaf writes it
        assert text.splitlines()[0].startswith("{0}{1}frame: 0|")
        assert len(text.splitlines()) == 3
        assert "\\N" not in text                 # a MicroDVD line break is "|"


def test_log_of_an_identical_pair_is_still_valid_json(tmp_path):
    ref, _ = _pair()
    r = _clip(ref, vs.RGB24, length=2)
    path = tmp_path / "identical.json"
    _drain(fa_vs.Metric(r, r, "psnr", log_path=str(path), log_format=1))
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["frames"][0]["metrics"]["psnr_r"] == "inf"


def test_bad_log_format(tmp_path):
    r, d = _scored()
    with pytest.raises(ValueError, match="log_format"):
        fa_vs.Metric(r, d, "psnr", log_path=str(tmp_path / "x"), log_format=7)


def test_pooled_scores():
    r, d = _scored(length=4)
    scored = fa_vs.Metric(r, d, ["psnr", "ssim"])
    pooled = fa_vs.pooled_scores(scored)
    assert set(pooled) == {"psnr_r", "psnr_g", "psnr_b", "float_ssim"}
    one = _props(scored)["float_ssim"]
    stats = pooled["float_ssim"]
    for key in ("min", "max", "mean", "harmonic_mean"):
        assert stats[key] == pytest.approx(one, rel=1e-9)
    assert fa_vs.pooled_scores(scored, props=["float_ssim"]).keys() == {"float_ssim"}


def test_no_readonly_warning_on_stderr(tmp_path):
    """torch's non-writable-array warning must not reach the user's terminal.

    A frame's memoryview is read-only and ``torch.from_numpy`` says so every
    time.  It has to be filtered inside the module, so this runs a real script
    in a real interpreter rather than under pytest's own warning capture, which
    resets the filters we are testing.
    """
    script = tmp_path / "run.py"
    script.write_text(
        "import numpy as np, vapoursynth as vs\n"
        "from frame_analytics import vapoursynth as fa_vs\n"
        "c = vs.core.std.BlankClip(width=64, height=64, format=vs.GRAY8, length=4)\n"
        "for _ in fa_vs.Metric(c, c, 'mse').frames(close=True):\n"
        "    pass\n",
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root)}   # the checkout, not any install
    proc = subprocess.run([sys.executable, str(script)], capture_output=True,
                          text=True, cwd=str(root), env=env)
    assert proc.returncode == 0, proc.stderr
    assert "not writable" not in proc.stderr
