"""The drawn spacing curve: the spline that never overshoots, the positions it gives with the count fixed or free,
and each technique spreading its slices by it."""
import pathlib

import numpy as np
import pytest

from core.plan import build
from core.spacing import spaced, spline

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"


def test_the_spline_passes_through_its_points_and_never_overshoots_them():
    pts = [[0, 20], [0.3, 8], [0.65, 8], [1, 25]]
    f = spline(pts)
    assert all(abs(f(t) - s) < 1e-9 for t, s in pts)
    ts = np.linspace(0, 1, 401)
    assert all(8 - 1e-9 <= f(t) <= 25 + 1e-9 for t in ts)                      # within the points' range: no overshoot
    assert all(abs(f(t) - 8) < 1e-9 for t in np.linspace(0.3, 0.65, 30))      # a flat stretch stays flat
    assert all(f(a) >= f(b) - 1e-9 for a, b in zip(ts, ts[1:]) if b <= 0.3)    # monotone between points
    g = spline([[0, 15], [0.5, 40], [1, 15]], periodic=True)
    assert abs(g(0) - g(1)) < 1e-9 and abs(g(0.5) - 40) < 1e-9


def test_positions_span_the_length_with_the_count_fixed_and_follow_the_curve_when_free():
    pts = [[0, 20], [0.3, 8], [0.65, 8], [1, 25]]
    pos, k = spaced(pts, 200, 3, 30, count=14)
    gaps = np.diff(pos)
    assert len(pos) == 14 and pos[0] > 0 and pos[-1] < 200
    assert abs(pos[0] - k * 20 / 2) < 1e-6 and abs(200 - pos[-1] - k * 25 / 2) < 1e-6   # half a spacing in at each end
    assert gaps[0] > gaps[5] and gaps[-1] > gaps[5] and abs(gaps[5] - 8 * k) < 1e-6       # the drawn proportions, scaled
    free, k1 = spaced(pts, 200, 3, 30)
    assert k1 == 1.0 and abs(np.diff(free)[len(free) // 2] - 8) < 0.5 and free[-1] <= 200
    ang, ka = spaced([[0, 15], [0.5, 40], [1, 15]], 180, 5, 45, count=8, periodic=True)
    closing = ka * spline([[0, 15], [0.5, 40], [1, 15]], periodic=True)(ang[-1] / 180)
    assert len(ang) == 8 and ang[0] == 0 and abs((180 - ang[-1]) - closing) < 1e-6              # the closing step lands on 180°
    assert spaced([[0, 10], [1, 10]], 100, 1, 50, count=5)[0] == pytest.approx([10, 30, 50, 70, 90])
    pos, k = spaced([[0, 100]], 100, 1, 20, count=3)                                          # one point: a flat curve, held to the range
    assert k > 1 and np.diff(pos) == pytest.approx([20 * k, 20 * k])                           # then scaled so three span the length


def test_each_technique_spreads_its_slices_by_the_curve():
    dense_middle = {"spacing_vary": True, "spacing_curve": [[0, 20], [0.5, 6], [1, 20]], "spacing_min": 3, "spacing_max": 30, "autofix": "off"}
    plan = build(EXAMPLES / "egg.stl", "stacked", {"distribution": "count", "count": 12, **dense_middle})
    z = sorted(s["M"][2][3] for s in plan["slices"] if s["group"] == "S")
    gaps = np.diff(z)
    assert len(z) == 12 and gaps[0] > 2 * gaps[5] and gaps[-1] > 2 * gaps[5]
    assert plan["spacing"][0]["param"] == "spacing_curve" and len(plan["spacing"][0]["positions"]) == 12

    plan = build(EXAMPLES / "torus.stl", "interlocked", {"nx": 7, "ny": 5, **dense_middle, "spacing_curve_y": [[0, 6], [1, 30]], "spacing_max": 40})
    by = {g: sorted(np.asarray(s["M"])[:3, 3] @ np.asarray(s["M"])[:3, 2] for s in plan["slices"] if s["group"] == g) for g in "XY"}
    assert np.diff(by["X"])[0] > 2 * np.diff(by["X"])[2]                          # the first family dense in the middle
    assert np.all(np.diff(np.diff(by["Y"])) > -1e-6)                              # the second opening up along its own axis
    assert {s["param"] for s in plan["spacing"]} == {"spacing_curve", "spacing_curve_y"}

    plan = build(EXAMPLES / "egg.stl", "radial", {"count": 8, "ring_count": 3, "spacing_vary": True, "spacing_curve": [[0, 10], [0.5, 40], [1, 10]],
                                                 "spacing_min": 5, "spacing_max": 60, "autofix": "off"})
    sp = plan["spacing"][0]
    assert sp["unit"] == "deg" and len(sp["positions"]) == 8 and sp["positions"][0] == 0
    gaps = np.diff(sp["positions"] + [180.0])
    assert gaps[0] < gaps[3] and abs(gaps[0] - gaps[-1]) < 2                      # dense at 0°, which is where 180° comes back to

    plan = build(EXAMPLES / "pyramid.stl", "curve", {"plane": "xz", "curve": [[-100, -30], [0, 0], [100, 30]], "count": 8, "spines": 1,
                                                    "spacing_vary": True, "spacing_curve": [[0, 40], [0.5, 10], [1, 40]], "spacing_min": 3, "spacing_max": 60, "autofix": "off"})
    assert len([s for s in plan["slices"] if s["label"].startswith("R-")]) == 8
    g = np.diff(plan["spacing"][0]["positions"])
    assert g[0] > 2 * g[3]


def test_slices_closer_than_the_material_are_an_error_and_a_scaled_curve_is_noted():
    plan = build(EXAMPLES / "egg.stl", "stacked", {"distribution": "count", "count": 60, "thickness": 3, "spacing_vary": True,   # 60 in 150 mm: 2.5 apart
                                                  "spacing_curve": [[0, 4], [0.5, 3], [1, 4]], "spacing_min": 3, "spacing_max": 30, "autofix": "off"})
    assert any("overlap" in e for e in plan["errors"]), plan["errors"]
    plan = build(EXAMPLES / "egg.stl", "stacked", {"distribution": "count", "count": 4, "spacing_vary": True,
                                                  "spacing_curve": [[0, 3], [1, 3]], "spacing_min": 3, "spacing_max": 30, "autofix": "off"})
    assert plan["spacing"][0]["scale"] > 1.5 and any("disagree" in n for n in plan["notes"])
