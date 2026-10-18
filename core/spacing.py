"""Non-uniform spacing: a curve the user draws of the distance between neighbouring slices against the position along
the axis — (t, s) control points, t from 0 at the model's start to 1 at its end, s in mm (or degrees for radial's
planes round the axis) — and the slice positions it gives.

The curve between the points is a monotone cubic spline (Fritsch–Carlson): it never overshoots a point, so what is
drawn is what is cut, and a flat stretch stays flat. Beyond the first and last point it is flat. For radial the
axis is the angle round the fan, 0..180° (a plane at θ and at θ+180° is the same plane), and the curve is periodic:
the last point is the first, so the fan closes on itself with no jump.

With the count fixed, the drawn spacings are relative: every one is scaled by one factor so exactly that many slices
span the length, the first half a spacing in from the start and the last half a spacing from the end (round the
axis: the first plane at 0 and the closing step landing on 180°). With the curve fixed instead, the spacings are
taken as drawn and the count follows.
"""
from __future__ import annotations
import math


def spline(points, periodic=False):
    """s(t) through the control points, monotone between them; flat beyond the ends; periodic ties the ends together."""
    pts = sorted((float(a), float(b)) for a, b in points)
    if not pts:
        return lambda t: 0.0
    if len(pts) == 1 or pts[0][0] == pts[-1][0]:
        v = pts[0][1]
        return lambda t: v
    if periodic:
        pts[-1] = (pts[-1][0], pts[0][1])
    x = [a for a, _ in pts]; y = [b for _, b in pts]; n = len(pts)
    h = [x[i + 1] - x[i] for i in range(n - 1)]
    d = [(y[i + 1] - y[i]) / h[i] if h[i] > 1e-12 else 0.0 for i in range(n - 1)]
    m = [0.0] * n
    for i in range(1, n - 1):
        m[i] = 0.0 if d[i - 1] * d[i] <= 0 else (d[i - 1] + d[i]) / 2
    if periodic:
        m[0] = m[-1] = 0.0 if d[-1] * d[0] <= 0 else (d[-1] + d[0]) / 2
    else:
        m[0], m[-1] = d[0], d[-1]
    for i in range(n - 1):                               # Fritsch–Carlson: tangents limited so no segment overshoots
        if d[i] == 0:
            m[i] = m[i + 1] = 0.0; continue
        a, b = m[i] / d[i], m[i + 1] / d[i]
        if a * a + b * b > 9:
            tau = 3 / math.sqrt(a * a + b * b); m[i], m[i + 1] = tau * a * d[i], tau * b * d[i]

    def f(t):
        t = min(x[-1], max(x[0], float(t)))
        i = 0
        while i < n - 2 and t > x[i + 1]:
            i += 1
        if h[i] <= 1e-12:
            return y[i]
        u = (t - x[i]) / h[i]; u2, u3 = u * u, u * u * u
        return (2 * u3 - 3 * u2 + 1) * y[i] + (u3 - 2 * u2 + u) * h[i] * m[i] + (-2 * u3 + 3 * u2) * y[i + 1] + (u3 - u2) * h[i] * m[i + 1]
    return f


def spaced(points, length, lo, hi, count=None, periodic=False, cap=500):
    """Slice positions from `points` along a length (0 at the start): with `count`, that many, the curve scaled to fit;
    without, as drawn until the length is used, at most `cap`. The curve is held within lo..hi. Returns (positions,
    scale factor): a factor far from 1 means the count and the drawn spacings disagree."""
    f0 = spline(points, periodic); s = lambda t: min(hi, max(lo, f0(t)))   # noqa: E731
    if length <= 0 or hi <= 0:
        return [], 1.0

    def walk(k, n=None):
        """Positions at k × the drawn spacing, `n` of them, or as many as fit."""
        out, p = [], 0.0 if periodic else k * s(0) / 2
        while len(out) < (n if n is not None else cap):
            out.append(p)
            step = k * s(p / length)
            if n is None and (p + step > length - 1e-9 if periodic else p + step > length - k * s(1) / 2 + 1e-9):
                break
            p += step
        return out, p                                    # p: where the step after the last slice lands

    if count is None:
        return walk(1.0)[0], 1.0
    count = max(1, int(count))

    def landing(k):                                      # the end of the walk, to land on the length
        pts, p = walk(k, count)
        return p if periodic else pts[-1] + k * s(1) / 2
    lo_k, hi_k = 1e-4, 1e4
    for _ in range(80):
        k = math.sqrt(lo_k * hi_k)
        if landing(k) > length:
            hi_k = k
        else:
            lo_k = k
    k = math.sqrt(lo_k * hi_k)
    return walk(k, count)[0], k
