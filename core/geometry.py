"""Frames, sections and ray casts. Everything 2D is a shapely (Multi)Polygon in a slice's local frame."""
from __future__ import annotations
import functools, math
import numpy as np
import shapely
import shapely.affinity
import trimesh
from shapely.geometry import Polygon, MultiPolygon, Point, LineString, MultiLineString, box
from shapely.ops import unary_union

try:
    import rtree  # noqa: F401 — trimesh's ray broad phase. Decided once: asking trimesh on every cast makes it compute
    HAVE_RTREE = True   # the bounds of every triangle before it finds out (0.15 s a cast on a 280k-face head, never cached)
except ImportError:                                       # the browser build has none
    HAVE_RTREE = False


def frame(origin, normal, up_hint=(0, 0, 1)) -> np.ndarray:
    """4x4 local->world. Local z = normal; local y = up_hint projected onto the plane; local x = y × z."""
    n = np.asarray(normal, float); n = n / np.linalg.norm(n)
    up = np.asarray(up_hint, float)
    if abs(np.dot(up, n)) > 0.999:
        up = np.array([0.0, 1, 0]) if abs(n[1]) < 0.9 else np.array([1.0, 0, 0])
    v = up - np.dot(up, n) * n; v /= np.linalg.norm(v)
    u = np.cross(v, n)
    M = np.eye(4); M[:3, 0] = u; M[:3, 1] = v; M[:3, 2] = n; M[:3, 3] = origin
    return M


def frame_xy(origin, x_axis, y_axis) -> np.ndarray:
    """4x4 from explicit in-plane axes (orthonormalised)."""
    u = np.asarray(x_axis, float); u = u / np.linalg.norm(u)
    v = np.asarray(y_axis, float); v = v - np.dot(v, u) * u; v = v / np.linalg.norm(v)
    M = np.eye(4); M[:3, 0] = u; M[:3, 1] = v; M[:3, 2] = np.cross(u, v); M[:3, 3] = origin
    return M


def rot_about(axis, deg) -> np.ndarray:
    return trimesh.transformations.rotation_matrix(math.radians(deg), axis)[:3, :3]


def to_local(M, pts, z=False) -> np.ndarray:
    """World points in M's frame: x, y on the slice (and its z, with `z`)."""
    pts = np.atleast_2d(np.asarray(pts, float))
    p = np.c_[pts, np.ones(len(pts))]
    return (np.linalg.inv(M) @ p.T).T[:, :3 if z else 2]


def to_world(M, pts2d, z=0.0) -> np.ndarray:
    pts2d = np.atleast_2d(np.asarray(pts2d, float))
    p = np.c_[pts2d, np.full(len(pts2d), z), np.ones(len(pts2d))]
    return (M @ p.T).T[:, :3]


def into(a_M, b_M, geom):
    """Geometry in b's local plane → a's local plane (both z = 0; exact for parallel frames, close enough for small tilts)."""
    T = np.linalg.inv(a_M) @ b_M
    return shapely.affinity.affine_transform(geom, [T[0, 0], T[0, 1], T[1, 0], T[1, 1], T[0, 3], T[1, 3]])


def min_dims(poly):
    """Side lengths of the minimum rotated bounding rectangle (w, h), sorted: the hull measured across each of its own
    edge directions, keeping the smallest box. Not GEOS's minimum_rotated_rectangle, which comes back as a flat line
    when the hull has several points on one straight edge (slot corners on a slanted side do that) — a 120 × 660 mm
    part measured 0.0 × 125, failed as "too thin" and was never split to fit the sheet."""
    hull = poly.convex_hull
    if hull.geom_type != "Polygon":                        # a point or a line: no width
        return 0.0, float(hull.length)
    h = np.asarray(hull.exterior.coords)
    e = np.diff(h, axis=0); e = e[np.hypot(*e.T) > 1e-12]; e /= np.hypot(*e.T)[:, None]
    w, t = np.ptp(h @ e.T, axis=0), np.ptp(h @ np.c_[-e[:, 1], e[:, 0]].T, axis=0)
    k = int(np.argmin(w * t))
    return tuple(sorted((float(w[k]), float(t[k]))))


def min_rect_angle(geom):
    """The turn (degrees) that lays the long side of GEOS's minimum rotated rectangle along x: a starting angle for
    nesting and for a label along a part. Where that rectangle degenerates to a line (see min_dims) 0 is as good a
    start as any, so this does not go through min_dims."""
    r = geom.minimum_rotated_rectangle
    if r.geom_type != "Polygon":
        return 0.0
    c = np.asarray(r.exterior.coords); e = c[1:] - c[:-1]
    i = int(np.argmax(np.hypot(e[:, 0], e[:, 1])))
    return -math.degrees(math.atan2(e[i, 1], e[i, 0]))


def even_odd(polys):
    """Rings sorted into faces and holes the even-odd way: a ring inside a ring is a hole, one inside that a face
    again — the symmetric difference of them all (a glyph's counters, a section's loops where there is no rtree to
    sort them). An invalid ring (a contour touching itself) is mended first."""
    return functools.reduce(lambda a, b: a.symmetric_difference(b), (g if g.is_valid else g.buffer(0) for g in polys))


def section_polygons(mesh: trimesh.Trimesh, M: np.ndarray) -> MultiPolygon:
    """Cross-section of the mesh on the local z=0 plane of M, as polygons (with holes) in local 2D."""
    sec = mesh.section(plane_origin=M[:3, 3], plane_normal=M[:3, 2])
    if sec is None or len(sec.entities) == 0:
        return MultiPolygon()
    path2d, _ = sec.to_2D(to_2D=np.linalg.inv(M))
    try:
        polys = list(path2d.polygons_full)
    except ImportError:                                   # trimesh sorts loops into holes with rtree; the browser build
        loops = [p for p in path2d.polygons_closed if p is not None and p.area > 1e-6]   # has none: even-odd fill is the same
        polys = [even_odd(loops)] if loops else []
    polys = [p for p in polys if p.is_valid and p.area > 1e-6]
    if not polys:
        return MultiPolygon()
    out = unary_union(polys)
    # A flat wall is two triangles, and the plane crossing their diagonal leaves a point on the straight edge — at a
    # different place on every layer. The same cube with a hole came out as 100 outlines alike in shape and unlike in
    # points, and intersecting them for the stack's aligned dowels kept every point of every layer: 393,370 vertices,
    # and the job never got past "placing slices" (reported). Off they come, 0.1 µm of tolerance. Douglas-Peucker,
    # not the topology-preserving simplify, which reads uninitialised memory in the browser's GEOS 3.12.
    lean = out.simplify(1e-4, preserve_topology=False)
    return as_multi(lean if lean.is_valid and not lean.is_empty else out)


def as_multi(g) -> MultiPolygon:
    if g is None or g.is_empty:
        return MultiPolygon()
    if isinstance(g, Polygon):
        return MultiPolygon([g])
    if isinstance(g, MultiPolygon):
        return g
    return MultiPolygon([p for p in shapely.get_parts(g) if isinstance(p, Polygon)])


def as_lines(g) -> MultiLineString:
    if g is None or g.is_empty:
        return MultiLineString()
    if isinstance(g, LineString):
        return MultiLineString([g])
    if isinstance(g, MultiLineString):
        return g
    return MultiLineString([l for l in shapely.get_parts(g) if isinstance(l, LineString)])


def label_box(text, font):
    """The room an engraved label takes at `font` mm: (w, h), about 0.62 em a character and an em of 1.4 capitals, never
    under 6 mm wide. The sheet placement and the per-piece export size a label the same way through this."""
    return max(6.0, 0.62 * font * len(text) + 2), font * 1.4


def label_spot(mp, w, h, prefer=None):
    """Where to put a w × h label on a part: the centre of a block that lies inside `prefer` (when given) or `mp` —
    tried upright and turned 90° — or, when no whole block fits, the widest spot there is, so a label is never left
    off. Returns (x, y, rotation in degrees)."""
    mp = as_multi(mp)
    if mp.is_empty:
        return (0.0, 0.0, 0)
    for cand in (prefer, mp):
        if cand is None or cand.is_empty:
            continue
        spot = _fits_block(cand, w, h)
        if spot:
            return spot
    best = max((shapely.maximum_inscribed_circle(g, 0.5) for g in mp.geoms), key=lambda c: c.length)
    return (float(best.coords[0][0]), float(best.coords[0][1]), 0)


def inside_spot(region, guides, w, h):
    """Where to engrave a w × h label on a part: inside the next layer's outline when the part carries one (`guides`,
    the stacked guide lines, in `region`'s frame), so the layer glued on top hides it, otherwise on the part's own
    widest spot. The next layer's footprint is rebuilt from the guide lines — the ring they enclose — and only used if
    the label fits inside it. A label that fits nowhere is still given the part's widest clearance rather than left
    off, so the maker trims it in the laser software instead of finding a part they cannot place. The sheet placement
    and the per-piece export both come through here, so a part's label lands in the same place in both."""
    region = as_multi(region)
    cover = None
    if guides:
        loops = [g for g in shapely.polygonize(guides).geoms if getattr(g, "area", 0) > 0]
        # a ring inside a ring is the layer above's hole (its inner edge is engraved too): the face it encloses is
        # not covered, so a label there would show through
        loops = [g for g in loops if not any(o is not g and Polygon(o.exterior).contains(g) for o in loops)]
        if loops:
            under = shapely.union_all(loops).intersection(region)
            if not under.is_empty:
                cover = under
    return label_spot(region, w, h, cover)                          # the next layer's footprint first, then the whole part


def _fits_block(mp, w, h):
    """Centre and rotation of a w × h block lying inside `mp`, or None. The largest inscribed circle's centre is where
    a block has the most clearance, so it is tried first; holes are respected, since `covers` sees them."""
    for g in sorted(as_multi(mp).geoms, key=lambda p: -p.area):
        px, py = shapely.maximum_inscribed_circle(g, 0.5).coords[0]
        for rot in (0, 90):
            bw, bh = (w, h) if rot == 0 else (h, w)
            if g.covers(box(px - bw / 2, py - bh / 2, px + bw / 2, py + bh / 2)):
                return (float(px), float(py), rot)
    return None


def line_segments_in_mesh(mesh, p0, d, span) -> list[tuple[float, float]]:
    """Inside-segments (s_in, s_out) of the line p0 + s*d against the mesh."""
    d = np.asarray(d, float); d = d / np.linalg.norm(d)
    origin = np.asarray(p0, float) - d * span
    if HAVE_RTREE:
        locs, _, _ = mesh.ray.intersects_location([origin], [d], multiple_hits=True)   # rtree broad phase, cached on the mesh
    else:                                                 # the browser build: a broad phase of our own — only the triangles
        from types import SimpleNamespace                 # whose bounding sphere the line passes through are tested
        from trimesh.ray.ray_triangle import ray_triangle_id
        cache = mesh.metadata.setdefault("lamina_spheres", {})
        if "c" not in cache:                              # once per mesh
            tri = mesh.triangles; c = tri.mean(1)
            cache.update(c=c, r2=(np.linalg.norm(tri - c[:, None], axis=2).max(1) + 1e-6) ** 2)
        v = cache["c"] - origin; t = v @ d
        near = np.flatnonzero(np.einsum("ij,ij->i", v, v) - t * t <= cache["r2"])   # squared distance to the line
        tree = SimpleNamespace(bounds=mesh.bounds, intersection=lambda b: near)
        _, _, locs = ray_triangle_id(mesh.triangles, [origin], [d], triangles_normal=mesh.face_normals, tree=tree, multiple_hits=True)
    if len(locs) == 0:
        return []
    s = np.sort((locs - p0) @ d)
    s = s[np.r_[True, np.diff(s) > 1e-6]]
    if len(s) % 2:
        s = s[:-1]
    return [(float(s[i]), float(s[i + 1])) for i in range(0, len(s), 2) if s[i + 1] - s[i] > 0.3]


def plane_plane_line(Ma, Mb):
    """Intersection line (p0, d) of the z=0 planes of frames Ma, Mb; None if parallel."""
    na, nb = Ma[:3, 2], Mb[:3, 2]
    d = np.cross(na, nb)
    if np.linalg.norm(d) < 1e-6:
        return None
    d = d / np.linalg.norm(d)
    A = np.vstack([na, nb, d]); b = np.array([na @ Ma[:3, 3], nb @ Mb[:3, 3], 0.0])
    return np.linalg.solve(A, b), d


def rect_along(a, b, width) -> Polygon:
    """Rectangle of `width` centred on the segment a->b (2D)."""
    return LineString([a, b]).buffer(width / 2, cap_style=2)


def circle(c, d) -> Polygon:
    return Point(c).buffer(d / 2, quad_segs=24)


def dowel(c, d, shape="round") -> Polygon:
    """Dowel hole of size d at c. Shapes follow Slicer: round, square, pencil (hexagon), cross, hslot, vslot."""
    x, y = c; r = d / 2
    if shape == "square":
        return Polygon([(x - r, y - r), (x + r, y - r), (x + r, y + r), (x - r, y + r)])
    if shape == "pencil":
        return Polygon([(x + r * math.cos(a), y + r * math.sin(a)) for a in np.arange(6) * math.pi / 3])
    if shape == "cross":
        w = d / 3
        return unary_union([rect_along((x - r, y), (x + r, y), w), rect_along((x, y - r), (x, y + r), w)])
    if shape == "hslot":
        return LineString([(x - r, y), (x + r, y)]).buffer(r / 2)
    if shape == "vslot":
        return LineString([(x, y - r), (x, y + r)]).buffer(r / 2)
    return circle(c, d)


def hole_reach(d, shape="round") -> float:
    """How far a dowel hole reaches from its centre: d/2 round, 0.71 d at a square's corner, 0.75 d at a slot's end."""
    return max(math.hypot(x, y) for x, y in dowel((0, 0), d, shape).exterior.coords)


LEVERAGE_OK = 0.8         # dowels that hold a piece: its far edge moves less than the play in their holes. 1.0 let
                           # rods carried up from a smaller layer stand in for three around the edge of a disc (0.69)


def dowel_leverage(piece, pts) -> float:
    """How well the dowels through a piece hold it: the farthest the piece reaches from their middle, over the widest
    spacing between two of them. The play in the holes lets the piece turn by about play / spacing, which its far edge
    feels × that reach — so this is how many times the holes' play the far edge can shift. Two dowels 46 mm apart in
    the middle of a 200 mm disc: 2.2. The same two near its edges: 0.55; three around near the edge: 0.64. One dowel
    is a hinge: inf. (The farthest point of a polygon from a point is one of its vertices, so the exterior is exact.)"""
    q = np.asarray(pts, float).reshape(-1, 2)
    if len(q) < 2:
        return math.inf
    spacing = np.linalg.norm(q[:, None] - q[None], axis=2).max()
    reach = np.linalg.norm(np.asarray(piece.exterior.coords) - q.mean(0), axis=1).max()
    return float(reach / spacing)


def poly_coords(mp: MultiPolygon) -> list:
    """[[exterior, hole, hole...], ...] as plain lists for JSON (bulk coordinate extraction: panels have 100s of holes)."""
    out = []
    for p in mp.geoms:
        rings = shapely.get_rings(p)
        coords = np.round(shapely.get_coordinates(rings), 4)
        parts = np.split(coords, np.cumsum(shapely.get_num_coordinates(rings))[:-1])
        out.append([r[:-1].tolist() for r in parts])
    return out


def line_coords(ml) -> list:
    return [list(map(list, np.round(np.asarray(l.coords), 4))) for l in as_lines(ml).geoms]
