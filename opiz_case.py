#!/usr/bin/env python3
"""
Orange Pi Zero LTS case — bottom shell + lid, with a side bay for the
right-angle 26-pin header connector and side clearance for the 13-pin header.

Pure numpy generator: solids are CSG of boxes/cylinders, evaluated on a
rectilinear grid (exact for boxes, fine staircase for round features) and
written as watertight binary STL.

Board measurements (PCB 46.1 x 48.1, M3 holes 3.1 mm from each edge, port
positions) follow Phil Dubach's measured Orange Pi Zero case
(github.com/phildubach/OrangePiZeroCase, CC BY-NC-SA 4.0).

Coordinates (top view, components up): PCB corner at x=0,y=0.
  y=0 edge   : micro-USB (top side), microSD (underside)
  y=48.1 edge: USB-A (vertical), Ethernet
  x=46.1 edge: 26-pin header -> side connector bay
  x=0 edge   : 13-pin header (pins sticking out sideways)
"""
import numpy as np, struct, sys, os

# ---------------- parameters (mm) ----------------
PCB_W, PCB_L, PCB_T = 46.1, 48.1, 1.6
GAP      = 0.3     # clearance PCB edge -> inner wall
WALL     = 2.0
FLOOR    = 2.0
LID_T    = 2.0
PCB_BELOW = 4.0    # standoff height (underside clearance; microSD is 1.9)
PCB_ABOVE = 15.0   # PCB top -> underside of lid (Ethernet is 13.1)

# 26-pin side connector bay (x=46.1 side). Measure your connector and adjust.
BAY_OUT    = 16.0  # how far the connector sticks out past the PCB edge
BAY_SLOT   = True  # slot in the bay end wall for a cable / ribbon
BAY_SLOT_Y = (6.0, 42.0)          # slot span along the edge (PCB coords)
BAY_SLOT_Z = (-3.0, 9.0)          # slot height, relative to PCB top
# 13-pin header side (x=0): clearance for sideways pins
PIN13_CLEAR = 9.0

HOLE_INSET = 3.1   # mounting hole centre from PCB edges
HOLES = [(HOLE_INSET, HOLE_INSET), (HOLE_INSET, PCB_L - HOLE_INSET),
         (PCB_W - HOLE_INSET, HOLE_INSET), (PCB_W - HOLE_INSET, PCB_L - HOLE_INSET)]
SCREW_CLEAR_D = 3.3   # M3 clearance (bottom)
SCREW_TAP_D   = 2.6   # M3 self-tap into lid posts
HEAD_D, HEAD_H = 6.2, 2.6
CORNER_R = 3.0
PG = 0.5              # port clearance each side

STEP = 0.2            # staircase resolution for round features

# derived heights (assembly frame, z=0 = underside of case)
Z_PB  = FLOOR + PCB_BELOW          # PCB bottom
Z_PT  = Z_PB + PCB_T               # PCB top
Z_RIM = Z_PT + PCB_ABOVE           # top of bottom shell

# inner cavity
IX0, IX1 = -GAP - PIN13_CLEAR, PCB_W + GAP + BAY_OUT
IY0, IY1 = -GAP, PCB_L + GAP
OX0, OX1, OY0, OY1 = IX0 - WALL, IX1 + WALL, IY0 - WALL, IY1 + WALL

# ---------------- CSG primitives ----------------
class Box:
    def __init__(s, x0, x1, y0, y1, z0, z1): s.b = (x0, x1, y0, y1, z0, z1)
    def lines(s):
        x0, x1, y0, y1, z0, z1 = s.b; return [x0, x1], [y0, y1], [z0, z1]
    def inside(s, X, Y, Z):
        x0, x1, y0, y1, z0, z1 = s.b
        return (X > x0) & (X < x1) & (Y > y0) & (Y < y1) & (Z > z0) & (Z < z1)

def circ_lines(c, r):
    n = max(4, int(np.ceil(2 * r / STEP)))
    t = np.linspace(-1, 1, n + 1)
    return list(c + r * np.sin(np.linspace(-np.pi / 2, np.pi / 2, n + 1)))

class Cyl:
    def __init__(s, cx, cy, r, z0, z1): s.c = (cx, cy, r, z0, z1)
    def lines(s):
        cx, cy, r, z0, z1 = s.c
        return circ_lines(cx, r), circ_lines(cy, r), [z0, z1]
    def inside(s, X, Y, Z):
        cx, cy, r, z0, z1 = s.c
        return ((X - cx) ** 2 + (Y - cy) ** 2 < r * r) & (Z > z0) & (Z < z1)

class RBox:  # box with rounded vertical edges
    def __init__(s, x0, x1, y0, y1, z0, z1, r): s.b = (x0, x1, y0, y1, z0, z1, r)
    def lines(s):
        x0, x1, y0, y1, z0, z1, r = s.b
        xs = [x0, x1] + circ_lines(x0 + r, r) + circ_lines(x1 - r, r)
        ys = [y0, y1] + circ_lines(y0 + r, r) + circ_lines(y1 - r, r)
        return xs, ys, [z0, z1]
    def inside(s, X, Y, Z):
        x0, x1, y0, y1, z0, z1, r = s.b
        cx = np.clip(X, x0 + r, x1 - r); cy = np.clip(Y, y0 + r, y1 - r)
        return ((X - cx) ** 2 + (Y - cy) ** 2 < r * r) & (X > x0) & (X < x1) \
            & (Y > y0) & (Y < y1) & (Z > z0) & (Z < z1)

def solid(add, sub):
    """add/sub: lists of primitives. Returns (xs, ys, zs, occ)."""
    xs, ys, zs = set(), set(), set()
    for p in add + sub:
        a, b, c = p.lines(); xs.update(np.round(a, 4)); ys.update(np.round(b, 4)); zs.update(np.round(c, 4))
    xs, ys, zs = (np.array(sorted(v)) for v in (xs, ys, zs))
    cx, cy, cz = (0.5 * (v[1:] + v[:-1]) for v in (xs, ys, zs))
    X, Y, Z = np.meshgrid(cx, cy, cz, indexing='ij')
    occ = np.zeros(X.shape, bool)
    for p in add: occ |= p.inside(X, Y, Z)
    for p in sub: occ &= ~p.inside(X, Y, Z)
    return xs, ys, zs, occ

# ---------------- mesh output ----------------
def faces(xs, ys, zs, occ):
    """Yield quads (4 corner points, CCW seen from outside) for all boundary faces."""
    P = np.pad(occ, 1)
    quads = []
    for ax in range(3):
        a = np.moveaxis(P, ax, 0)
        d = a[1:].astype(np.int8) - a[:-1].astype(np.int8)   # +1: filled above -> face normal -ax
        d = d[:, 1:-1, 1:-1]
        coords = [xs, ys, zs]
        ca = coords[ax]; o = [coords[(ax + 1) % 3], coords[(ax + 2) % 3]]
        # moveaxis keeps remaining axes in original order
        rest = [i for i in range(3) if i != ax]
        cu, cv = coords[rest[0]], coords[rest[1]]
        for sgn in (1, -1):
            idx = np.argwhere(d == sgn)
            if not len(idx): continue
            k, i, j = idx.T
            pa = ca[k]
            u0, u1, v0, v1 = cu[i], cu[i + 1], cv[j], cv[j + 1]
            def pt(u, v):
                p = np.zeros((len(k), 3)); p[:, ax] = pa; p[:, rest[0]] = u; p[:, rest[1]] = v; return p
            q = np.stack([pt(u0, v0), pt(u1, v0), pt(u1, v1), pt(u0, v1)], 1)
            # orientation: (u,v,ax) right-handedness
            rh = (rest[0], rest[1], ax) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
            outward_pos = (sgn == -1)  # filled below the face -> normal +ax
            if outward_pos != rh: q = q[:, ::-1]
            quads.append(q)
    return np.concatenate(quads)

def write_stl(path, q):
    tris = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    rec = np.zeros(len(tris), dtype=[('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
    rec['n'] = n; rec['v'] = tris
    with open(path, 'wb') as f:
        f.write(b'Orange Pi Zero LTS case'.ljust(80, b' ')); f.write(struct.pack('<I', len(tris))); f.write(rec.tobytes())
    return len(tris)

# ---------------- parts ----------------
def bottom_parts():
    add = [RBox(OX0, OX1, OY0, OY1, 0, Z_RIM, CORNER_R)]
    sub = [Box(IX0, IX1, IY0, IY1, FLOOR, Z_RIM + 1)]
    posts = []
    for hx, hy in HOLES:
        posts += [Cyl(hx, hy, 3.0, FLOOR - 0.01, Z_PB), Cyl(hx, hy, 4.0, FLOOR - 0.01, FLOOR + 1.5)]
        sub += [Cyl(hx, hy, SCREW_CLEAR_D / 2, -1, Z_PB + 1), Cyl(hx, hy, HEAD_D / 2, -1, HEAD_H)]
    # y=0 wall: micro-USB, plug-overmold relief, microSD + finger scoop, antenna wire notch
    sub += [Box(6.3 - PG, 14.2 + PG, OY0 - 1, IY0 + 1, Z_PT - PG, Z_PT + 2.9 + PG),
            Box(6.3 - 3.5, 14.2 + 3.5, OY0 - 1, OY0 + WALL - 0.8, Z_PT - 3.5, Z_PT + 6.4),
            Box(22.4 - PG, 33.4 + PG, OY0 - 1, IY0 + 1, Z_PB - 1.9 - PG, Z_PB - 0.8 + PG),
            Box(19.5, 36.3, OY0 - 1, OY0 + 1.0, FLOOR + 0.6, Z_PB + 0.8),
            Box(38.5, 42.0, OY0 - 1, IY0 + 1, Z_RIM - 4.0, Z_RIM + 1)]
    # y=48.1 wall: USB-A (vertical), Ethernet
    sub += [Box(6.66 - PG, 12.48 + PG, IY1 - 1, OY1 + 1, Z_PT + 0.54 - PG, Z_PT + 0.54 + 13.12 + PG),
            Box(15.22 - PG, 31.38 + PG, IY1 - 1, OY1 + 1, Z_PT - PG, Z_PT + 13.14 + PG)]
    # bay end wall cable slot
    if BAY_SLOT:
        sub += [Box(IX1 - 1, OX1 + 1, BAY_SLOT_Y[0], BAY_SLOT_Y[1], Z_PT + BAY_SLOT_Z[0], Z_PT + BAY_SLOT_Z[1])]
    # vent slots in the 13-pin side wall
    for i in range(6):
        y = 10 + i * 5
        sub += [Box(OX0 - 1, IX0 + 1, y, y + 1.6, Z_PT + 2, Z_PT + 11)]
    return add, sub, posts

def bottom():
    add, sub, posts = bottom_parts()
    # posts must survive the cavity cut but not the screw holes
    screw = [p for p in sub if isinstance(p, Cyl)]
    others = [p for p in sub if not isinstance(p, Cyl)]
    allp = add + posts + sub
    xs_, ys_, zs_ = set(), set(), set()
    for p in allp:
        a, b, c = p.lines(); xs_.update(np.round(a, 4)); ys_.update(np.round(b, 4)); zs_.update(np.round(c, 4))
    xs, ys, zs = (np.array(sorted(v)) for v in (xs_, ys_, zs_))
    cx, cy, cz = (0.5 * (v[1:] + v[:-1]) for v in (xs, ys, zs))
    X, Y, Z = np.meshgrid(cx, cy, cz, indexing='ij')
    occ = add[0].inside(X, Y, Z)
    for p in others: occ &= ~p.inside(X, Y, Z)
    for p in posts: occ |= p.inside(X, Y, Z)
    for p in screw: occ &= ~p.inside(X, Y, Z)
    return xs, ys, zs, occ

def lid():
    """Print orientation: plate on the bed (z=0..LID_T), lip + posts point up."""
    L = 1.2; LIP_H = 1.2; LG = 0.25
    add = [RBox(OX0, OX1, OY0, OY1, 0, LID_T, CORNER_R),
           Box(IX0 + LG, IX1 - LG, IY0 + LG, IY1 - LG, LID_T - 0.01, LID_T + LIP_H)]
    sub = [Box(IX0 + LG + L, IX1 - LG - L, IY0 + LG + L, IY1 - LG - L, LID_T, LID_T + LIP_H + 1),
           # lip notches: antenna wire, and over the Ethernet / USB-A tops
           Box(38.5, 42.0, IY0 - 1, IY0 + 3, LID_T, LID_T + LIP_H + 1),
           Box(6.0, 32.0, IY1 - 3, IY1 + 1, LID_T, LID_T + LIP_H + 1)]
    # lid is mirrored in x when flipped onto the case: x_print = -x_assembly
    posts, holes = [], []
    for hx, hy in HOLES:
        posts.append(Cyl(-hx, hy, 2.75, LID_T - 0.01, LID_T + PCB_ABOVE))
        holes.append(Cyl(-hx, hy, SCREW_TAP_D / 2, LID_T + 1.0, LID_T + PCB_ABOVE + 1))
    # mirror the box primitives too
    def mx(b):
        x0, x1 = b.b[0], b.b[1]
        if isinstance(b, RBox): return RBox(-x1, -x0, *b.b[2:])
        return Box(-x1, -x0, *b.b[2:])
    add = [mx(p) for p in add]; sub = [mx(p) for p in sub]
    vents = []
    for i in range(7):            # over the SoC / board centre
        x = 11 + i * 3.6
        vents.append(Box(-(x + 1.8), -x, 12, 36, -1, LID_T + 1))
    for i in range(3):            # over the connector bay
        x = PCB_W + 3 + i * 4
        vents.append(Box(-(x + 1.8), -x, 10, 38, -1, LID_T + 1))
    allp = add + sub + posts + holes + vents
    xs_, ys_, zs_ = set(), set(), set()
    for p in allp:
        a, b, c = p.lines(); xs_.update(np.round(a, 4)); ys_.update(np.round(b, 4)); zs_.update(np.round(c, 4))
    xs, ys, zs = (np.array(sorted(v)) for v in (xs_, ys_, zs_))
    cx, cy, cz = (0.5 * (v[1:] + v[:-1]) for v in (xs, ys, zs))
    X, Y, Z = np.meshgrid(cx, cy, cz, indexing='ij')
    occ = np.zeros(X.shape, bool)
    for p in add: occ |= p.inside(X, Y, Z)
    for p in sub: occ &= ~p.inside(X, Y, Z)
    for p in vents: occ &= ~p.inside(X, Y, Z)
    for p in posts: occ |= p.inside(X, Y, Z)
    for p in holes: occ &= ~p.inside(X, Y, Z)
    return xs, ys, zs, occ

def board_dummy():
    """Rough board + connectors (assembly frame) for fit previews."""
    return [Box(0, PCB_W, 0, PCB_L, Z_PB, Z_PT),
            Box(15.22, 31.38, 27.5, PCB_L + 4, Z_PT, Z_PT + 13.14),       # Ethernet
            Box(6.66, 12.48, 36, PCB_L + 1, Z_PT + 0.54, Z_PT + 13.66),  # USB-A
            Box(6.3, 14.2, -1, 5, Z_PT, Z_PT + 2.9),                     # micro-USB
            Box(22.4, 33.4, -0.5, 14, Z_PB - 1.9, Z_PB - 0.8),           # microSD
            Box(40, PCB_W + 13, 7, 41, Z_PT - 1, Z_PT + 7),              # 26-pin + side connector (guess)
            Box(-7, 2.5, 7.5, 40.5, Z_PT + 1, Z_PT + 3.5)]               # 13-pin sideways pins (guess)

if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else '.'
    os.makedirs(out, exist_ok=True)
    for name, fn in [('opiz_case_bottom', bottom), ('opiz_case_lid', lid)]:
        xs, ys, zs, occ = fn()
        n = write_stl(os.path.join(out, name + '.stl'), faces(xs, ys, zs, occ))
        print(f'{name}: grid {occ.shape}, {n} triangles, '
              f'{xs[0]:.1f}..{xs[-1]:.1f} x {ys[0]:.1f}..{ys[-1]:.1f} x {zs[0]:.1f}..{zs[-1]:.1f}')
    print(f'outer size: {OX1-OX0:.1f} x {OY1-OY0:.1f} x {Z_RIM+LID_T:.1f} mm')
