"""
Generate the three proposer "cartoon" molecule groups for the MCMM steps
schematic (figures/20260619_mcmm_steps_overview.svg).

Each cartoon draws the SAME starting macrocycle (a 6-atom backbone ring with
one aromatic side chain) as a faint ghost, plus the result of one sample move
from that proposer drawn solid, with an arrow showing the motion:

  - DBT concerted backbone rotation : a backbone arc swings while the ring
    stays closed (side chain unchanged).
  - Side-chain dihedral kick        : the aromatic side chain rotates about
    its Cα–Cβ bond to a new rotamer (backbone unchanged).
  - Cartesian kick                  : every atom is displaced by a small
    isotropic perturbation (the whole molecule jiggles).

The script emits a standalone preview SVG (all three cartoons on a blank
canvas) for visual checking, and a fragment file holding just the three
positioned <g> groups for pasting into the main schematic.

Usage:
    pixi run python scripts/mcmm_cartoons.py \\
        --preview_svg  /tmp/mcmm_cartoons_preview.svg \\
        --fragment_out /tmp/mcmm_cartoons_fragment.svg
"""

from __future__ import annotations

import math
from pathlib import Path

import click

# ---- base molecule in local coordinates (origin = ring centre) ----
_R = 5.0
_BACKBONE = {
    "B0": (0.0, -_R),
    "B1": (4.33, -2.5),
    "B2": (4.33, 2.5),
    "B3": (0.0, _R),
    "B4": (-4.33, 2.5),
    "B5": (-4.33, -2.5),
}
_RING_BONDS = [
    ("B0", "B1"),
    ("B1", "B2"),
    ("B2", "B3"),
    ("B3", "B4"),
    ("B4", "B5"),
    ("B5", "B0"),
]
_SA = (6.93, -4.0)  # Cβ (side-chain pivot atom), attached to B1
_AR_C = (9.18, -5.3)  # aromatic ring centre
_AR_R = 1.7
# aromatic hexagon vertices; angle 150° vertex faces the Cβ atom
_AR_ANGLES = [150, 210, 270, 330, 30, 90]


def _rotate(p: tuple, pivot: tuple, deg: float) -> tuple:
    """
    Rotate point p about pivot by deg degrees.

    Params:
        p: tuple : (x, y) point to rotate
        pivot: tuple : (x, y) centre of rotation
        deg: float : rotation angle in degrees (SVG sense, y-down)
    Returns:
        tuple : rotated (x, y)
    """
    a = math.radians(deg)
    dx, dy = p[0] - pivot[0], p[1] - pivot[1]
    return (
        pivot[0] + dx * math.cos(a) - dy * math.sin(a),
        pivot[1] + dx * math.sin(a) + dy * math.cos(a),
    )


def _aromatic_verts(center: tuple) -> list:
    """
    Build the six aromatic-ring vertices around a centre.

    Params:
        center: tuple : (x, y) aromatic ring centre
    Returns:
        list : six (x, y) vertices, vertex 0 facing the Cβ atom
    """
    return [
        (
            center[0] + _AR_R * math.cos(math.radians(a)),
            center[1] + _AR_R * math.sin(math.radians(a)),
        )
        for a in _AR_ANGLES
    ]


def _atom(x: float, y: float, fill: str, r: float = 0.82, op: float = 1.0) -> str:
    """
    Emit one atom circle.

    Params:
        x: float : x coordinate
        y: float : y coordinate
        fill: str : fill / self-stroke colour
        r: float : radius (mm)
        op: float : opacity
    Returns:
        str : SVG <circle> element
    """
    return (
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{fill}" stroke="{fill}" '
        f'stroke-width="0.45" opacity="{op}"/>'
    )


def _bond(
    p: tuple, q: tuple, color: str = "#333333", w: float = 0.5, op: float = 1.0
) -> str:
    """
    Emit one bond path between two points.

    Params:
        p: tuple : (x, y) start
        q: tuple : (x, y) end
        color: str : stroke colour
        w: float : stroke width
        op: float : opacity
    Returns:
        str : SVG <path> element
    """
    return (
        f'<path d="M {p[0]:.2f},{p[1]:.2f} L {q[0]:.2f},{q[1]:.2f}" stroke="{color}" '
        f'stroke-width="{w}" fill="none" opacity="{op}"/>'
    )


def _draw_molecule(
    atoms: dict, ar_verts: list, color: str, op: float, atom_fill: str
) -> list:
    """
    Emit all bonds and atoms for one molecule state (backbone + side chain + aromatic).

    Params:
        atoms: dict : backbone atom name -> (x, y), plus key 'SA'
        ar_verts: list : six aromatic vertices (x, y)
        color: str : bond / aromatic colour
        op: float : opacity for the whole state
        atom_fill: str : backbone atom fill colour
    Returns:
        list : list of SVG element strings
    """
    out = []
    for n1, n2 in _RING_BONDS:
        out.append(_bond(atoms[n1], atoms[n2], color, 0.55, op))
    out.append(_bond(atoms["B1"], atoms["SA"], color, 0.55, op))  # Cα–Cβ
    out.append(_bond(atoms["SA"], ar_verts[0], color, 0.55, op))  # Cβ–aromatic
    for i in range(6):  # aromatic ring
        out.append(_bond(ar_verts[i], ar_verts[(i + 1) % 6], color, 0.55, op))
    # inner aromatic circle to signal aromaticity
    out.append(
        f'<circle cx="{_centroid(ar_verts)[0]:.2f}" cy="{_centroid(ar_verts)[1]:.2f}" '
        f'r="0.9" fill="none" stroke="{color}" stroke-width="0.35" opacity="{op}"/>'
    )
    for n in ["B0", "B1", "B2", "B3", "B4", "B5"]:
        out.append(_atom(*atoms[n], atom_fill, 0.95, op))
    out.append(_atom(*atoms["SA"], atom_fill, 0.95, op))
    return out


def _centroid(pts: list) -> tuple:
    """
    Compute the centroid of a list of points.

    Params:
        pts: list : list of (x, y)
    Returns:
        tuple : (x, y) centroid
    """
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _base_state() -> tuple:
    """
    Build the unmodified starting molecule.

    Params: None
    Returns:
        tuple : (atoms dict incl. 'SA', aromatic vertices list)
    """
    atoms = dict(_BACKBONE)
    atoms["SA"] = _SA
    return atoms, _aromatic_verts(_AR_C)


_GHOST = "#b9b9b9"  # ghost (start) colour
_GHOST_OP = 0.45


def _cartoon_dbt() -> list:
    """
    Build the DBT concerted-backbone-rotation cartoon (local coords).

    Params: None
    Returns:
        list : SVG element strings
    """
    atoms, ar = _base_state()
    out = _draw_molecule(atoms, ar, _GHOST, _GHOST_OP, _GHOST)  # ghost start
    moved_set = {"B2", "B3", "B4"}
    moved = dict(atoms)
    for n in moved_set:  # swing lower arc, ring stays closed
        moved[n] = _rotate(atoms[n], (0.0, 0.0), 25)
    # ring bonds: blue where they touch a moved atom, dark (unchanged) otherwise
    solid = []
    for n1, n2 in _RING_BONDS:
        col = "#7d97c4" if (n1 in moved_set or n2 in moved_set) else "#333333"
        solid.append(_bond(moved[n1], moved[n2], col, 0.6))
    # side chain unchanged → solid dark
    solid += [
        _bond(atoms["B1"], atoms["SA"], "#333333", 0.55),
        _bond(atoms["SA"], ar[0], "#333333", 0.55),
    ]
    for i in range(6):
        solid.append(_bond(ar[i], ar[(i + 1) % 6], "#333333", 0.55))
    cc = _centroid(ar)
    solid.append(
        f'<circle cx="{cc[0]:.2f}" cy="{cc[1]:.2f}" r="0.9" fill="none" stroke="#333333" stroke-width="0.35"/>'
    )
    for n in ["B0", "B1", "B2", "B3", "B4", "B5"]:
        solid.append(_atom(*moved[n], "#3d5a8a" if n in moved_set else "#333333"))
    solid.append(_atom(*atoms["SA"], "#333333"))
    out += solid
    # motion arrow tracking the largest-moving atom (B3)
    a, b = atoms["B3"], moved["B3"]
    ext = (b[0] + (b[0] - a[0]) * 0.5, b[1] + (b[1] - a[1]) * 0.5)
    out.append(
        f'<path d="M {a[0]:.2f},{a[1]:.2f} Q {(a[0]+ext[0])/2-1.8:.2f},{(a[1]+ext[1])/2+1.2:.2f} {ext[0]:.2f},{ext[1]:.2f}" '
        f'stroke="#c0392b" stroke-width="0.6" fill="none" marker-end="url(#cArrow)"/>'
    )
    return out


def _cartoon_dihedral() -> list:
    """
    Build the side-chain dihedral-kick cartoon (local coords).

    Params: None
    Returns:
        list : SVG element strings
    """
    atoms, ar = _base_state()
    out = _draw_molecule(atoms, ar, _GHOST, _GHOST_OP, _GHOST)  # ghost start
    # backbone + Cα–Cβ drawn solid (unchanged)
    solid = [_bond(atoms[n1], atoms[n2], "#333333", 0.55) for n1, n2 in _RING_BONDS]
    solid.append(_bond(atoms["B1"], atoms["SA"], "#333333", 0.55))
    for n in ["B0", "B1", "B2", "B3", "B4", "B5"]:
        solid.append(_atom(*atoms[n], "#333333"))
    solid.append(_atom(*atoms["SA"], "#333333"))
    # rotate the aromatic group about Cβ to a new rotamer
    ar_moved = [_rotate(v, atoms["SA"], 78) for v in ar]
    solid.append(_bond(atoms["SA"], ar_moved[0], "#2f8f7f", 0.6))
    for i in range(6):
        solid.append(_bond(ar_moved[i], ar_moved[(i + 1) % 6], "#2f8f7f", 0.6))
    c = _centroid(ar_moved)
    solid.append(
        f'<circle cx="{c[0]:.2f}" cy="{c[1]:.2f}" r="0.9" fill="none" stroke="#2f8f7f" stroke-width="0.35"/>'
    )
    out += solid
    # curved motion arrow from old to new aromatic centroid, bowed around Cβ
    o, n = _centroid(ar), c
    out.append(
        f'<path d="M {o[0]:.2f},{o[1]:.2f} Q {atoms["SA"][0]+3.0:.2f},{atoms["SA"][1]:.2f} {n[0]:.2f},{n[1]:.2f}" '
        f'stroke="#c0392b" stroke-width="0.5" fill="none" marker-end="url(#cArrow)"/>'
    )
    return out


def _cartoon_cartesian() -> list:
    """
    Build the Cartesian-kick cartoon (local coords).

    Params: None
    Returns:
        list : SVG element strings
    """
    atoms, ar = _base_state()
    out = _draw_molecule(atoms, ar, _GHOST, _GHOST_OP, _GHOST)  # ghost start
    # deterministic small per-atom offsets (isotropic jiggle)
    offs = {
        "B0": (0.9, -0.7),
        "B1": (1.2, 0.5),
        "B2": (-0.7, 1.1),
        "B3": (0.5, 1.2),
        "B4": (-1.1, -0.5),
        "B5": (-0.8, 0.8),
        "SA": (1.1, -0.8),
    }
    moved = {n: (atoms[n][0] + offs[n][0], atoms[n][1] + offs[n][1]) for n in atoms}
    ar_off = (0.9, -0.9)
    ar_moved = [(v[0] + ar_off[0], v[1] + ar_off[1]) for v in ar]
    solid = [_bond(moved[n1], moved[n2], "#a05a52", 0.6) for n1, n2 in _RING_BONDS]
    solid.append(_bond(moved["B1"], moved["SA"], "#a05a52", 0.6))
    solid.append(_bond(moved["SA"], ar_moved[0], "#a05a52", 0.6))
    for i in range(6):
        solid.append(_bond(ar_moved[i], ar_moved[(i + 1) % 6], "#a05a52", 0.6))
    c = _centroid(ar_moved)
    solid.append(
        f'<circle cx="{c[0]:.2f}" cy="{c[1]:.2f}" r="0.9" fill="none" stroke="#a05a52" stroke-width="0.35"/>'
    )
    for n in ["B0", "B1", "B2", "B3", "B4", "B5"]:
        solid.append(_atom(*moved[n], "#9c4a3f"))
    solid.append(_atom(*moved["SA"], "#9c4a3f"))
    out += solid
    # a few small displacement arrows
    for n in ["B0", "B3", "B5"]:
        a, b = atoms[n], moved[n]
        ext = (b[0] + (b[0] - a[0]) * 1.6, b[1] + (b[1] - a[1]) * 1.6)
        out.append(
            f'<path d="M {a[0]:.2f},{a[1]:.2f} L {ext[0]:.2f},{ext[1]:.2f}" '
            f'stroke="#c0392b" stroke-width="0.45" fill="none" marker-end="url(#cArrow)"/>'
        )
    return out


def _group(elems: list, cx: float, cy: float) -> str:
    """
    Wrap cartoon elements in a translate group.

    Params:
        elems: list : SVG element strings (local coords)
        cx: float : translate x
        cy: float : translate y
    Returns:
        str : SVG <g> block
    """
    body = "\n    ".join(elems)
    return f'  <g transform="translate({cx},{cy})">\n    {body}\n  </g>'


_CARTOON_DEFS = (
    '<marker id="cArrow" refX="0" refY="0" orient="auto-start-reverse" '
    'markerWidth="1" markerHeight="1" viewBox="0 0 1 1" preserveAspectRatio="xMidYMid">'
    '<path transform="scale(0.4)" style="fill:context-stroke;stroke:context-stroke;stroke-width:1pt" '
    'd="M 5.77,0 -2.88,5 V -5 Z"/></marker>'
)

# row centres in the main schematic (match the three move-box centres)
_ROWS = {"dbt": (177.0, 34.5), "dih": (177.0, 50.5), "cart": (177.0, 66.5)}


def _build_groups() -> str:
    """
    Build the three positioned cartoon groups for the main schematic.

    Params: None
    Returns:
        str : concatenated SVG <g> blocks
    """
    return "\n".join(
        [
            _group(_cartoon_dbt(), *_ROWS["dbt"]),
            _group(_cartoon_dihedral(), *_ROWS["dih"]),
            _group(_cartoon_cartesian(), *_ROWS["cart"]),
        ]
    )


@click.command()
@click.option(
    "--preview_svg",
    type=click.Path(path_type=Path),
    default=Path("/tmp/mcmm_cartoons_preview.svg"),
)
@click.option(
    "--fragment_out",
    type=click.Path(path_type=Path),
    default=Path("/tmp/mcmm_cartoons_fragment.svg"),
)
def main(preview_svg: Path, fragment_out: Path) -> None:
    """
    Write a standalone preview SVG and a paste-ready fragment of the cartoons.

    Params:
        preview_svg: Path : standalone preview output (cartoons on blank canvas)
        fragment_out: Path : fragment file with the three positioned <g> blocks
    Returns:
        None
    """
    groups = _build_groups()
    fragment_out.write_text(groups + "\n")

    preview = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="240mm" height="90mm" '
        f'viewBox="140 22 80 56" version="1.1">\n'
        f"  <defs>{_CARTOON_DEFS}</defs>\n"
        f'  <rect x="140" y="22" width="80" height="56" fill="#ffffff"/>\n'
        f"{groups}\n</svg>\n"
    )
    preview_svg.write_text(preview)
    click.echo(f"wrote {preview_svg} and {fragment_out}")


if __name__ == "__main__":
    main()
