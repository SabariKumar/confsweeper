"""
Ray-trace the three MCMM proposer before/after structures in PyMOL, using
Sabari's BallnStick preset and pymolrc render settings, for inclusion in
figures/20260619_mcmm_steps_overview.svg.

Run inside the isolated PyMOL conda env (not the pixi env):
    mamba run -n pymol_render python scripts/mcmm_pymol_render.py \
        --in_dir results/mcmm_pymol --out_dir results/mcmm_pymol

For each proposer it overlays the start conformer (grey ghost, transparent)
against the moved conformer (BallnStick, carbons gray30) with the moved atoms
recoloured, on one shared camera, and writes a ray-traced transparent PNG.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pymol

pymol.finish_launching(["pymol", "-qc"])
from pymol import cmd, preset  # noqa: E402


def _apply_global_settings() -> None:
    """
    Apply the pymolrc global render settings (white bg, ray-trace outline, AA).

    Params: None
    Returns:
        None
    """
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", "off")
    cmd.set("orthoscopic", 0)
    cmd.set("transparency", 0.5)
    cmd.set("dash_gap", 0)
    cmd.set("ray_trace_mode", 1)
    cmd.set("ray_texture", 2)
    cmd.set("antialias", 3)
    cmd.set("ambient", 0.5)
    cmd.set("spec_count", 5)
    cmd.set("shininess", 50)
    cmd.set("specular", 1)
    cmd.set("reflect", 0.1)
    cmd.space("cmyk")


def _ball_n_stick(selection: str) -> None:
    """
    Apply the BallnStick preset to a selection (Sabari's pymolrc command).

    Params:
        selection: str : PyMOL object/selection name
    Returns:
        None
    """
    cmd.set("dash_radius", 0.035)
    cmd.set("surface_quality", 2)
    cmd.set("surface_type", 4)
    cmd.set("depth_cue", "off")
    preset.ball_and_stick(selection=selection, mode=1)
    cmd.color("gray30", f"({selection}) and elem C")


_HILITE = {"dbt": "marine", "dih": "teal", "cart": "orange"}


def _render_one(name: str, in_dir: Path, out_dir: Path, view: list | None) -> list:
    """
    Render one proposer overlay and return the camera view used.

    Params:
        name: str : proposer key ('dbt' | 'dih' | 'cart')
        in_dir: Path : directory holding start.pdb and <name>.pdb + selection.json
        out_dir: Path : output directory for the PNG
        view: list | None : shared camera view; if None it is computed from start
    Returns:
        list : the 18-element PyMOL view matrix used (for reuse across proposers)
    """
    sel = json.loads((in_dir / "selection.json").read_text())
    moved_key = {"dbt": "dbt_moved", "dih": "dih_moved", "cart": None}[name]

    cmd.reinitialize()
    _apply_global_settings()
    cmd.load(str(in_dir / "start.pdb"), "ghost")
    cmd.load(str(in_dir / f"{name}.pdb"), "after")
    cmd.remove("hydro")

    _ball_n_stick("ghost")
    _ball_n_stick("after")

    # ghost = faded grey, translucent
    cmd.color("grey80", "ghost")
    cmd.set("stick_transparency", 0.65, "ghost")
    cmd.set("sphere_transparency", 0.65, "ghost")

    # highlight the moved region on the "after" object
    if moved_key is not None:
        idx = sel[moved_key]
        if idx:
            sel_str = "after and index " + "+".join(str(i + 1) for i in idx)
            cmd.color(_HILITE[name], sel_str)
    else:  # cartesian: whole molecule moved
        cmd.color(_HILITE["cart"], "after and elem C")

    if view is None:
        cmd.orient("ghost")
        cmd.zoom("ghost", buffer=3.5, complete=1)
        view = cmd.get_view()
    cmd.set_view(view)

    cmd.set("ray_shadows", 1)
    out_png = out_dir / f"{name}.png"
    cmd.ray(1400, 1200)
    cmd.png(str(out_png), dpi=300, ray=0)
    print(f"wrote {out_png}")
    return view


def main(in_dir: Path, out_dir: Path) -> None:
    """
    Render all three proposer overlays on one shared camera.

    Params:
        in_dir: Path : directory with start/dbt/dih/cart PDBs + selection.json
        out_dir: Path : PNG output directory
    Returns:
        None
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    view = _render_one("dbt", in_dir, out_dir, None)  # establishes shared view
    _render_one("dih", in_dir, out_dir, view)
    _render_one("cart", in_dir, out_dir, view)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_dir", type=Path, default=Path("results/mcmm_pymol"))
    ap.add_argument("--out_dir", type=Path, default=Path("results/mcmm_pymol"))
    args = ap.parse_args()
    main(args.in_dir, args.out_dir)
