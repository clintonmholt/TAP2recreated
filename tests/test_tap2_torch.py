"""Tests for the differentiable TAP2 score.

Self-contained: the Fv used here is generated geometry, not a real antibody, so
the suite runs without downloading structures.  ``test_matches_reference``
additionally compares against TAP2recreated's published-score implementation and
skips when that package or FreeSASA is unavailable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tap2_torch.constants import ATOM37_SLOT, PROBE_RADIUS, RESIDUE_ATOMS
from tap2_torch.score import tap2_scores
from tap2_torch.soft_sasa import fibonacci_sphere, soft_sasa
from tap2_torch.spec import build_spec

RESIDUE_CYCLE = list(RESIDUE_ATOMS)


def make_fv(n_per_chain: int = 40, dtype: torch.dtype = torch.float64):
    """Build a two-chain Fv on a loose helix, numbered 1..n in IMGT."""
    names, coords, valid, heavy, numbers = [], [], [], [], []
    for chain, (origin, offset) in enumerate([((0.0, 0.0, 0.0), 0), ((7.0, 5.0, 1.0), 3)]):
        for k in range(n_per_chain):
            resname = RESIDUE_CYCLE[(k * 7 + offset) % len(RESIDUE_CYCLE)]
            names.append(resname)
            heavy.append(chain == 0)
            numbers.append(k + 1)
            angle = 0.55 * k
            base = np.array([origin[0] + 9 * math.cos(angle),
                             origin[1] + 9 * math.sin(angle),
                             origin[2] + 1.9 * k])
            slots = np.zeros((37, 3))
            occupied = np.zeros(37, dtype=bool)
            for j, atom in enumerate(RESIDUE_ATOMS[resname]):
                radius = 1.3 + 0.95 * j
                slots[ATOM37_SLOT[atom]] = base + [radius * math.cos(angle + 0.42 * j),
                                                   radius * math.sin(angle + 0.42 * j),
                                                   0.33 * j * math.cos(0.7 * j)]
                occupied[ATOM37_SLOT[atom]] = True
            coords.append(slots)
            valid.append(occupied)
    spec = build_spec(names, is_heavy_chain=np.array(heavy),
                      imgt_numbers=np.array(numbers), dtype=dtype)
    return (torch.as_tensor(np.stack(coords), dtype=dtype),
            spec,
            torch.as_tensor(np.stack(valid)))


@pytest.fixture(scope="module")
def fv():
    return make_fv()


def test_hard_sasa_is_exact_shrake_rupley():
    """In hard mode the neighbour-list sparsification must change nothing."""
    coords, spec, valid = make_fv(n_per_chain=12)
    residue, slot = (spec.atom_valid & valid).nonzero(as_tuple=True)
    xyz = coords[residue, slot]
    radii = spec.atom_radii[residue, slot]
    points = fibonacci_sphere(40, dtype=torch.float64)

    expanded = (radii + PROBE_RADIUS).numpy()
    reference = np.zeros(len(xyz))
    for i in range(len(xyz)):
        test = xyz[i].numpy() + expanded[i] * points.numpy()
        free = np.ones(len(points), dtype=bool)
        for j in range(len(xyz)):
            if i != j:
                free &= np.linalg.norm(test - xyz[j].numpy(), axis=1) >= expanded[j]
        reference[i] = 4 * np.pi * expanded[i] ** 2 * free.mean()

    got = soft_sasa(xyz, radii, n_points=40, hard=True, sphere_points=points).numpy()
    np.testing.assert_allclose(got, reference, atol=1e-10)


def test_invariant_to_rigid_motion(fv):
    """Every score is a function of interatomic distances alone."""
    coords, spec, valid = fv
    torch.manual_seed(0)
    axis = torch.nn.functional.normalize(torch.randn(3, dtype=coords.dtype), dim=0)
    angle = 0.7
    cross = torch.tensor([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]], dtype=coords.dtype)
    rotation = (torch.eye(3, dtype=coords.dtype) + math.sin(angle) * cross
                + (1 - math.cos(angle)) * cross @ cross)
    moved = coords @ rotation.T + torch.tensor([11.0, -4.0, 2.5], dtype=coords.dtype)

    before = tap2_scores(coords, spec, atom_valid=valid, n_sasa_points=40)
    after = tap2_scores(moved, spec, atom_valid=valid, n_sasa_points=40)
    for key in before:
        assert float(before[key]) == pytest.approx(float(after[key]), rel=1e-8, abs=1e-8)


@pytest.mark.parametrize("key", ["psh", "ppc", "pnc"])
def test_gradient_matches_finite_differences(fv, key):
    """Analytic gradients must agree with central differences."""
    coords, spec, valid = fv
    kwargs = dict(spec=spec, atom_valid=valid, n_sasa_points=40)
    x = coords.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(tap2_scores(x, **kwargs)[key], x)

    torch.manual_seed(1)
    flat = (spec.atom_valid & valid).flatten().nonzero().flatten()
    picks = flat[torch.randperm(len(flat))[:6]]
    eps = 1e-5
    for pick in picks:
        residue, slot = int(pick) // 37, int(pick) % 37
        for axis in range(3):
            plus, minus = coords.clone(), coords.clone()
            plus[residue, slot, axis] += eps
            minus[residue, slot, axis] -= eps
            numeric = (float(tap2_scores(plus, **kwargs)[key])
                       - float(tap2_scores(minus, **kwargs)[key])) / (2 * eps)
            assert numeric == pytest.approx(float(grad[residue, slot, axis]),
                                            rel=1e-4, abs=1e-7)


def test_soft_converges_toward_hard(fv):
    """Shrinking every temperature must close the gap to the published rules."""
    coords, spec, valid = fv
    hard = float(tap2_scores(coords, spec, atom_valid=valid, hard=True, n_sasa_points=40)["psh"])
    gaps = []
    for scale in (1.0, 0.5, 0.2, 0.1):
        soft = tap2_scores(
            coords, spec, atom_valid=valid, n_sasa_points=40,
            tau_surface=1.0 * scale, tau_min=0.2 * scale, tau_vicinity=0.25 * scale,
            tau_salt=0.2 * scale, tau_sasa=0.3 * scale, core_softening=0.0,
            taper_start=7.5 - 1.0 * scale,
        )
        gaps.append(abs(float(soft["psh"]) - hard))
    assert gaps == sorted(gaps, reverse=True), gaps
    assert gaps[-1] < 0.01 * abs(hard)


def test_core_softening_bounds_clash_gradient(fv):
    """A clash must not be allowed to dominate the gradient."""
    coords, spec, valid = fv
    clashed = coords.clone()
    # Drag one residue's side chain onto a neighbour's.
    clashed[5, ATOM37_SLOT["CA"]] = coords[7, ATOM37_SLOT["CA"]] + 0.05

    norms = {}
    for softening in (0.0, 1.2):
        x = clashed.clone().requires_grad_(True)
        out = tap2_scores(x, spec, atom_valid=valid, n_sasa_points=40,
                          core_softening=softening)["psh"]
        (grad,) = torch.autograd.grad(out, x)
        norms[softening] = float(grad.norm())
    assert norms[1.2] < norms[0.0]
    assert math.isfinite(norms[1.2])


def test_rejects_unsupported_residue():
    with pytest.raises(ValueError, match="unsupported residues"):
        build_spec(["ALA", "MSE"], is_heavy_chain=[True, True], imgt_numbers=[1, 2])


def test_matches_reference():
    """Hard mode must reproduce TAP2recreated exactly, exposure held fixed.

    The two differ only in the surface calculation (FreeSASA Lee-Richards with
    psa rounding, versus Shrake-Rupley here), so the reference exposure is
    injected to compare the rest of the score.
    """
    tap_score = pytest.importorskip("tap_score")
    tap_surface = pytest.importorskip("tap_surface")
    pdb = pytest.importorskip("pathlib").Path(__file__).parent / "data" / "synthetic_fv.pdb"
    if not pdb.exists():
        pytest.skip("synthetic_fv.pdb not present")
    from Bio.PDB import PDBParser

    reference = tap_score.calculate(tap_score.prepare(str(pdb)))
    model = next(PDBParser(QUIET=True).get_structure("fv", str(pdb)).get_models())
    residues = [r for chain in model for r in chain if r.id[0] == " "]
    n = len(residues)
    coords = np.zeros((n, 37, 3))
    valid = np.zeros((n, 37), dtype=bool)
    oxt = np.zeros(n, dtype=bool)
    for i, residue in enumerate(residues):
        for atom in residue:
            slot = ATOM37_SLOT.get(atom.id)
            if slot is None or atom.element in {"H", "D"}:
                continue
            coords[i, slot], valid[i, slot] = atom.coord, True
            oxt[i] |= atom.id == "OXT"
    spec = build_spec([r.resname for r in residues],
                      is_heavy_chain=np.array([r.parent.id == "H" for r in residues]),
                      imgt_numbers=np.array([r.id[1] for r in residues]),
                      include_oxt=oxt, dtype=torch.float64)
    relative, _ = tap_surface.side_chain_accessibility(str(pdb))
    exposure = torch.tensor([relative.get(r.parent.id, {}).get(f"{r.id[1]}{r.id[2].strip()}", 0.0)
                             for r in residues], dtype=torch.float64)
    got = tap2_scores(torch.as_tensor(coords), spec, atom_valid=torch.as_tensor(valid),
                      relative_exposure=exposure, hard=True)
    for key, want in zip(("cdr_length", "psh", "ppc", "pnc", "sfvcsp"), reference):
        assert float(got[key]) == pytest.approx(want, rel=1e-9, abs=1e-9)
