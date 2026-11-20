"""Finite-difference gradient check, soft-to-hard convergence, and timing."""
import sys, time
import numpy as np
import torch

sys.path.insert(0, ".")

from validation.pdb_to_atom37 import load
from tap2_torch.score import tap2_scores

torch.manual_seed(0)
path = sys.argv[1]
coords, spec, valid, residues = load(path, dtype=torch.float64)
print(f"{spec.n_residues} residues, {int((spec.atom_valid & valid).sum())} heavy atoms\n")

kw = dict(spec=spec, atom_valid=valid, n_sasa_points=60, max_neighbors=96)

# --- analytic vs central finite differences -------------------------------
for key in ("psh", "ppc", "pnc", "sfvcsp"):
    x = coords.clone().requires_grad_(True)
    out = tap2_scores(x, **kw)[key]
    (grad,) = torch.autograd.grad(out, x)

    flat_valid = (spec.atom_valid & valid).flatten().nonzero().flatten()
    picks = flat_valid[torch.randperm(len(flat_valid))[:12]]
    errors, mags = [], []
    eps = 1e-5
    for pick in picks:
        res, slot = int(pick) // 37, int(pick) % 37
        for axis in range(3):
            plus, minus = coords.clone(), coords.clone()
            plus[res, slot, axis] += eps
            minus[res, slot, axis] -= eps
            numeric = (float(tap2_scores(plus, **kw)[key]) - float(tap2_scores(minus, **kw)[key])) / (2 * eps)
            analytic = float(grad[res, slot, axis])
            errors.append(abs(numeric - analytic))
            mags.append(abs(numeric))
    scale = max(max(mags), 1e-12)
    print(f"{key:<7} value={float(out):>12.5f}  grad norm={float(grad.norm()):>10.4f}  "
          f"max FD error={max(errors):.2e}  relative={max(errors)/scale:.2e}  "
          f"{'OK' if max(errors)/scale < 1e-5 else 'CHECK'}")

# --- soft converges toward hard as the temperatures shrink ----------------
hard = tap2_scores(coords, spec=spec, atom_valid=valid, hard=True, n_sasa_points=60, max_neighbors=96)
print("\nscale   psh          ppc        pnc        sfvcsp")
print(f"{'hard':<7} {float(hard['psh']):<12.4f} {float(hard['ppc']):<10.4f} "
      f"{float(hard['pnc']):<10.4f} {float(hard['sfvcsp']):<10.4f}")
for scale in (1.0, 0.5, 0.2, 0.1):
    soft = tap2_scores(
        coords, spec=spec, atom_valid=valid, n_sasa_points=60, max_neighbors=96,
        tau_surface=1.0 * scale, tau_min=0.2 * scale, tau_vicinity=0.25 * scale,
        tau_salt=0.2 * scale, tau_sasa=0.3 * scale, core_softening=0.0,
        taper_start=7.5 - 1.0 * scale,
    )
    print(f"{scale:<7.2f} {float(soft['psh']):<12.4f} {float(soft['ppc']):<10.4f} "
          f"{float(soft['pnc']):<10.4f} {float(soft['sfvcsp']):<10.4f}")

# --- timing, float32, as it would run in the sampler ----------------------
c32 = coords.to(torch.float32).requires_grad_(True)
s32 = spec.to(dtype=torch.float32)
for points in (60, 100):
    for _ in range(2):
        out = tap2_scores(c32, spec=s32, atom_valid=valid, n_sasa_points=points, max_neighbors=96)["psh"]
        (g,) = torch.autograd.grad(out, c32)
    start = time.perf_counter()
    reps = 5
    for _ in range(reps):
        out = tap2_scores(c32, spec=s32, atom_valid=valid, n_sasa_points=points, max_neighbors=96)["psh"]
        (g,) = torch.autograd.grad(out, c32)
    print(f"\n{points} SASA points: {1000*(time.perf_counter()-start)/reps:.1f} ms per "
          f"forward+backward (CPU, {spec.n_residues} residues)")
