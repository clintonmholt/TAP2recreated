"""Soft Shrake-Rupley: exactness in hard mode, and how it compares with psa/FreeSASA."""
import sys
import numpy as np
import torch

sys.path.insert(0, ".")

from validation.pdb_to_atom37 import load
from tap2_torch.soft_sasa import contact_area, fibonacci_sphere, soft_sasa
from tap2_torch.constants import PROBE_RADIUS
from tap_surface import side_chain_accessibility

path = sys.argv[1]
coords, spec, valid, residues = load(path)
flat_res, flat_slot = (spec.atom_valid & valid).nonzero(as_tuple=True)
xyz = coords[flat_res, flat_slot].numpy()
radii = spec.atom_radii[flat_res, flat_slot].numpy()
points = fibonacci_sphere(100, dtype=torch.float64)

# --- independent brute-force Shrake-Rupley, same point set ----------------
def brute(xyz, radii, probe=PROBE_RADIUS, pts=points.numpy()):
    expanded = radii + probe
    out = np.zeros(len(xyz))
    for i in range(len(xyz)):
        test = xyz[i] + expanded[i] * pts                      # [P, 3]
        free = np.ones(len(pts), dtype=bool)
        for j in range(len(xyz)):
            if j == i:
                continue
            free &= np.linalg.norm(test - xyz[j], axis=1) >= expanded[j]
        out[i] = 4 * np.pi * expanded[i] ** 2 * free.mean()
    return out

reference_sr = brute(xyz, radii)
hard_sr = soft_sasa(coords[flat_res, flat_slot], spec.atom_radii[flat_res, flat_slot],
                    n_points=100, hard=True, sphere_points=points, max_neighbors=96).numpy()
delta = np.abs(hard_sr - reference_sr)
print(f"atoms: {len(xyz)}")
print(f"hard soft_sasa vs brute-force SR : max|diff| = {delta.max():.3e} A^2  "
      f"{'MATCH' if delta.max() < 1e-9 else 'MISMATCH'}")

# --- soft converges to hard as tau -> 0 -----------------------------------
print("\ntau (A)   max|soft-hard| (A^2)   mean|soft-hard|")
for tau in (1.0, 0.5, 0.3, 0.1, 0.05, 0.02):
    soft = soft_sasa(coords[flat_res, flat_slot], spec.atom_radii[flat_res, flat_slot],
                     n_points=100, tau=tau, sphere_points=points, max_neighbors=96).numpy()
    d = np.abs(soft - hard_sr)
    print(f"{tau:<9.2f} {d.max():<22.4f} {d.mean():.4f}")

# --- relative side-chain exposure: SR vs the reference FreeSASA/LR path ----
sc = spec.is_side_chain[flat_res, flat_slot].numpy()
contact = contact_area(torch.as_tensor(hard_sr), torch.as_tensor(radii)).numpy()
area = np.zeros(spec.n_residues)
np.add.at(area, flat_res.numpy(), contact * sc)
rel_sr = 100 * area / spec.reference_area.numpy()

relative, _ = side_chain_accessibility(path)
rel_ref = np.array([relative.get(r.parent.id, {}).get(f"{r.id[1]}{r.id[2].strip()}", 0.0)
                    for r in residues])
agree = ((rel_sr >= 7.5) == (rel_ref >= 7.5)).mean()
print(f"\nrelative side-chain exposure, Shrake-Rupley vs FreeSASA Lee-Richards:")
print(f"  Pearson r              = {np.corrcoef(rel_sr, rel_ref)[0, 1]:.4f}")
print(f"  mean |difference|      = {np.abs(rel_sr - rel_ref).mean():.2f} percent")
print(f"  7.5% surface-call agreement = {100 * agree:.1f}%  "
      f"({int((((rel_sr >= 7.5) != (rel_ref >= 7.5))).sum())} of {len(rel_sr)} residues differ)")
