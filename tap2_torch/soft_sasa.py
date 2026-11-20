"""Differentiable Shrake-Rupley solvent accessibility.

Shrake-Rupley places a fixed set of test points on each atom's expanded sphere
of radius ``r_i + probe`` and counts the fraction not buried inside any
neighbour's expanded sphere.  The burial test is an indicator function, so the
standard algorithm has zero gradient almost everywhere and jumps as points cross
a neighbour's surface.  Here each indicator is replaced by a logistic in the
signed distance to that neighbour's sphere, giving a score that is smooth in the
coordinates while converging to exact Shrake-Rupley as ``tau -> 0``.

Gradients reach the coordinates through two paths: the test points translate
with their own atom, and the signed distances depend on every neighbour's
position.

The neighbour list is an *exact* sparsification, not an approximation: a test
point of atom ``i`` can only be occluded by ``j`` when
``d_ij < (r_i + probe) + (r_j + probe)``, so pairs outside that radius
contribute identically zero and dropping them loses no gradient.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .constants import PROBE_RADIUS


def fibonacci_sphere(n_points: int, *, device=None, dtype=torch.float32) -> Tensor:
    """Return ``n_points`` near-uniform unit vectors via the golden spiral.

    Deterministic, so the score is reproducible across steps and processes.

    Returns:
        Unit vectors. Shape: ``[n_points, 3]``.
    """
    k = torch.arange(n_points, device=device, dtype=torch.float64)
    # z spans (-1, 1) with half-cell offsets; phi advances by the golden angle.
    z = 1.0 - (2.0 * k + 1.0) / n_points
    radius = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    phi = k * (math.pi * (3.0 - math.sqrt(5.0)))
    points = torch.stack([radius * torch.cos(phi), radius * torch.sin(phi), z], dim=-1)
    return points.to(dtype)


def soft_sasa(
    coords: Tensor,
    radii: Tensor,
    *,
    probe_radius: float = PROBE_RADIUS,
    n_points: int = 100,
    tau: float = 0.3,
    max_neighbors: int | None = None,
    hard: bool = False,
    sphere_points: Tensor | None = None,
) -> Tensor:
    """Compute per-atom solvent-accessible surface area, differentiably.

    Args:
        coords: Heavy-atom coordinates in Angstrom. Shape: ``[a, 3]``.
        radii: Van der Waals radii in Angstrom. Shape: ``[a]``.
        probe_radius: Solvent probe radius in Angstrom.
        n_points: Test points per atom. 100 matches RFD4-Proteina's
            ``calculate_atomwise_rasa`` default and TNP's calibration.
        tau: Logistic width in Angstrom for the burial test. Smaller is closer
            to exact Shrake-Rupley but gives sharper gradients.
        max_neighbors: Upper bound on neighbours retained per atom. ``None``
            sizes the neighbour list from the data on every call, which is both
            exact and minimal; set it only to cap peak memory, and note that a
            cap below the true occluder count makes the result approximate.
        hard: Use the exact indicator instead of the logistic. Not
            differentiable; for validation against a reference implementation.
        sphere_points: Precomputed unit vectors. Shape: ``[n_points, 3]``.

    Returns:
        Per-atom SASA in square Angstrom. Shape: ``[a]``.

    Raises:
        ValueError: If ``coords`` is not ``[a, 3]``.
    """
    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError(f"coords must be [a, 3], got {tuple(coords.shape)}")
    n_atoms = coords.shape[0]
    if n_atoms == 0:
        return coords.new_zeros((0,))

    expanded = radii + probe_radius  # [a]
    if sphere_points is None:
        sphere_points = fibonacci_sphere(n_points, device=coords.device, dtype=coords.dtype)
    sphere_points = sphere_points.to(device=coords.device, dtype=coords.dtype)

    # Pairwise atom distances; detached only for *choosing* neighbours.
    with torch.no_grad():
        pair_distance = torch.cdist(coords.detach(), coords.detach())  # [a, a]
        interaction = expanded[:, None] + expanded[None, :]  # [a, a]
        candidate = pair_distance < interaction
        candidate.fill_diagonal_(False)
        # Size the neighbour list from the data: every atom inside the
        # interaction radius is kept, and nothing else is ever needed.
        n_candidates = int(candidate.sum(dim=1).max().item()) if n_atoms > 1 else 0
        keep = n_candidates if max_neighbors is None else min(n_candidates, max_neighbors)
        keep = min(keep, n_atoms - 1) if n_atoms > 1 else 0
        if keep > 0:
            # Nearest `keep` atoms, with non-candidates pushed to the end.
            ranked = torch.where(candidate, pair_distance, torch.full_like(pair_distance, float("inf")))
            ranked.fill_diagonal_(float("inf"))
            neighbor_index = ranked.topk(keep, dim=1, largest=False).indices  # [a, keep]
            neighbor_valid = torch.gather(candidate, 1, neighbor_index)  # [a, keep]

    area_per_atom = 4.0 * math.pi * expanded**2  # [a]
    if keep == 0:
        return area_per_atom

    # Test points, carried by their own atom so gradients reach coords[i].
    test_points = coords[:, None, :] + expanded[:, None, None] * sphere_points[None, :, :]  # [a, p, 3]

    neighbor_coords = coords[neighbor_index]  # [a, keep, 3]
    neighbor_expanded = expanded[neighbor_index]  # [a, keep]

    # Signed burial depth of each test point inside each neighbour's sphere.
    delta = test_points[:, :, None, :] - neighbor_coords[:, None, :, :]  # [a, p, keep, 3]
    distance = delta.square().sum(-1).clamp_min(1e-12).sqrt()  # [a, p, keep]
    depth = neighbor_expanded[:, None, :] - distance  # > 0 when buried

    if hard:
        occupancy = (depth > 0).to(coords.dtype)
    else:
        occupancy = torch.sigmoid(depth / tau)
    occupancy = occupancy * neighbor_valid[:, None, :].to(coords.dtype)

    # Fraction of each point that stays free of every neighbour.
    free = torch.prod(1.0 - occupancy, dim=-1)  # [a, p]
    return area_per_atom * free.mean(dim=-1)


def contact_area(sasa: Tensor, radii: Tensor, *, probe_radius: float = PROBE_RADIUS) -> Tensor:
    """Convert solvent-accessible area to psa-style contact area.

    ``psa`` reports the area of each atom's own van der Waals sphere that the
    probe centre can touch, which is the accessible area scaled by
    ``(r / (r + probe))**2``.  TNP's ``surface.py`` applies the same factor.

    Args:
        sasa: Per-atom solvent-accessible area. Shape: ``[a]``.
        radii: Per-atom van der Waals radii. Shape: ``[a]``.

    Returns:
        Per-atom contact area. Shape: ``[a]``.
    """
    return sasa * (radii / (radii + probe_radius)) ** 2
