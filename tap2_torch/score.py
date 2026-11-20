"""Differentiable TAP2: total CDR length, PSH, PPC, PNC and SFvCSP.

The published score (see ``tap_score.py`` in TAP2recreated) is a sum over pairs
of CDR-vicinity, surface-exposed residues weighted by the inverse square of
their closest heavy-atom separation.  Four of its steps are step functions with
no useful gradient, and each is replaced here by a temperature-controlled smooth
form that converges to the original as the temperatures go to zero:

=================================  ==================================================
Published rule                     Smooth replacement
=================================  ==================================================
relative side-chain SASA >= 7.5%   logistic gate on relative SASA (``tau_surface``)
closest heavy-atom distance        soft-min over atom pairs (``tau_min``)
pair cutoff at 7.5 A               cosine taper from ``taper_start`` to 7.5 A
4 A CDR-vicinity test              logistic gate on distance (``tau_vicinity``)
salt bridge within 3.2 A           logistic gate blending toward Gly (``tau_salt``)
=================================  ==================================================

One further change is deliberate rather than cosmetic.  The published weight
``2 / d^2`` diverges as residues approach, and its gradient ``-4 / d^3`` diverges
faster, so a single mild clash dominates the gradient of the whole score.
``core_softening`` replaces ``d^2`` with ``d^2 + core_softening^2``, which is
bounded and has vanishing gradient at contact.  TAP2recreated measured the
effect of a 1.2 A floor on agreement with published scores (PSH Pearson r rose
from 0.195 to 0.689 on clash-bearing FlashABB models), and 1.2 A is the default
here for the same reason.

``hard=True`` restores every published rule exactly, including
``core_softening = 0``, and is used to validate against the reference
implementation.  It is not differentiable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .soft_sasa import contact_area, fibonacci_sphere, soft_sasa
from .constants import (
    CA_SLOT,
    GLY_HYDROPATHY,
    PROBE_RADIUS,
    SALT_BRIDGE_CUTOFF,
    SURFACE_THRESHOLD,
)

METRIC_KEYS = ("cdr_length", "psh", "ppc", "pnc", "sfvcsp")


@dataclass
class TAP2Spec:
    """Per-residue constants for one Fv, independent of coordinates.

    Everything here is fixed by the sequence and the numbering, so a spec is
    built once and reused for every sampling step.  Build one with
    :func:`tap2_torch.spec.build_spec`.

    Attributes:
        hydropathy: TAP rescaled hydrophobicity. Shape: ``[n]``.
        charge: Formal residue charge. Shape: ``[n]``.
        reference_area: psa Ala-X-Ala side-chain contact area. Shape: ``[n]``.
        is_anchor: Residue lies in a CDR or its two flanking anchors. Shape: ``[n]``.
        is_cdr: Residue lies in a CDR. Shape: ``[n]``.
        is_heavy_chain: Residue belongs to the heavy chain. Shape: ``[n]``.
        atom_valid: Occupied atom37 slots. Shape: ``[n, 37]``.
        is_side_chain: Slots psa counts as side chain (all but N, C, O). Shape: ``[n, 37]``.
        atom_radii: NACCESS radii per slot, zero where unoccupied. Shape: ``[n, 37]``.
        donor_slots: Salt-bridge donor atoms (Lys NZ, Arg NH1/NH2). Shape: ``[n, 37]``.
        acceptor_slots: Salt-bridge acceptor atoms (Asp OD1/2, Glu OE1/2). Shape: ``[n, 37]``.
    """

    hydropathy: Tensor
    charge: Tensor
    reference_area: Tensor
    is_anchor: Tensor
    is_cdr: Tensor
    is_heavy_chain: Tensor
    atom_valid: Tensor
    is_side_chain: Tensor
    atom_radii: Tensor
    donor_slots: Tensor
    acceptor_slots: Tensor

    @property
    def n_residues(self) -> int:
        return int(self.hydropathy.shape[0])

    def to(self, *, device=None, dtype=None) -> "TAP2Spec":
        """Move to a device, casting only the floating-point fields."""

        def move(value: Tensor) -> Tensor:
            if value.dtype.is_floating_point:
                return value.to(device=device, dtype=dtype)
            return value.to(device=device)

        return TAP2Spec(**{name: move(getattr(self, name)) for name in self.__dataclass_fields__})


def _cosine_taper(distance: Tensor, start: float, end: float) -> Tensor:
    """Smoothly switch from 1 below ``start`` to 0 above ``end``."""
    span = max(end - start, 1e-6)
    ramp = ((distance - start) / span).clamp(0.0, 1.0)
    return 0.5 * (1.0 + torch.cos(math.pi * ramp))


def _noisy_or(terms: Tensor, dim: int) -> Tensor:
    """Probabilistic OR, ``1 - prod(1 - p)``; exact Boolean OR on 0/1 input."""
    return 1.0 - torch.prod(1.0 - terms, dim=dim)


def _pair_min_distance(
    coords: Tensor,
    atom_valid: Tensor,
    *,
    cutoff: float,
    ca_cutoff: float,
    tau_min: float,
    hard: bool,
    pair_chunk: int,
) -> tuple[Tensor, Tensor]:
    """Closest heavy-atom distance between every nearby residue pair.

    Residue pairs whose CA atoms are far apart cannot have close heavy atoms, so
    only pairs that could fall within ``cutoff`` are evaluated; the rest are
    reported as non-candidates and contribute nothing.

    Returns:
        ``(distance, candidate)``, both ``[n, n]``, symmetric with a zero
        diagonal in ``candidate``.
    """
    n_residues = coords.shape[0]
    ca = coords[:, CA_SLOT, :]
    with torch.no_grad():
        detached = coords.detach()
        ca_distance = torch.cdist(ca.detach(), ca.detach())
        # Exact bound: a residue's atoms lie within `reach` of its own CA, so a
        # closest-atom distance below `cutoff` forces
        # d(CA_i, CA_j) < reach_i + reach_j + cutoff. Pairs outside that bound
        # contribute identically zero, so dropping them is exact, and the bound
        # is much tighter than a fixed CA cutoff.
        offsets = (detached - ca.detach()[:, None, :]).norm(dim=-1)
        reach = offsets.masked_fill(~atom_valid, 0.0).amax(dim=-1)  # [n]
        bound = (reach[:, None] + reach[None, :] + cutoff).clamp(max=ca_cutoff)
        candidate = ca_distance < bound
        candidate.fill_diagonal_(False)
        rows, cols = torch.triu(candidate, diagonal=1).nonzero(as_tuple=True)

    distance = coords.new_full((n_residues, n_residues), float("inf"))
    if rows.numel() == 0:
        return distance, candidate

    values = []
    for start in range(0, rows.numel(), pair_chunk):
        chunk_rows = rows[start : start + pair_chunk]
        chunk_cols = cols[start : start + pair_chunk]
        left = coords[chunk_rows]  # [p, 37, 3]
        right = coords[chunk_cols]  # [p, 37, 3]
        delta = left[:, :, None, :] - right[:, None, :, :]  # [p, 37, 37, 3]
        pair_distance = delta.square().sum(-1).clamp_min(1e-12).sqrt()  # [p, 37, 37]
        valid = atom_valid[chunk_rows][:, :, None] & atom_valid[chunk_cols][:, None, :]
        masked = pair_distance.masked_fill(~valid, float("inf"))
        flat = masked.flatten(1)
        if hard:
            values.append(flat.min(dim=-1).values)
        else:
            # Soft-min: -tau * logsumexp(-d / tau), shifted for stability.
            shift = flat.min(dim=-1, keepdim=True).values.detach()
            values.append(shift.squeeze(-1) - tau_min * torch.logsumexp(-(flat - shift) / tau_min, dim=-1))
    pair_values = torch.cat(values)

    flat_index = rows * n_residues + cols
    distance = distance.flatten().scatter(0, flat_index, pair_values).view(n_residues, n_residues)
    distance = torch.minimum(distance, distance.transpose(0, 1))
    return distance, candidate


def _salt_bridge_neutralization(
    coords: Tensor,
    spec: TAP2Spec,
    surface: Tensor,
    *,
    tau_salt: float,
    hard: bool,
) -> Tensor:
    """Per-residue weight for substituting glycine hydropathy.

    TAP neutralises the hydrophobicity of *exposed* salt-bridge partners while
    leaving their charges intact.  Donor and acceptor atom sets are small (at
    most two atoms on a handful of residues), so this is evaluated densely.

    Returns:
        Neutralisation weight in ``[0, 1]``. Shape: ``[n]``.
    """
    donor_residue, donor_slot = spec.donor_slots.nonzero(as_tuple=True)
    acceptor_residue, acceptor_slot = spec.acceptor_slots.nonzero(as_tuple=True)
    if donor_residue.numel() == 0 or acceptor_residue.numel() == 0:
        return torch.zeros_like(surface)

    donor_xyz = coords[donor_residue, donor_slot]  # [d, 3]
    acceptor_xyz = coords[acceptor_residue, acceptor_slot]  # [a, 3]
    distance = torch.cdist(donor_xyz, acceptor_xyz)  # [d, a]

    if hard:
        close = (distance <= SALT_BRIDGE_CUTOFF).to(coords.dtype)
    else:
        close = torch.sigmoid((SALT_BRIDGE_CUTOFF - distance) / tau_salt)
    # Only pairs where both partners are exposed are neutralised.
    gate = close * surface[donor_residue][:, None] * surface[acceptor_residue][None, :]

    neutral = torch.zeros_like(surface)
    donor_weight = _noisy_or(gate, dim=1)  # [d]
    acceptor_weight = _noisy_or(gate, dim=0)  # [a]
    # A residue may contribute several donor or acceptor atoms; OR them together.
    neutral = neutral.index_put((donor_residue,), donor_weight, accumulate=True)
    neutral = neutral.index_put((acceptor_residue,), acceptor_weight, accumulate=True)
    return neutral.clamp(max=1.0)


def tap2_scores(
    coords: Tensor,
    spec: TAP2Spec,
    *,
    atom_valid: Tensor | None = None,
    relative_exposure: Tensor | None = None,
    hard: bool = False,
    tau_surface: float = 1.0,
    tau_min: float = 0.2,
    tau_vicinity: float = 0.25,
    tau_salt: float = 0.2,
    tau_sasa: float = 0.3,
    core_softening: float = 1.2,
    taper_start: float = 6.5,
    pair_cutoff: float = 7.5,
    vicinity_radius: float = 4.0,
    ca_cutoff: float = 26.0,
    n_sasa_points: int = 100,
    max_neighbors: int | None = None,
    pair_chunk: int = 4096,
    sphere_points: Tensor | None = None,
) -> dict[str, Tensor]:
    """Score one Fv from atom37 coordinates.

    Args:
        coords: Atom37 coordinates in Angstrom. Shape: ``[n, 37, 3]``.
        spec: Per-residue constants for this Fv.
        atom_valid: Occupied slots, intersected with ``spec.atom_valid``. Use it
            to drop atoms the decoder did not emit. Shape: ``[n, 37]``.
        relative_exposure: Relative side-chain accessibility in percent,
            replacing the internal soft Shrake-Rupley calculation. Pass the
            reference ``psa``/FreeSASA values to validate the rest of the score
            against a reference implementation, or a cheaper burial proxy of
            your own. Shape: ``[n]``.
        hard: Reproduce the published rules exactly. Overrides ``core_softening``
            to zero and disables every smooth gate. Not differentiable.
        tau_surface: Logistic width (percent RASA) of the exposure gate.
        tau_min: Soft-min temperature (A) for closest-atom distances.
        tau_vicinity: Logistic width (A) of the 4 A CDR-vicinity test.
        tau_salt: Logistic width (A) of the salt-bridge test.
        tau_sasa: Logistic width (A) of the Shrake-Rupley burial test.
        core_softening: Distance (A) added in quadrature to the pair weight
            denominator, bounding the clash singularity. Zero recovers ``2/d^2``.
        taper_start: Distance (A) at which the pair weight starts tapering.
        pair_cutoff: Distance (A) at which the pair weight reaches zero.
        vicinity_radius: CDR-vicinity radius (A). TAP2recreated found 4.0
            reproduces the published workbook, though the paper describes 4.5.
        ca_cutoff: Hard ceiling (A) on the CA-CA prefilter radius. The radius
            actually used is the exact per-pair bound and is usually far smaller.
        n_sasa_points: Shrake-Rupley test points per atom.
        max_neighbors: Optional cap on neighbours per atom in the SASA
            calculation. ``None`` sizes the list exactly from the data.
        pair_chunk: Residue pairs per chunk, bounding peak memory.
        sphere_points: Precomputed unit sphere points. Shape: ``[n_sasa_points, 3]``.

    Returns:
        Mapping of :data:`METRIC_KEYS` to scalar tensors. ``cdr_length`` is an
        integer count and carries no gradient; the other four are differentiable
        with respect to ``coords`` unless ``hard`` is set.
    """
    if coords.ndim != 3 or coords.shape[1:] != (37, 3):
        raise ValueError(f"coords must be [n, 37, 3], got {tuple(coords.shape)}")
    if coords.shape[0] != spec.n_residues:
        raise ValueError(f"coords has {coords.shape[0]} residues, spec has {spec.n_residues}")
    if hard:
        core_softening = 0.0

    valid = spec.atom_valid if atom_valid is None else (spec.atom_valid & atom_valid)

    # --- exposure -------------------------------------------------------
    if relative_exposure is None:
        flat_residue, flat_slot = valid.nonzero(as_tuple=True)
        flat_coords = coords[flat_residue, flat_slot]  # [a, 3]
        flat_radii = spec.atom_radii[flat_residue, flat_slot]  # [a]
        if sphere_points is None:
            sphere_points = fibonacci_sphere(n_sasa_points, device=coords.device, dtype=coords.dtype)
        atom_sasa = soft_sasa(
            flat_coords,
            flat_radii,
            probe_radius=PROBE_RADIUS,
            n_points=n_sasa_points,
            tau=tau_sasa,
            max_neighbors=max_neighbors,
            hard=hard,
            sphere_points=sphere_points,
        )
        atom_contact = contact_area(atom_sasa, flat_radii)

        # Side-chain contact area per residue, as psa defines side chain.
        side_chain_atom = spec.is_side_chain[flat_residue, flat_slot]
        residue_area = torch.zeros_like(spec.hydropathy).index_put(
            (flat_residue,), atom_contact * side_chain_atom.to(coords.dtype), accumulate=True
        )
        relative_area = 100.0 * residue_area / spec.reference_area.clamp_min(1e-6)
    else:
        relative_area = relative_exposure.to(device=coords.device, dtype=coords.dtype)
    if hard:
        surface = (relative_area >= SURFACE_THRESHOLD).to(coords.dtype)
    else:
        surface = torch.sigmoid((relative_area - SURFACE_THRESHOLD) / tau_surface)

    # --- closest-atom distances and CDR vicinity ------------------------
    distance, candidate = _pair_min_distance(
        coords,
        valid,
        cutoff=pair_cutoff,
        ca_cutoff=ca_cutoff,
        tau_min=tau_min,
        hard=hard,
        pair_chunk=pair_chunk,
    )
    candidate_f = candidate.to(coords.dtype)
    finite_distance = torch.where(candidate, distance, torch.full_like(distance, 1e4))

    anchor = spec.is_anchor.to(coords.dtype)
    seeds = anchor * surface
    if hard:
        close_to_seed = (finite_distance < vicinity_radius).to(coords.dtype)
    else:
        close_to_seed = torch.sigmoid((vicinity_radius - finite_distance) / tau_vicinity)
    near = _noisy_or(close_to_seed * seeds[None, :] * candidate_f, dim=1)
    vicinity = surface * (anchor + (1.0 - anchor) * near)

    # --- pair weights ---------------------------------------------------
    if hard:
        taper = (finite_distance <= pair_cutoff).to(coords.dtype)
    else:
        taper = _cosine_taper(finite_distance, taper_start, pair_cutoff)
    weight = candidate_f * taper / (finite_distance.square() + core_softening**2)
    weight = weight - torch.diag_embed(weight.diagonal())  # zero the diagonal

    # --- hydropathy with exposed salt bridges neutralised ---------------
    neutral = _salt_bridge_neutralization(coords, spec, surface, tau_salt=tau_salt, hard=hard)
    hydropathy = spec.hydropathy + neutral * (GLY_HYDROPATHY - spec.hydropathy)

    # Each metric is a quadratic form in the shared weight matrix. Because the
    # matrix is symmetric with a zero diagonal, v^T W v sums every ordered pair
    # (i, j) and (j, i), which is the factor of 2 in TAP's published sum.
    def quadratic(values: Tensor) -> Tensor:
        gated = vicinity * values
        return gated @ weight @ gated

    positive = spec.charge.clamp_min(0.0)
    negative = (-spec.charge).clamp_min(0.0)

    heavy = spec.is_heavy_chain.to(coords.dtype)
    heavy_charge = (surface * spec.charge * heavy).sum()
    light_charge = (surface * spec.charge * (1.0 - heavy)).sum()

    return {
        "cdr_length": spec.is_cdr.sum().to(coords.dtype).detach(),
        "psh": quadratic(hydropathy),
        "ppc": quadratic(positive),
        "pnc": quadratic(negative),
        "sfvcsp": heavy_charge * light_charge,
    }
