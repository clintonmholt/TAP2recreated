"""Build a :class:`~tap2_torch.score.TAP2Spec` from sequence and numbering.

A spec holds only what is fixed by the sequence and the antibody numbering, so
it is built once per design problem and reused for every sampling step.  Three
entry points are provided, in decreasing order of how much they infer:

* :func:`build_spec` -- explicit residue names and masks.
* :func:`spec_from_atom_array` -- an AtomWorks ``AtomArrayPlus`` or biotite
  ``AtomArray``, numbered with ANARCII if it carries no IMGT annotation.
* :func:`spec_from_residue_types` -- RFD4-Proteina's decoded ``residue_type``
  token indices, reusing CDR masks resolved once from the input array.

TAP2 is defined on IMGT numbering.  RFD4-Proteina's own
``annotate_antibody_numbering`` uses Chothia, so these helpers request the IMGT
scheme from ANARCII rather than reusing that annotation.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from torch import Tensor

from .constants import (
    ATOM37_SLOT,
    BACKBONE_ATOMS,
    CHARGE,
    REFERENCE_SIDE_CHAIN_AREAS,
    RESIDUE_ATOMS,
    SALT_BRIDGE_ACCEPTORS,
    SALT_BRIDGE_DONORS,
    naccess_radius,
    rescaled_hydropathy,
    verify_atom37_order,
)
from .score import TAP2Spec

CDR_RANGES_IMGT: tuple[tuple[int, int], ...] = ((27, 38), (56, 65), (105, 117))
"""TAP's IMGT CDR definitions (``CDRS`` in TAP2recreated's ``tap_score.py``)."""

ANCHOR_RANGES_IMGT: tuple[tuple[int, int], ...] = tuple(
    (start - 2, end + 2) for start, end in CDR_RANGES_IMGT
)
"""CDRs plus their two flanking anchor residues on each side."""


def in_ranges(numbers: np.ndarray, ranges: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Return a mask of positions falling in any inclusive range.

    Insertion codes are ignored, which is correct for IMGT: residue 111A of
    CDR3 carries number 111 and so falls inside ``(105, 117)``.
    """
    numbers = np.asarray(numbers)
    return np.logical_or.reduce([(numbers >= low) & (numbers <= high) for low, high in ranges])


def build_spec(
    residue_names: list[str],
    *,
    is_heavy_chain: np.ndarray | list[bool],
    imgt_numbers: np.ndarray | list[int] | None = None,
    is_cdr: np.ndarray | list[bool] | None = None,
    is_anchor: np.ndarray | list[bool] | None = None,
    include_oxt: np.ndarray | list[bool] | None = None,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> TAP2Spec:
    """Assemble a spec from three-letter residue names and masks.

    Args:
        residue_names: Three-letter names, one per residue, in the same order as
            the coordinate tensor's residue axis.
        is_heavy_chain: True for heavy-chain residues. Shape: ``[n]``.
        imgt_numbers: IMGT positions, used to derive the CDR and anchor masks.
            Shape: ``[n]``. Ignored when both masks are given explicitly.
        is_cdr: Explicit CDR mask, overriding ``imgt_numbers``. Shape: ``[n]``.
        is_anchor: Explicit CDR-plus-anchor mask. Shape: ``[n]``.
        include_oxt: Residues carrying a C-terminal OXT atom. Shape: ``[n]``.
        device: Torch device for the resulting tensors.
        dtype: Floating-point dtype for the resulting tensors.

    Returns:
        A spec ready to pass to :func:`~tap2_torch.score.tap2_scores`.

    Raises:
        ValueError: On an unsupported residue name, a length mismatch, or when
            neither numbering nor explicit masks are supplied.
    """
    verify_atom37_order()
    n_residues = len(residue_names)
    unsupported = sorted(set(residue_names) - REFERENCE_SIDE_CHAIN_AREAS.keys())
    if unsupported:
        raise ValueError(f"unsupported residues {unsupported}")
    heavy = np.asarray(is_heavy_chain, dtype=bool)
    if heavy.shape != (n_residues,):
        raise ValueError(f"is_heavy_chain must have length {n_residues}, got {heavy.shape}")

    if is_cdr is None or is_anchor is None:
        if imgt_numbers is None:
            raise ValueError("supply imgt_numbers, or both is_cdr and is_anchor")
        numbers = np.asarray(imgt_numbers, dtype=int)
        if numbers.shape != (n_residues,):
            raise ValueError(f"imgt_numbers must have length {n_residues}, got {numbers.shape}")
        is_cdr = in_ranges(numbers, CDR_RANGES_IMGT) if is_cdr is None else is_cdr
        is_anchor = in_ranges(numbers, ANCHOR_RANGES_IMGT) if is_anchor is None else is_anchor
    cdr = np.asarray(is_cdr, dtype=bool)
    anchor = np.asarray(is_anchor, dtype=bool)
    oxt = np.zeros(n_residues, dtype=bool) if include_oxt is None else np.asarray(include_oxt, dtype=bool)

    atom_valid = np.zeros((n_residues, 37), dtype=bool)
    is_side_chain = np.zeros((n_residues, 37), dtype=bool)
    atom_radii = np.zeros((n_residues, 37), dtype=np.float64)
    donor_slots = np.zeros((n_residues, 37), dtype=bool)
    acceptor_slots = np.zeros((n_residues, 37), dtype=bool)

    for index, resname in enumerate(residue_names):
        atoms = list(RESIDUE_ATOMS[resname])
        if oxt[index]:
            atoms.append("OXT")
        for atom_name in atoms:
            slot = ATOM37_SLOT[atom_name]
            atom_valid[index, slot] = True
            atom_radii[index, slot] = naccess_radius(resname, atom_name)
            if atom_name not in BACKBONE_ATOMS:
                is_side_chain[index, slot] = True
        for atom_name in SALT_BRIDGE_DONORS.get(resname, ()):
            donor_slots[index, ATOM37_SLOT[atom_name]] = True
        for atom_name in SALT_BRIDGE_ACCEPTORS.get(resname, ()):
            acceptor_slots[index, ATOM37_SLOT[atom_name]] = True

    def as_float(values: np.ndarray) -> Tensor:
        return torch.as_tensor(values, dtype=dtype, device=device)

    def as_bool(values: np.ndarray) -> Tensor:
        return torch.as_tensor(values, dtype=torch.bool, device=device)

    return TAP2Spec(
        hydropathy=as_float(np.array([rescaled_hydropathy(name) for name in residue_names])),
        charge=as_float(np.array([CHARGE.get(name, 0.0) for name in residue_names])),
        reference_area=as_float(np.array([REFERENCE_SIDE_CHAIN_AREAS[name] for name in residue_names])),
        is_anchor=as_bool(anchor),
        is_cdr=as_bool(cdr),
        is_heavy_chain=as_bool(heavy),
        atom_valid=as_bool(atom_valid),
        is_side_chain=as_bool(is_side_chain),
        atom_radii=as_float(atom_radii),
        donor_slots=as_bool(donor_slots),
        acceptor_slots=as_bool(acceptor_slots),
    )


# --------------------------------------------------------------------------
# IMGT numbering
# --------------------------------------------------------------------------


@lru_cache(maxsize=256)
def _number_imgt_cached(sequence: str) -> tuple[tuple[int, str, str], ...]:
    """Number one chain with ANARCII in the IMGT scheme.

    Returns:
        One ``(imgt_number, insertion_code, one_letter)`` triple per aligned
        residue, gaps dropped. Empty when ANARCII recognises no domain.
    """
    from anarcii import Anarcii

    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        numberer = Anarcii(cpu=True, ncpu=1, verbose=False)
        numberer.number({"query": sequence})
        domains = list(numberer.to_scheme("imgt").values())
    finally:
        torch.set_num_threads(threads)

    records: list[tuple[int, str, str]] = []
    for domain in domains:
        for position, residue in domain.get("numbering") or ():
            if residue == "-":
                continue
            number, insertion = position if isinstance(position, tuple) else (position, " ")
            records.append((int(number), str(insertion).strip(), residue))
    return tuple(records)


def imgt_numbers_for_sequence(sequence: str) -> np.ndarray:
    """Return per-residue IMGT numbers for a chain, or -1 where unassigned.

    Args:
        sequence: One-letter amino-acid sequence of a single chain.

    Returns:
        IMGT numbers aligned to ``sequence``. Shape: ``[len(sequence)]``.

    Raises:
        ValueError: If ANARCII recognises no variable domain in the sequence.
    """
    records = _number_imgt_cached(sequence)
    if not records:
        raise ValueError("ANARCII found no variable domain in this chain")
    numbers = np.full(len(sequence), -1, dtype=int)
    cursor = 0
    for number, _insertion, residue in records:
        found = sequence.find(residue, cursor)
        if found < 0:
            continue
        numbers[found] = number
        cursor = found + 1
    return numbers
