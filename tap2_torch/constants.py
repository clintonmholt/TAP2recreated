"""Chemistry constants for the differentiable TAP2 score.

Three sources are merged here, deliberately kept separate so each can be
checked against its origin:

* ``KYTE_DOOLITTLE``, ``CHARGE``, ``SALT_BRIDGE_*`` -- the TAP1/TAP2 papers, as
  implemented in ``tap_score.py`` of TAP2recreated.
* ``NACCESS_*`` radii and ``REFERENCE_SIDE_CHAIN_AREAS`` -- Oxford TNP's
  ``surface.py`` (BSD-3-Clause, University of Oxford), vendored in
  TAP2recreated as ``tap_surface.py``.  These reproduce the ``psa`` program on
  which the TAP thresholds were calibrated.
* ``AF2_ATOM_TYPES`` -- RFD4-Proteina's atom37 slot order
  (``rfproteina.utils.constants.af2_atom_types``, identical to the ``aa`` block
  of ``resources/aa_rna_dna_glycan_ncaa.yaml``).  ``verify_atom37_order``
  cross-checks it against the live package when RFD4 is importable.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# TAP scoring chemistry
# --------------------------------------------------------------------------

KYTE_DOOLITTLE: dict[str, float] = {
    "ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5,
    "GLN": -3.5, "GLU": -3.5, "GLY": -0.4, "HIS": -3.2, "ILE": 4.5,
    "LEU": 3.8, "LYS": -3.9, "MET": 1.9, "PHE": 2.8, "PRO": -1.6,
    "SER": -0.8, "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL": 4.2,
}
"""Kyte-Doolittle hydropathy, by three-letter residue name."""

CHARGE: dict[str, float] = {"ASP": -1.0, "GLU": -1.0, "LYS": 1.0, "ARG": 1.0, "HIS": 0.1}
"""Formal residue charges used by TAP (histidine carries 0.1)."""


def rescaled_hydropathy(resname: str) -> float:
    """Return TAP's rescaled hydrophobicity, ``1 + (KD + 4.5) / 9``."""
    return 1.0 + (KYTE_DOOLITTLE[resname] + 4.5) / 9.0


GLY_HYDROPATHY = rescaled_hydropathy("GLY")
"""Value assigned to exposed salt-bridge partners in the PSH sum."""

SALT_BRIDGE_DONORS: dict[str, tuple[str, ...]] = {"LYS": ("NZ",), "ARG": ("NH1", "NH2")}
SALT_BRIDGE_ACCEPTORS: dict[str, tuple[str, ...]] = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2")}
SALT_BRIDGE_CUTOFF = 3.2
"""Donor-acceptor heavy-atom distance (A) defining a salt bridge."""

# --------------------------------------------------------------------------
# psa / NACCESS surface conventions (Oxford TNP, BSD-3-Clause)
# --------------------------------------------------------------------------

PROBE_RADIUS = 1.4

_C_ALIPHATIC, _C_AROMATIC, _N, _O, _S = 1.87, 1.76, 1.65, 1.40, 1.85

NACCESS_BACKBONE_RADII: dict[str, float] = {
    "N": _N, "CA": _C_ALIPHATIC, "C": _C_AROMATIC, "O": _O, "OXT": _O, "CB": _C_ALIPHATIC,
}

NACCESS_SIDE_CHAIN_RADII: dict[str, dict[str, float]] = {
    "ARG": {"CG": _C_ALIPHATIC, "CD": _C_ALIPHATIC, "NE": _N, "CZ": _C_AROMATIC,
            "NH1": _N, "NH2": _N},
    "ASN": {"CG": _C_AROMATIC, "OD1": _O, "ND2": _N},
    "ASP": {"CG": _C_AROMATIC, "OD1": _O, "OD2": _O},
    "CYS": {"SG": _S},
    "GLN": {"CG": _C_ALIPHATIC, "CD": _C_AROMATIC, "OE1": _O, "NE2": _N},
    "GLU": {"CG": _C_ALIPHATIC, "CD": _C_AROMATIC, "OE1": _O, "OE2": _O},
    "HIS": {"CG": _C_AROMATIC, "ND1": _N, "CD2": _C_AROMATIC, "CE1": _C_AROMATIC, "NE2": _N},
    "ILE": {"CG1": _C_ALIPHATIC, "CG2": _C_ALIPHATIC, "CD1": _C_ALIPHATIC},
    "LEU": {"CG": _C_ALIPHATIC, "CD1": _C_ALIPHATIC, "CD2": _C_ALIPHATIC},
    "LYS": {"CG": _C_ALIPHATIC, "CD": _C_ALIPHATIC, "CE": _C_ALIPHATIC, "NZ": 1.50},
    "MET": {"CG": _C_ALIPHATIC, "SD": _S, "CE": _C_ALIPHATIC},
    "PHE": {"CG": _C_AROMATIC, "CD1": _C_AROMATIC, "CD2": _C_AROMATIC,
            "CE1": _C_AROMATIC, "CE2": _C_AROMATIC, "CZ": _C_AROMATIC},
    "PRO": {"CG": _C_ALIPHATIC, "CD": _C_ALIPHATIC},
    "SER": {"OG": _O},
    "THR": {"OG1": _O, "CG2": _C_ALIPHATIC},
    "TRP": {"CG": _C_AROMATIC, "CD1": _C_AROMATIC, "CD2": _C_AROMATIC, "NE1": _N,
            "CE2": _C_AROMATIC, "CE3": _C_AROMATIC, "CZ2": _C_AROMATIC,
            "CZ3": _C_AROMATIC, "CH2": _C_AROMATIC},
    "TYR": {"CG": _C_AROMATIC, "CD1": _C_AROMATIC, "CD2": _C_AROMATIC, "CE1": _C_AROMATIC,
            "CE2": _C_AROMATIC, "CZ": _C_AROMATIC, "OH": _O},
    "VAL": {"CG1": _C_ALIPHATIC, "CG2": _C_ALIPHATIC},
}

ELEMENT_RADII: dict[str, float] = {"C": 1.80, "N": 1.60, "O": 1.40, "S": 1.85}

REFERENCE_SIDE_CHAIN_AREAS: dict[str, float] = {
    "ALA": 23.160, "ARG": 62.660, "ASN": 30.932, "ASP": 29.399, "CYS": 31.950,
    "GLN": 41.672, "GLU": 38.610, "GLY": 10.840, "HIS": 45.612, "ILE": 45.711,
    "LEU": 46.608, "LYS": 51.529, "MET": 51.944, "PHE": 51.536, "PRO": 39.281,
    "SER": 23.620, "THR": 31.930, "TRP": 65.080, "TYR": 52.912, "VAL": 37.887,
}
"""psa side-chain contact areas (A^2) in an extended Ala-X-Ala tripeptide."""

BACKBONE_ATOMS = frozenset({"N", "C", "O"})
"""psa counts every atom but backbone N, C and O as side chain (so CA and OXT count)."""

SURFACE_THRESHOLD = 7.5
"""Relative side-chain accessibility (%) above which TAP calls a residue exposed."""


def naccess_radius(resname: str, atom_name: str, element: str | None = None) -> float:
    """Return the NACCESS radius of one atom, mirroring ``tap_surface._radius``."""
    radius = NACCESS_SIDE_CHAIN_RADII.get(resname, {}).get(atom_name)
    if radius is None and resname in REFERENCE_SIDE_CHAIN_AREAS:
        radius = NACCESS_BACKBONE_RADII.get(atom_name)
    if radius is None:
        radius = ELEMENT_RADII.get(element or atom_name[:1], 1.80)
    return radius


# --------------------------------------------------------------------------
# RFD4-Proteina atom37 slot order
# --------------------------------------------------------------------------

AF2_ATOM_TYPES: tuple[str, ...] = (
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD",
    "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3",
    "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2", "OH", "CZ", "CZ2",
    "CZ3", "NZ", "OXT",
)
"""atom37 slot order used by RFD4-Proteina's decoder output."""

ATOM37_SLOT: dict[str, int] = {name: i for i, name in enumerate(AF2_ATOM_TYPES)}
CA_SLOT = ATOM37_SLOT["CA"]

# Per-residue atom37 layout, derived from the NACCESS tables plus the backbone.
# Mirrors the `aa` block of RFD4's vocabulary YAML for the standard 20 residues.
RESIDUE_ATOMS: dict[str, tuple[str, ...]] = {
    resname: ("N", "CA", "C", "O")
    + (("CB",) if resname != "GLY" else ())
    + tuple(NACCESS_SIDE_CHAIN_RADII.get(resname, {}))
    for resname in REFERENCE_SIDE_CHAIN_AREAS
}
"""Heavy atoms of each standard residue, excluding the optional C-terminal OXT."""


def verify_atom37_order() -> None:
    """Raise if RFD4-Proteina is importable and disagrees with ``AF2_ATOM_TYPES``.

    Call once at setup so a vocabulary change upstream fails loudly instead of
    silently scoring the wrong atoms.
    """
    try:
        from rfproteina.utils.constants import af2_atom_types
    except Exception:  # pragma: no cover - RFD4 not installed
        return
    if tuple(af2_atom_types) != AF2_ATOM_TYPES:
        raise RuntimeError(
            "RFD4-Proteina atom37 slot order has changed; update AF2_ATOM_TYPES. "
            f"Expected {AF2_ATOM_TYPES}, got {tuple(af2_atom_types)}."
        )
