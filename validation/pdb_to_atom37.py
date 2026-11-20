"""Load a two-chain Fv PDB into atom37 tensors, for validating against TAP2recreated.

Run these from the package root, e.g.
    PYTHONPATH=.:/path/to/TAP2recreated python validation/validate_hard.py structure.pdb
"""
from __future__ import annotations
import sys
import numpy as np
import torch
from Bio.PDB import PDBParser

sys.path.insert(0, ".")
from tap2_torch.constants import ATOM37_SLOT, REFERENCE_SIDE_CHAIN_AREAS
from tap2_torch.spec import build_spec


def load(pdb, heavy="H", light="L", dtype=torch.float64):
    model = next(PDBParser(QUIET=True).get_structure("fv", pdb).get_models())
    residues = [r for chain in model for r in chain if r.id[0] == " "]
    names = [r.resname for r in residues]
    assert not set(names) - REFERENCE_SIDE_CHAIN_AREAS.keys()
    n = len(residues)
    coords = np.zeros((n, 37, 3), dtype=np.float64)
    valid = np.zeros((n, 37), dtype=bool)
    oxt = np.zeros(n, dtype=bool)
    for i, residue in enumerate(residues):
        for atom in residue:
            if atom.element in {"H", "D"}:
                continue
            slot = ATOM37_SLOT.get(atom.id)
            if slot is None:
                continue
            coords[i, slot] = atom.coord
            valid[i, slot] = True
            if atom.id == "OXT":
                oxt[i] = True
    spec = build_spec(
        names,
        is_heavy_chain=np.array([r.parent.id == heavy for r in residues]),
        imgt_numbers=np.array([r.id[1] for r in residues]),
        include_oxt=oxt,
        dtype=dtype,
    )
    # Keep only slots the file actually provides.
    return torch.as_tensor(coords, dtype=dtype), spec, torch.as_tensor(valid), residues
