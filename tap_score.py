"""Reconstruct published TAP2 metrics from an IMGT-numbered Fv PDB."""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from tap_surface import side_chain_accessibility

METRICS = ("CDR_length", "PSH", "PPC", "PNC", "SFvCSP")
CDRS = ((27, 38), (56, 65), (105, 117))
ANCHORS = tuple((start - 2, end + 2) for start, end in CDRS)
KYTE_DOOLITTLE = {
    "ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5,
    "GLN": -3.5, "GLU": -3.5, "GLY": -0.4, "HIS": -3.2, "ILE": 4.5,
    "LEU": 3.8, "LYS": -3.9, "MET": 1.9, "PHE": 2.8, "PRO": -1.6,
    "SER": -0.8, "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL": 4.2,
}
CHARGE = {"ASP": -1.0, "GLU": -1.0, "LYS": 1.0, "ARG": 1.0, "HIS": 0.1}
DONORS = {"LYS": ("NZ",), "ARG": ("NH1", "NH2")}
ACCEPTORS = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2")}


def in_ranges(numbers, ranges):
    return np.logical_or.reduce([(numbers >= low) & (numbers <= high)
                                 for low, high in ranges])


def prepare(pdb, heavy="H", light="L"):
    """Extract residue features and shortest squared heavy-atom distances."""
    model = next(PDBParser(QUIET=True).get_structure("fv", pdb).get_models())
    if heavy == light or set(model.child_dict) != {heavy, light}:
        raise ValueError(f"{pdb}: expected exactly the distinct Fv chains {heavy!r} and {light!r}")
    residues = [res for chain in model for res in chain if res.id[0] == " "]
    names = [res.resname for res in residues]
    if unknown := set(names) - KYTE_DOOLITTLE.keys():
        raise ValueError(f"{pdb}: unsupported residues {sorted(unknown)}")
    chain_ids = [res.parent.id for res in residues]
    if set(chain_ids) != {heavy, light}:
        raise ValueError(f"{pdb}: both chains must contain standard Fv residues")
    positions = np.array([res.id[1] for res in residues])
    if not all(in_ranges(positions[np.array(chain_ids) == chain], CDRS).any()
               for chain in (heavy, light)):
        raise ValueError(f"{pdb}: could not locate CDRs on both chains")

    relative, _ = side_chain_accessibility(str(pdb))
    exposure = np.array([relative.get(res.parent.id, {}).get(
        f"{res.id[1]}{res.id[2].strip()}", 0.0
    ) for res in residues])

    xyz, owner = [], []
    for i, residue in enumerate(residues):
        for atom in residue:
            if atom.element not in {"H", "D"}:
                xyz.append(atom.coord)
                owner.append(i)
    xyz, owner = np.asarray(xyz, dtype=float), np.asarray(owner)
    a, b = cKDTree(xyz).query_pairs(7.5, output_type="ndarray").T
    first, second = owner[a], owner[b]
    separate = first != second
    a, b, first, second = (values[separate] for values in (a, b, first, second))
    delta = xyz[a] - xyz[b]
    distance2 = np.einsum("ij,ij->i", delta, delta)
    minimum = np.full((len(residues), len(residues)), np.inf)
    np.minimum.at(minimum, (first, second), distance2)
    minimum = np.minimum(minimum, minimum.T)

    donor_xyz, donor_res, acceptor_xyz, acceptor_res = [], [], [], []
    for i, residue in enumerate(residues):
        for atom in residue:
            if atom.id in DONORS.get(residue.resname, ()):
                donor_xyz.append(atom.coord)
                donor_res.append(i)
            if atom.id in ACCEPTORS.get(residue.resname, ()):
                acceptor_xyz.append(atom.coord)
                acceptor_res.append(i)
    salt_left, salt_right = np.array([], dtype=int), np.array([], dtype=int)
    if donor_xyz and acceptor_xyz:
        u, v = np.where(cdist(donor_xyz, acceptor_xyz, "sqeuclidean") <= 3.2**2)
        salt_left, salt_right = np.asarray(donor_res)[u], np.asarray(acceptor_res)[v]

    return (
        positions, np.array(chain_ids) == heavy, exposure, minimum, salt_left, salt_right,
        np.array([1.0 + (KYTE_DOOLITTLE[name] + 4.5) / 9.0 for name in names]),
        np.array([CHARGE.get(name, 0.0) for name in names]),
    )


def calculate(features):
    """Apply the exposure, vicinity, patch, and chain-charge rules."""
    positions, heavy, exposure, distance2, donors, acceptors, hydropathy, charge = features
    hydropathy = hydropathy.copy()
    surface = exposure >= 7.5
    keep = surface[donors] & surface[acceptors]
    donors, acceptors = donors[keep], acceptors[keep]
    neutral = np.concatenate((donors, acceptors))
    hydropathy[neutral] = 1.0 + (KYTE_DOOLITTLE["GLY"] + 4.5) / 9.0

    anchors = in_ranges(positions, ANCHORS)
    cdr = in_ranges(positions, CDRS)
    # The reference workbook reproduces with exposed CDR and anchor seeds.
    seeds = anchors & surface
    near = (distance2[:, seeds] < 4.0**2).any(axis=1)
    vicinity = surface & (anchors | near)
    within = np.isfinite(distance2) & (distance2 <= 7.5**2)
    i, j = np.where(np.triu(within & vicinity[:, None] & vicinity, k=1))
    if (distance2[i, j] == 0).any():
        raise ValueError("Two residues have coincident heavy atoms in the scoring region")
    weight = 2 / distance2[i, j]  # TAP's sum includes both (i, j) and (j, i).
    positive = np.maximum(charge, 0)
    negative = np.maximum(-charge, 0)

    return np.array([
        cdr.sum(),
        np.dot(weight, hydropathy[i] * hydropathy[j]),
        np.dot(weight, positive[i] * positive[j]),
        np.dot(weight, negative[i] * negative[j]),
        charge[surface & heavy].sum() * charge[surface & ~heavy].sum(),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdb", type=Path, nargs="+", help="IMGT-numbered, two-chain Fv PDBs")
    parser.add_argument("--heavy", default="H", help="heavy-chain ID (default H)")
    parser.add_argument("--light", default="L", help="light-chain ID (default L)")
    args = parser.parse_args()
    writer = csv.writer(sys.stdout)
    writer.writerow(("model", *METRICS))
    for path in args.pdb:
        scores = calculate(prepare(path, args.heavy, args.light))
        writer.writerow((path.name, int(scores[0]), *(f"{s:.4f}" for s in scores[1:])))


if __name__ == "__main__":
    main()
