"""Benchmark the differentiable scorer against the published one on real models.

Answers three questions across a directory of IMGT-numbered Fv PDBs:

1. Is the torch port faithful?  Hard mode with the reference FreeSASA exposure
   injected should reproduce ``tap_score.calculate`` to machine precision.  Any
   drift here is a port bug.
2. What does swapping Lee-Richards for Shrake-Rupley cost?  Reported as the
   correlation of relative side-chain exposure, and as the fraction of residues
   whose 7.5% surface call changes.  This is the gap the README's validation
   table leaves open on real structures.
3. How far is the differentiable score from the published one?  Hard mode with
   Shrake-Rupley exposure, then the soft score at the shipped temperatures,
   with and without the 1.2 A core softening.

Writes one row per model and prints an aggregate table.

Usage:
    PYTHONPATH=. python validation/benchmark_exposure.py \\
        /path/to/ABB2_AllCSTModels/CST_Models_ABB2_Run1 \\
        --workers 2 --out differentiable_benchmark.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

METRICS = ("cdr_length", "psh", "ppc", "pnc", "sfvcsp")
VARIANTS = ("hard_lr", "hard_sr", "soft_sr", "soft_sr_floored")


def score_one(path: Path) -> dict[str, float] | None:
    """Score one model every way, or return None if it cannot be processed."""
    from tap_score import calculate, prepare
    from tap_surface import side_chain_accessibility

    from tap2_torch.constants import ATOM37_SLOT
    from tap2_torch.score import tap2_scores
    from tap2_torch.spec import build_spec
    from Bio.PDB import PDBParser

    try:
        reference = calculate(prepare(str(path)))

        model = next(PDBParser(QUIET=True).get_structure("fv", str(path)).get_models())
        residues = [r for chain in model for r in chain if r.id[0] == " "]
        n = len(residues)
        coords = np.zeros((n, 37, 3))
        valid = np.zeros((n, 37), dtype=bool)
        oxt = np.zeros(n, dtype=bool)
        for i, residue in enumerate(residues):
            for atom in residue:
                if atom.element in {"H", "D"}:
                    continue
                slot = ATOM37_SLOT.get(atom.id)
                if slot is None:
                    continue
                coords[i, slot], valid[i, slot] = atom.coord, True
                oxt[i] |= atom.id == "OXT"
        spec = build_spec(
            [r.resname for r in residues],
            is_heavy_chain=np.array([r.parent.id == "H" for r in residues]),
            imgt_numbers=np.array([r.id[1] for r in residues]),
            include_oxt=oxt,
            dtype=torch.float64,
        )
        coords_t = torch.as_tensor(coords, dtype=torch.float64)
        valid_t = torch.as_tensor(valid)

        relative, _ = side_chain_accessibility(str(path))
        exposure_lr = torch.tensor(
            [relative.get(r.parent.id, {}).get(f"{r.id[1]}{r.id[2].strip()}", 0.0) for r in residues],
            dtype=torch.float64,
        )

        common = dict(spec=spec, atom_valid=valid_t, n_sasa_points=100)
        runs = {
            "hard_lr": tap2_scores(coords_t, relative_exposure=exposure_lr, hard=True, **common),
            "hard_sr": tap2_scores(coords_t, hard=True, **common),
            "soft_sr": tap2_scores(coords_t, core_softening=0.0, **common),
            "soft_sr_floored": tap2_scores(coords_t, core_softening=1.2, **common),
        }

        # Relative side-chain exposure under Shrake-Rupley, for the SASA comparison.
        from tap2_torch.soft_sasa import contact_area, soft_sasa

        residue_index, slot_index = (spec.atom_valid & valid_t).nonzero(as_tuple=True)
        radii = spec.atom_radii[residue_index, slot_index]
        sasa = soft_sasa(coords_t[residue_index, slot_index], radii, n_points=100, hard=True)
        contact = contact_area(sasa, radii) * spec.is_side_chain[residue_index, slot_index]
        area = torch.zeros(n, dtype=torch.float64).index_put((residue_index,), contact, accumulate=True)
        exposure_sr = 100.0 * area / spec.reference_area

        lr, sr = exposure_lr.numpy(), exposure_sr.numpy()
        row: dict[str, float] = {
            "model": path.name,
            "n_residues": n,
            "exposure_pearson": float(np.corrcoef(lr, sr)[0, 1]) if n > 2 else float("nan"),
            "exposure_mae": float(np.abs(lr - sr).mean()),
            "surface_call_disagree": int(((lr >= 7.5) != (sr >= 7.5)).sum()),
        }
        for j, metric in enumerate(METRICS):
            row[f"{metric}_reference"] = float(reference[j])
            for variant in VARIANTS:
                row[f"{metric}_{variant}"] = float(runs[variant][metric])
        return row
    except Exception as error:  # a bad model should not abort the sweep
        return {"model": path.name, "error": f"{type(error).__name__}: {error}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdb_folder", type=Path, help="directory of IMGT-numbered Fv PDBs")
    parser.add_argument("--out", type=Path, default=Path("differentiable_benchmark.csv"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="score only the first N models")
    args = parser.parse_args()

    paths = sorted(args.pdb_folder.glob("*.pdb"))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        parser.error(f"no .pdb files in {args.pdb_folder}")
    print(f"scoring {len(paths)} models with {args.workers} worker(s)", flush=True)

    rows = []
    with ProcessPoolExecutor(max(args.workers, 1)) as pool:
        for i, row in enumerate(pool.map(score_one, paths), start=1):
            rows.append(row)
            if i % 25 == 0 or i == len(paths):
                print(f"  {i}/{len(paths)}", flush=True)

    good = [r for r in rows if "error" not in r]
    failed = [r for r in rows if "error" in r]
    fieldnames = list(good[0]) if good else ["model", "error"]
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames + ["error"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nscored {len(good)} models, {len(failed)} failed -> {args.out}")
    for row in failed[:5]:
        print(f"  FAILED {row['model']}: {row['error']}")
    if not good:
        return

    def column(name: str) -> np.ndarray:
        return np.array([r[name] for r in good], dtype=float)

    print("\n--- relative side-chain exposure: Shrake-Rupley vs FreeSASA Lee-Richards ---")
    disagree = column("surface_call_disagree")
    residues = column("n_residues")
    print(f"  Pearson r            : median {np.median(column('exposure_pearson')):.4f}  "
          f"min {column('exposure_pearson').min():.4f}")
    print(f"  mean |difference|    : median {np.median(column('exposure_mae')):.2f} percent  "
          f"max {column('exposure_mae').max():.2f}")
    print(f"  7.5% call disagreement: {disagree.sum():.0f} of {residues.sum():.0f} residues "
          f"({100 * disagree.sum() / residues.sum():.2f}%), "
          f"worst model {disagree.max():.0f} residues")

    print("\n--- agreement with published TAP2 scores ---")
    header = f"{'metric':<11} {'variant':<17} {'Pearson':>8} {'Spearman':>9} {'MAE':>10} {'bias':>10} {'max|err|':>10}"
    print(header)
    from scipy.stats import pearsonr, spearmanr

    for metric in METRICS:
        want = column(f"{metric}_reference")
        for variant in VARIANTS:
            got = column(f"{metric}_{variant}")
            diff = got - want
            if np.allclose(want, want[0]) or np.allclose(got, got[0]):
                r = s = float("nan")
            else:
                r, s = pearsonr(got, want).statistic, spearmanr(got, want).statistic
            print(f"{metric:<11} {variant:<17} {r:>8.4f} {s:>9.4f} "
                  f"{np.abs(diff).mean():>10.4f} {diff.mean():>10.4f} {np.abs(diff).max():>10.4f}")

    print("\nhard_lr is the port-fidelity check: it should be ~0 on every line above.")


if __name__ == "__main__":
    main()
