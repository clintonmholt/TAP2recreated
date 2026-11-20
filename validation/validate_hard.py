"""Hard-mode agreement with TAP2recreated, exposure held fixed."""
import sys
import numpy as np
import torch

sys.path.insert(0, ".")

from validation.pdb_to_atom37 import load
from tap2_torch.score import tap2_scores
from tap_score import calculate, prepare
from tap_surface import side_chain_accessibility

torch.set_printoptions(precision=8)

for path in sys.argv[1:]:
    reference = calculate(prepare(path))
    coords, spec, valid, residues = load(path)

    # Reference exposure, exactly as tap_score.prepare computes it.
    relative, _ = side_chain_accessibility(path)
    exposure = torch.tensor(
        [relative.get(r.parent.id, {}).get(f"{r.id[1]}{r.id[2].strip()}", 0.0) for r in residues],
        dtype=torch.float64,
    )

    mine = tap2_scores(coords, spec, atom_valid=valid, relative_exposure=exposure, hard=True)
    ours = np.array([float(mine[k]) for k in ("cdr_length", "psh", "ppc", "pnc", "sfvcsp")])
    delta = np.abs(ours - reference)
    rel = delta / np.maximum(np.abs(reference), 1e-12)
    print(f"{path}")
    print(f"  reference : {np.round(reference, 6)}")
    print(f"  torch     : {np.round(ours, 6)}")
    print(f"  max|diff| = {delta.max():.3e}   max rel = {rel.max():.3e}  "
          f"{'MATCH' if rel.max() < 1e-9 else 'MISMATCH'}")
