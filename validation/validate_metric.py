"""End-to-end TAP2GuidanceMetric test against a mock RFD4 `decoded` namespace."""
import sys, types
import numpy as np, torch

sys.path.insert(0, ".")

# Stub rfproteina.encoding so the token-index mapping works without RFD4 installed.
# Token order follows resources/aa_rna_dna_glycan_ncaa.yaml: '<M>' then the 21 aa.
AA = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS",
      "MET","PHE","PRO","SER","THR","TRP","TYR","VAL","UNK"]
TOKENS = ["<M>"] + AA + ["NAG", "A", "DA"]
enc = types.SimpleNamespace(tokens=TOKENS, token_to_idx={t: i for i, t in enumerate(TOKENS)})
sys.modules["rfproteina"] = types.ModuleType("rfproteina")
mod = types.ModuleType("rfproteina.encoding"); mod.UNIFIED_ATOM37_ENCODING = enc
sys.modules["rfproteina.encoding"] = mod

from biotite.structure import AtomArray
from validation.pdb_to_atom37 import load
from tap2_torch.rfd4_metric import TAP2GuidanceMetric
from tap2_torch.score import tap2_scores

coords, spec, valid, residues = load("/home/claude/checks/fv.pdb", dtype=torch.float32)
spec32 = spec.to(dtype=torch.float32)
n = spec.n_residues
pad_extra = 7                      # padding + a non-protein token, as a real batch has
total = n + pad_extra

# --- mock decoded namespace, batch of 2 with padding ----------------------
atom37 = torch.zeros(2, total, 37, 3)
atom37[:, :n] = coords
atom37 = atom37.requires_grad_(True)

residue_type = torch.full((2, total), -1, dtype=torch.long)
for i, residue in enumerate(residues):
    residue_type[:, i] = enc.token_to_idx[residue.resname]
residue_type[:, n] = enc.token_to_idx["NAG"]       # a glycan token, must be ignored
padding_mask = torch.zeros(2, total, dtype=torch.bool)
padding_mask[:, : n + 1] = True                    # NAG is real but not scorable

# --- mock input AtomArray, one token per residue, IMGT pre-annotated ------
def make_array():
    array = AtomArray(n + 1)
    array.coord[:] = 0.0
    array.chain_id[:] = np.array([r.parent.id for r in residues] + ["H"])
    array.res_id[:] = np.array([r.id[1] for r in residues] + [999])
    array.res_name[:] = np.array([r.resname for r in residues] + ["NAG"])
    array.atom_name[:] = "CA"
    array.element[:] = "C"
    array.set_annotation("imgt_numbering",
                         np.array([r.id[1] for r in residues] + [-1], dtype=int))
    return array

metric = TAP2GuidanceMetric(heavy_chain="H", light_chain="L", strict=True,
                            score_kwargs={"n_sasa_points": 60})
out = metric.compute(atom37, residue_type, padding_mask, [make_array(), make_array()])

print("outputs:", ", ".join(sorted(out)))
print(f"n_scored = {float(out['n_scored'])}  (expect 2.0)")
reference = tap2_scores(coords, spec32, atom_valid=valid, n_sasa_points=60)
for key in ("psh", "ppc", "pnc", "sfvcsp", "cdr_length"):
    got, want = float(out[key]), 2 * float(reference[key])
    ok = abs(got - want) <= 1e-3 * max(abs(want), 1.0)
    print(f"  {key:<11} batch={got:>12.4f}  2x single={want:>12.4f}  {'OK' if ok else 'MISMATCH'}")

cost = out["developability_cost"]
(grad,) = torch.autograd.grad(cost, atom37)
print(f"\ndevelopability_cost = {float(cost):.6f}")
print(f"gradient norm       = {float(grad.norm()):.6f}")
print(f"gradient on padding = {float(grad[:, n + 1:].abs().max()):.3e}  (expect 0)")
print(f"gradient on glycan  = {float(grad[:, n].abs().max()):.3e}  (expect 0)")
print(f"finite              = {bool(torch.isfinite(grad).all())}")
