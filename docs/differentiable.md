# Differentiable TAP2 (`tap2_torch`)

Differentiable TAP2 developability scoring in PyTorch, with an
[RFD4-Proteina](https://github.com/RosettaCommons/RFD4-Proteina) metric-guidance
metric so the score can steer sampling instead of only filtering afterwards.

Takes atom37 coordinates (or an AtomWorks / biotite `AtomArray`) and returns
scalar tensors for total CDR length, PSH, PPC, PNC and SFvCSP, differentiable
with respect to the coordinates.

## Install

`tap2_torch/` lives beside `tap_score.py` in this repository. Copy
`configs/guidance/tap2.yaml` into RFD4-Proteina's own `configs/guidance/`, and
put `tap2_torch/` on the `PYTHONPATH` of the environment running
`rfd4-generate`. The extra dependencies are kept out of the base
`requirements.txt` so the published-score scorer stays light: `torch`, `numpy`
and `biotite` are required, `anarcii` only if you let it do IMGT numbering
itself, and RFD4-Proteina only for the guidance metric.

```sh
python -m pip install -r requirements-torch.txt
python -m pytest tests/ -q          # 9 tests, no downloads needed
```

## Use it standalone

```python
import torch
from tap2_torch import build_spec, tap2_scores

spec = build_spec(residue_names, is_heavy_chain=is_heavy, imgt_numbers=imgt)
coords = coords.requires_grad_(True)              # [n_residues, 37, 3], Angstrom
scores = tap2_scores(coords, spec)
scores["psh"].backward()
```

`spec` holds only what the sequence and numbering fix, so build it once per
design problem and reuse it every step.

## Use it to guide RFD4-Proteina

```sh
rfd4-generate inputs=... out_dir=./runs/tap2-guided num_replicates=1 +guidance=tap2
```

The metric reads the `decoded` namespace, so RFD4 runs its differentiable
autoencoder decoder on active steps and the gradient reaches `local_latents`
through full atom37 coordinates. `decoded.atom37_coords` carries gradient while
`decoded.residue_type` is discrete, which is the right split for TAP2:
hydropathy and charge are per-residue constants and all of the score's
coordinate dependence is geometric.

Guidance always *maximises*, so point a negative weight at
`tap2.developability_cost`. Start with the shipped window
(`t in [0.5, 0.9]`, `w = -0.02`), compare against the same seed unguided, and
raise the magnitude only once the trajectory is stable.

## What was made differentiable, and how

Four published rules are step functions with no useful gradient. Each has a
temperature-controlled replacement that converges to the original as the
temperature goes to zero.

| Published rule | Replacement | Knob |
| --- | --- | --- |
| relative side-chain SASA >= 7.5% | logistic gate on relative SASA | `tau_surface` |
| closest heavy-atom distance | soft-min over atom pairs | `tau_min` |
| hard pair cutoff at 7.5 A | cosine taper from 6.5 A | `taper_start` |
| 4 A CDR-vicinity test | logistic gate on distance | `tau_vicinity` |
| salt bridge within 3.2 A | logistic blend toward Gly hydropathy | `tau_salt` |

Side-chain accessibility is a **soft Shrake-Rupley**: test points on each atom's
expanded sphere, with the burial indicator replaced by a logistic in the signed
distance to each neighbour's sphere. Gradients reach coordinates twice over,
through the test points (which translate with their own atom) and through the
neighbour distances. The NACCESS radii, the `(r/(r+probe))^2` contact-area
rescale and the per-residue reference areas are kept from TNP's `surface.py`, so
the values stay on the scale TAP's 7.5% threshold was calibrated against.
Shrake-Rupley with 100 points also matches RFD4's own
`calculate_atomwise_rasa` convention. `psa`'s output rounding is dropped,
since rounding has zero gradient.

One further change is substantive rather than cosmetic. The published weight
`2/d^2` diverges as residues approach and its gradient `-4/d^3` diverges faster,
so a single mild clash dominates the gradient of the whole score.
`core_softening` (default **1.2 A**) replaces `d^2` with
`d^2 + core_softening^2`, which is bounded and has vanishing gradient at
contact. 1.2 A is the floor TAP2recreated already measured: it raised PSH
Pearson *r* against published scores from 0.195 to 0.689 on clash-bearing
FlashABB models. Set it to `0.0` to recover the published form exactly.

`hard=True` restores every published rule, including `core_softening = 0`. It is
not differentiable and exists to validate against the reference implementation.

## Validation

Run on `tests/data/synthetic_fv.pdb`, a two-chain Fv of **generated geometry**
(256 residues, 2137 heavy atoms, not a real antibody), in float64.

| Check | Result |
| --- | --- |
| Hard mode vs `tap_score.calculate`, reference exposure injected | max relative difference **1.8e-15** across all five metrics, insertion codes included |
| Hard-mode soft Shrake-Rupley vs independent brute-force SR | **exact** (0.0 A^2) |
| Shrake-Rupley vs FreeSASA Lee-Richards relative exposure | Pearson *r* **0.998**, mean abs difference 2.3%, and **100%** agreement on the 7.5% surface call (0 of 256 residues differ) |
| Analytic gradients vs central finite differences | agree to **1e-8** relative (PSH) |
| Soft to hard as temperatures shrink | monotone, within **0.01%** of hard at the tightest setting |
| Rigid-motion invariance | all five metrics invariant to rotation and translation |

Hard mode and the reference agree to machine precision only when the reference
exposure is injected, because the two use different SASA algorithms. The 100%
surface-call agreement above says the substitution does not change which
residues TAP calls exposed, but that is one synthetic structure: **re-run the
exposure comparison on the 664 archived ABodyBuilder2 models** before trusting
the soft score's absolute scale. `relative_exposure=` accepts the reference
FreeSASA values if you want reporting that matches published TAP2 exactly.

### Two findings worth knowing

**SFvCSP is a dead guidance target.** Its only dependence on coordinates runs
through the exposure gate, which is saturated for nearly every residue, so its
measured gradient norm is **0.0**. It is still reported, but
`weights.sfvcsp` defaults to zero. Treat SFvCSP as a filter, not an objective.

**PSH is flagged when too low as well as too high** (amber at `<= 110.11` and
`>= 168.06`). Minimising PSH would walk designs out of the clinical-stage
distribution into a maximally polar paratope. The cost outputs are therefore
two-sided hinges that are flat inside the green band and normalised by each
metric's amber-to-red span, so guidance stops pushing once a score is in range.

## Performance

CPU, 256 residues, forward plus backward, float32: **455 ms** at 60 SASA points,
**522 ms** at 100. No GPU was available to measure, so the GPU figure is
unmeasured; the work is dense tensor math on tensors of a few million elements
and should be far cheaper there. Two sparsifications are *exact* rather than
approximations, so they cost nothing in accuracy:

- the SASA neighbour list is sized from the data each call (a test point can
  only be occluded by atoms inside `r_i + r_j + 2*probe`);
- residue pairs are prefiltered by the exact bound
  `d(CA_i, CA_j) < reach_i + reach_j + cutoff`.

Reduce `n_sasa_points`, narrow the guidance window, or lower `pair_chunk` if
memory is tight.

## Caveats

- The CDR-vicinity radius is **4.0 A**, the value TAP2recreated found reproduces
  the published workbook, not the 4.5 A the TAP2 paper describes.
- TAP2 is defined on **IMGT** numbering. RFD4's own
  `annotate_antibody_numbering` uses Chothia, so it is deliberately not reused;
  these helpers request IMGT from ANARCII, or read an `imgt_numbering`
  annotation off the input array.
- `static_vicinity=True` (the default) resolves CDR masks once from the input
  CIF rather than renumbering every step. It assumes the token ordering of the
  coordinate tensor matches the input array's token order, which is the same
  correspondence RFD4 relies on when it slices per-token logits by the padding
  mask. Co-designed sequence that changes residue identity mid-trajectory is
  handled (hydropathy is re-gathered each step), but insertions that shift
  numbering are not.
- Non-protein tokens (ligands, glycans, nucleic acids) and padding are excluded
  and receive zero gradient.
- Guidance optimises exactly this proxy. It does not guarantee foldability or
  that a design binds anything; keep refolding and real TAP2 as post-hoc filters.

## Attribution

Derived from two sources, both acknowledged in the code:

- **TAP1/TAP2** scoring rules: Raybould et al., PNAS 2019
  ([doi:10.1073/pnas.1810576116](https://doi.org/10.1073/pnas.1810576116)) and
  Commun Biol 2024
  ([doi:10.1038/s42003-023-05744-8](https://doi.org/10.1038/s42003-023-05744-8)),
  as implemented in `tap_score.py`.
- **psa / NACCESS surface conventions**: radii and reference areas come from
  Oxford TNP's `surface.py` (BSD-3-Clause, University of Oxford), vendored in
  this repository as `tap_surface.py`. See [LICENCE](../LICENCE).

`tap2_torch/constants.py` keeps the three provenances in separate blocks so each
can be checked against its origin.
