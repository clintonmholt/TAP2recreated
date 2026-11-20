"""RFD4-Proteina guidance metric for TAP2 developability.

Wires :func:`tap2_torch.score.tap2_scores` into RFD4-Proteina's metric-guidance
hook.  The guidance path hands a metric the model's clean prediction each active
sampling step, differentiates one scalar output with respect to ``bb_ca`` and
``local_latents``, and adds the weighted gradient to the sampler state.

This metric reads the ``decoded`` namespace, so RFD4 runs its differentiable
autoencoder decoder on active steps and the gradient reaches the latents through
full atom37 coordinates.  That is what makes side-chain-aware scoring possible
during sampling: ``decoded.atom37_coords`` carries gradient, while
``decoded.residue_type`` is discrete, which is exactly the right split for TAP2.
Hydropathy and charge are per-residue constants, and all of the score's
coordinate dependence is geometric.

Guidance always *maximises* the weighted objective, so point a negative-weight
schedule at a cost output such as ``developability_cost``, or a positive-weight
schedule at a reward.

Example config (``configs/guidance/tap2.yaml``)::

    metrics:
      tap2:
        _target_: tap2_torch.rfd4_metric.TAP2GuidanceMetric
        heavy_chain: H
        light_chain: L
    terms:
      tap2.developability_cost:
        bb_ca:
          _target_: rfproteina.guidance.schedules.window
          _partial_: true
          w: -0.02
          t_start: 0.5
          t_end: 0.9
"""

from __future__ import annotations

import logging
from functools import cached_property
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .constants import REFERENCE_SIDE_CHAIN_AREAS, verify_atom37_order
from .score import METRIC_KEYS, TAP2Spec, tap2_scores
from .spec import build_spec, imgt_numbers_for_sequence, in_ranges, ANCHOR_RANGES_IMGT, CDR_RANGES_IMGT

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised only with RFD4 installed
    from rfproteina.metrics.metric import Metric
except Exception:  # pragma: no cover - allows standalone testing

    class Metric:  # minimal stand-in with the same contract
        differentiable = False

        def __init__(self, **_: Any) -> None:
            pass


# Green/amber and amber/red limits from the 2024 TAP2 paper, Table 1, in the
# order (red_low, amber_low, amber_high, red_high); None means unbounded.
# Identical to CUTOFFS_TAP2 in TAP2recreated's benchmark.py.
TAP2_BOUNDS: dict[str, tuple[float | None, float | None, float | None, float | None]] = {
    "cdr_length": (37, 42, 55, 63),
    "psh": (95.58, 110.11, 168.06, 201.59),
    "ppc": (None, None, 1.32, 4.22),
    "pnc": (None, None, 2.00, 4.42),
    "sfvcsp": (-30.60, -6.00, None, None),
}

THREE_LETTER = set(REFERENCE_SIDE_CHAIN_AREAS)


def _two_sided_hinge(value: Tensor, bounds: tuple[float | None, ...]) -> Tensor:
    """Squared distance outside the green band, in units of the amber-to-red span.

    Zero inside the green band, so guidance stops pushing once a score is in the
    clinical-stage range rather than driving it to an extreme the TAP thresholds
    were never meant to endorse.  Normalising by the amber-to-red span puts the
    four metrics on a comparable scale.
    """
    red_low, amber_low, amber_high, red_high = bounds
    cost = torch.zeros((), dtype=value.dtype, device=value.device)
    if amber_high is not None:
        span = abs((red_high - amber_high) if red_high is not None else amber_high) or 1.0
        cost = cost + ((value - amber_high).clamp_min(0.0) / span) ** 2
    if amber_low is not None:
        span = abs((amber_low - red_low) if red_low is not None else amber_low) or 1.0
        cost = cost + ((amber_low - value).clamp_min(0.0) / span) ** 2
    return cost


class TAP2GuidanceMetric(Metric):
    """Differentiable TAP2 developability score for metric guidance.

    Args:
        heavy_chain: Chain ID of the heavy chain in the input annotated CIF.
        light_chain: Chain ID of the light chain. ``None`` scores a single-domain
            construct (a nanobody); SFvCSP is then zero by construction.
        weights: Per-metric multipliers for ``developability_cost``. Defaults to
            equal weight on ``psh``, ``ppc`` and ``pnc``, and zero on
            ``sfvcsp``, whose only coordinate dependence is through the exposure
            gate and whose gradient is therefore near zero in practice.
        static_vicinity: Resolve the CDR and anchor masks once from the input
            atom array and reuse them for every step. Recommended: it avoids
            re-running ANARCII inside the sampling loop, and in a design run the
            loop positions are fixed by the design problem anyway. When False,
            the masks are re-derived from each step's decoded sequence.
        strict: Raise when a sample cannot be scored. Off by default so a single
            unnumberable sample cannot abort a long generation run.
        score_kwargs: Forwarded to :func:`~tap2_torch.score.tap2_scores`, e.g.
            ``core_softening``, ``tau_surface``, ``n_sasa_points``.
        metric_kwargs: Forwarded to RFD4's ``Metric`` base (tags, aggregation).

    Returns from :meth:`compute`:
        ``psh``, ``ppc``, ``pnc``, ``sfvcsp``, ``cdr_length`` -- raw scores
        summed over the batch, so each sample receives its own gradient;
        ``*_cost`` -- green-band hinges evaluated per sample and summed;
        ``developability_cost`` -- their weighted sum, the usual guidance
        target; ``*_mean`` -- per-sample averages for logging, detached;
        ``n_scored`` -- how many samples contributed.
    """

    differentiable = True

    def __init__(
        self,
        *,
        heavy_chain: str = "H",
        light_chain: str | None = "L",
        weights: dict[str, float] | None = None,
        static_vicinity: bool = True,
        strict: bool = False,
        score_kwargs: dict[str, Any] | None = None,
        **metric_kwargs: Any,
    ) -> None:
        super().__init__(**metric_kwargs)
        verify_atom37_order()
        self.heavy_chain = heavy_chain
        self.light_chain = light_chain
        self.weights = {"psh": 1.0, "ppc": 1.0, "pnc": 1.0, "sfvcsp": 0.0} | (weights or {})
        self.static_vicinity = static_vicinity
        self.strict = strict
        self.score_kwargs = dict(score_kwargs or {})
        self._spec_cache: dict[Any, TAP2Spec] = {}
        self._vicinity_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._warned = False

    @cached_property
    def kwargs_to_compute_args(self) -> dict[str, tuple[str, ...]]:
        """Read decoded atom37 coordinates plus the input atom arrays.

        Reading ``("decoded", ...)`` makes RFD4 run its differentiable decoder on
        active steps, which is where the side-chain coordinates come from.
        """
        return {
            "atom37_coords": ("decoded", "atom37_coords"),
            "residue_type": ("decoded", "residue_type"),
            "padding_mask": ("decoded", "padding_mask"),
            "atom_array_batch": ("ground_truth_atom_array",),
        }

    @property
    def optional_kwargs(self) -> frozenset[str]:
        return frozenset({"atom_array_batch"})

    # ------------------------------------------------------------------
    # token metadata
    # ------------------------------------------------------------------

    @cached_property
    def _token_names(self) -> list[str]:
        """Map RFD4 token index to three-letter residue name."""
        from rfproteina.encoding import UNIFIED_ATOM37_ENCODING

        return list(UNIFIED_ATOM37_ENCODING.tokens)

    def _chain_and_vicinity(
        self, atom_array: Any, padding_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resolve per-token chain membership and CDR masks from an atom array.

        Uses IMGT numbering, which is what TAP2 is defined on. RFD4's own
        ``annotate_antibody_numbering`` uses Chothia, so it is not reused here.

        The coordinate tensor's residue axis is padded, while the atom array
        holds only real tokens, so the per-token values are scattered back onto
        the positions ``padding_mask`` marks as real, in order. That ordering is
        the same correspondence RFD4 itself relies on when it slices per-token
        logits by the padding mask.

        Args:
            atom_array: Input AtomArray for one sample.
            padding_mask: True for real tokens. Shape: ``[n]``.

        Returns:
            ``(is_heavy_chain, is_cdr, is_anchor)``, each ``[n]``, false on padding.

        Raises:
            ValueError: In strict mode, when the array's token count does not
                match the number of unpadded positions.
        """
        # AtomWorks tokens, not biotite residues: an atomized ligand contributes
        # one token per atom, so token starts and residue starts differ.
        try:
            from atomworks.ml.utils.token import get_token_starts
        except ImportError:  # standalone use without AtomWorks
            from biotite.structure import get_residue_starts as get_token_starts

        n_tokens = int(padding_mask.shape[0])
        real = np.flatnonzero(padding_mask)
        tokens = atom_array[get_token_starts(atom_array)]
        n_real = min(len(tokens), len(real))
        if len(tokens) != len(real):
            message = (
                f"atom array has {len(tokens)} tokens but {len(real)} unpadded positions; "
                "scoring the overlap only"
            )
            if self.strict:
                raise ValueError(message)
            if not self._warned:
                logger.warning("TAP2 guidance: %s", message)
                self._warned = True
        target = real[:n_real]

        chain_attr = "chain_iid" if "chain_iid" in atom_array.get_annotation_categories() else "chain_id"
        token_chain = np.asarray(getattr(tokens, chain_attr))[:n_real]
        is_heavy = np.zeros(n_tokens, dtype=bool)
        is_heavy[target] = token_chain == self.heavy_chain

        numbers = np.full(n_tokens, -1, dtype=int)
        categories = set(atom_array.get_annotation_categories())
        if "imgt_numbering" in categories:
            numbers[target] = np.asarray(tokens.imgt_numbering)[:n_real]
        else:
            from biotite.sequence import ProteinSequence

            token_names = np.asarray(tokens.res_name)[:n_real]
            for chain in np.unique(token_chain):
                rows = np.flatnonzero(token_chain == chain)
                keep = np.array([name in THREE_LETTER for name in token_names[rows]])
                if not keep.any():
                    continue
                sequence = "".join(
                    ProteinSequence.convert_letter_3to1(name) for name in token_names[rows][keep]
                )
                try:
                    numbers[target[rows[keep]]] = imgt_numbers_for_sequence(sequence)
                except Exception as error:
                    if self.strict:
                        raise
                    logger.warning("IMGT numbering failed for chain %s: %s", chain, error)
        return is_heavy, in_ranges(numbers, CDR_RANGES_IMGT), in_ranges(numbers, ANCHOR_RANGES_IMGT)

    def _spec_for_sample(
        self,
        residue_type: Tensor,
        scored: Tensor,
        padding_mask: Tensor,
        atom_array: Any,
        sample_index: int,
    ) -> TAP2Spec:
        """Build or reuse the per-residue spec for one sample."""
        names = [self._token_names[int(index)] for index in residue_type[scored].tolist()]
        if self.static_vicinity and sample_index in self._vicinity_cache:
            is_heavy, is_cdr, is_anchor = self._vicinity_cache[sample_index]
        else:
            is_heavy, is_cdr, is_anchor = self._chain_and_vicinity(
                atom_array, padding_mask.detach().cpu().numpy()
            )
            if self.static_vicinity:
                self._vicinity_cache[sample_index] = (is_heavy, is_cdr, is_anchor)
        mask = scored.detach().cpu().numpy()
        key = (tuple(names), mask.tobytes(), sample_index)
        cached = self._spec_cache.get(key)
        if cached is not None:
            return cached
        spec = build_spec(
            names,
            is_heavy_chain=is_heavy[mask] if self.light_chain is not None else np.ones(len(names), bool),
            is_cdr=is_cdr[mask],
            is_anchor=is_anchor[mask],
        )
        self._spec_cache[key] = spec
        return spec

    # ------------------------------------------------------------------

    def compute(
        self,
        atom37_coords: Tensor,
        residue_type: Tensor,
        padding_mask: Tensor,
        atom_array_batch: Any = None,
    ) -> dict[str, Tensor]:
        """Score a batch of decoded predictions.

        Args:
            atom37_coords: Decoded coordinates in Angstrom, with gradient.
                Shape: ``[b, n, 37, 3]``.
            residue_type: RFD4 token indices; ``-1`` outside the emitted set.
                Shape: ``[b, n]``.
            padding_mask: True for real tokens. Shape: ``[b, n]``.
            atom_array_batch: One input AtomArray per sample, carrying chain IDs
                and any numbering annotations.

        Returns:
            Raw scores, per-sample hinge costs, ``developability_cost`` and
            detached per-sample means, all scalar tensors. Guidance
            differentiates whichever one the config selects.
        """
        device, dtype = atom37_coords.device, atom37_coords.dtype
        zero = torch.zeros((), device=device, dtype=dtype)
        totals = {key: zero.clone() for key in METRIC_KEYS}
        # Hinges are evaluated per sample and then summed: the TAP2 thresholds
        # describe one antibody, so hinging a batch sum would compare a batch
        # total against a per-antibody limit.
        costs = {f"{key}_cost": zero.clone() for key in METRIC_KEYS}
        total_cost = zero.clone()
        n_scored = 0

        for index in range(atom37_coords.shape[0]):
            # Score only real, emitted, standard amino acids: ligands, glycans,
            # nucleic acids and padding carry no TAP chemistry.
            types = residue_type[index]
            standard = torch.tensor(
                [0 <= int(t) < len(self._token_names) and self._token_names[int(t)] in THREE_LETTER
                 for t in types.tolist()],
                device=device,
            )
            scored = padding_mask[index] & standard
            if int(scored.sum()) < 2:
                continue
            array = atom_array_batch[index] if atom_array_batch is not None else None
            if array is None:
                if self.strict:
                    raise ValueError("TAP2GuidanceMetric needs ground_truth_atom_array for chain IDs")
                if not self._warned:
                    logger.warning("no atom array available; skipping TAP2 guidance")
                    self._warned = True
                continue
            try:
                spec = self._spec_for_sample(
                    types, scored, padding_mask[index], array, index
                ).to(device=device, dtype=dtype)
                scores = tap2_scores(atom37_coords[index][scored], spec, **self.score_kwargs)
            except Exception as error:
                if self.strict:
                    raise
                if not self._warned:
                    logger.warning("TAP2 guidance skipped a sample: %s", error)
                    self._warned = True
                continue
            for key in METRIC_KEYS:
                totals[key] = totals[key] + scores[key]
                sample_cost = _two_sided_hinge(scores[key], TAP2_BOUNDS[key])
                costs[f"{key}_cost"] = costs[f"{key}_cost"] + sample_cost
                weight = self.weights.get(key, 0.0)
                if weight:
                    total_cost = total_cost + weight * sample_cost
            n_scored += 1

        scale = max(n_scored, 1)
        means = {f"{key}_mean": totals[key].detach() / scale for key in METRIC_KEYS}
        return {
            **totals,
            **costs,
            **means,
            "developability_cost": total_cost,
            "n_scored": torch.tensor(float(n_scored), device=device, dtype=dtype),
        }
