"""
modules/bleu.py
---------------
BLEU scorer extended with NMISS.

Key design choices (from the original implementation):

1. Preprocessing: regex-based tokenization preserving accented Italian characters
   (À-Ö, Ø-ö, ø-ÿ), without stemming. BLEU traditionally does not stem.

2. Brevity Penalty: standard BP on f_ref = BLEU(sys, ref), as in the BLEU
   definition of the paper and in the published results; set to 1 only for
   f_cxt, as disclosed in the paper footnote: "In the contextual component,
   X is applied only through its lexical overlap operator. Hence, in case of
   BLEU, the brevity penalty is set to 1 to fairly evaluate the context length."

3. Residual in BLEU space: computed directly on Counter objects (per n-gram
   order) without reconstructing a token list, avoiding spurious n-grams that
   would arise from re-stitching.

4. λ1 and λ2: sum of clipped match counts across all n-gram orders (n=1..4).

Algorithm 1 — Stage 1 (Reference-based):
    f_ref = BLEU(sys, ref)   with the standard brevity penalty
    λ1    = sum of clipped match counts across n=1..max_n

Algorithm 1 — Stage 2 (Context-based):
    N     = {sys_cnt_n - overlap_n}  for each n  (Counter subtraction)
    A     = N ∩ G(ctx)               for each n  → λ2 = sum of |A_n|
    f_cxt = BLEU(N, ctx)             with bp=1, operating on residual Counters

Algorithm 1 — Stage 3 (NMISS combination, Equation 3):
    f_NMISS = max(f_ref, (λ1·f_ref + λ2·f_cxt) / (λ1 + λ2))
"""

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from nltk.util import ngrams as nltk_ngrams

from modules.core import nmiss_combine

# f_ref uses the standard brevity penalty (paper definition and published results);
# f_cxt uses bp = 1 (paper footnote on the contextual component).
BLEU_REF_BREVITY_PENALTY = True


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class BleuNMISSResult:
    """
    All intermediate and final scores for one (system, reference, context)
    triple under BLEU-NMISS.

    Attributes
    ----------
    bleu_ref : float
        Baseline BLEU score  f_ref = BLEU(sys, ref)  with the standard brevity penalty.
    bleu_cxt : float
        Context BLEU score   f_cxt = BLEU(residual, ctx)  with bp=1.
    bleu_nmiss : float
        Final NMISS score (Equation 3).
    precisions_ref : List[float]
        Per-order modified precisions for f_ref (n=1..max_n).
    precisions_cxt : List[float]
        Per-order modified precisions for f_cxt (n=1..max_n).
    lam1 : int
        λ1 = sum of clipped match counts across n=1..max_n (reference).
    lam2 : int
        λ2 = sum of clipped match counts across n=1..max_n (context).
    max_n : int
        Maximum n-gram order used.
    weights : List[float]
        Per-order weights used for BLEU computation.
    """
    bleu_ref:       float
    bleu_cxt:       float
    bleu_nmiss:     float
    precisions_ref: List[float]
    precisions_cxt: List[float]
    lam1:           int
    lam2:           int
    max_n:          int
    weights:        List[float]


# ---------------------------------------------------------------------------
# BLEU primitives
# ---------------------------------------------------------------------------

def _bleu_from_precisions(
    precisions: List[float],
    bp: float,
    weights: List[float],
    smoothing: float = 1e-9,
) -> float:
    """
    Compute BLEU from per-order precisions, brevity penalty, and weights.

    Uses log-space summation for numerical stability. Zero precisions are
    replaced by *smoothing* to avoid log(0).
    """
    s = sum(
        w * math.log(p if p > 0 else smoothing)
        for w, p in zip(weights, precisions)
    )
    return bp * math.exp(s)


def _bleu_match_space(
    sys_tokens: List[str],
    ref_tokens: List[str],
    max_n: int = 4,
    weights: Optional[List[float]] = None,
    smoothing: float = 1e-9,
) -> Tuple[float, List[float], int, Dict[int, Counter], Dict[int, Counter]]:
    """
    Compute BLEU(sys, ref) with the standard brevity penalty and return all intermediate counters
    needed for the NMISS residual computation.

    Parameters
    ----------
    sys_tokens, ref_tokens : List[str]
        Preprocessed token lists.
    max_n : int
        Maximum n-gram order (default: 4).
    weights : List[float], optional
        Per-order weights. Defaults to uniform 1/max_n.
    smoothing : float
        Smoothing value for zero precisions.

    Returns
    -------
    bleu : float
    precisions : List[float]
    lam1 : int         — sum of clipped match counts across n=1..max_n
    sys_cnt_by_n : Dict[int, Counter]
    overlap_by_n : Dict[int, Counter]
    """
    if weights is None:
        weights = [1.0 / max_n] * max_n

    precisions    = []
    sys_cnt_by_n  = {}
    overlap_by_n  = {}
    lam1          = 0

    for n in range(1, max_n + 1):
        sys_ngr = list(nltk_ngrams(sys_tokens, n))
        ref_ngr = list(nltk_ngrams(ref_tokens,  n))

        sys_cnt     = Counter(sys_ngr)
        ref_cnt     = Counter(ref_ngr)
        overlap_cnt = sys_cnt & ref_cnt
        match_n     = sum(overlap_cnt.values())
        sys_total   = sum(sys_cnt.values())

        p_n = match_n / sys_total if sys_total > 0 else 0.0

        precisions.append(p_n)
        sys_cnt_by_n[n]  = sys_cnt
        overlap_by_n[n]  = overlap_cnt
        lam1            += match_n

    # standard brevity penalty on the reference component
    c, r = len(sys_tokens), len(ref_tokens)
    bp = 0.0 if c == 0 else (1.0 if c > r else math.exp(1.0 - r / c))
    bleu = _bleu_from_precisions(precisions, bp=bp, weights=weights, smoothing=smoothing)
    return bleu, precisions, lam1, sys_cnt_by_n, overlap_by_n


def _residual_counters(
    sys_cnt_by_n: Dict[int, Counter],
    overlap_by_n: Dict[int, Counter],
    max_n: int = 4,
) -> Dict[int, Counter]:
    """
    Compute residual n-gram Counters: sys_cnt_n - overlap_n (floored at 0).

    Works directly on Counter objects to avoid spurious n-grams that would
    arise from reconstructing a token list.

    Returns
    -------
    Dict[int, Counter]
        Residual Counter for each n-gram order.
    """
    return {
        n: sys_cnt_by_n[n] - overlap_by_n[n]
        for n in range(1, max_n + 1)
    }


def _bleu_on_residual(
    residual_by_n: Dict[int, Counter],
    ctx_tokens: List[str],
    max_n: int = 4,
    weights: Optional[List[float]] = None,
    smoothing: float = 1e-9,
) -> Tuple[float, List[float], int]:
    """
    Compute BLEU(residual, ctx) with bp=1, operating on residual Counters
    directly (no token reconstruction).

    Parameters
    ----------
    residual_by_n : Dict[int, Counter]
        Residual n-gram Counters from :func:`_residual_counters`.
    ctx_tokens : List[str]
        Preprocessed context tokens.
    max_n, weights, smoothing : see :func:`_bleu_match_space`.

    Returns
    -------
    bleu : float
    precisions : List[float]
    lam2 : int   — sum of context-supported residual match counts
    """
    if weights is None:
        weights = [1.0 / max_n] * max_n

    ctx_cnt_by_n = {
        n: Counter(list(nltk_ngrams(ctx_tokens, n)))
        for n in range(1, max_n + 1)
    }

    precisions = []
    lam2       = 0

    for n in range(1, max_n + 1):
        res_cnt = residual_by_n.get(n, Counter())
        ctx_cnt = ctx_cnt_by_n[n]

        overlap = res_cnt & ctx_cnt
        match_n = sum(overlap.values())
        total_n = sum(res_cnt.values())

        p_n = match_n / total_n if total_n > 0 else 0.0
        precisions.append(p_n)
        lam2 += match_n

    bleu = _bleu_from_precisions(precisions, bp=1.0, weights=weights, smoothing=smoothing)
    return bleu, precisions, lam2


# ---------------------------------------------------------------------------
# Public scorer
# ---------------------------------------------------------------------------

def bleu_nmiss(
    sys_tokens: List[str],
    ref_tokens: List[str],
    ctx_tokens: List[str],
    max_n: int = 4,
    weights: Optional[List[float]] = None,
    smoothing: float = 1e-9,
) -> BleuNMISSResult:
    """
    Compute BLEU extended with NMISS (Algorithm 1, Equation 3).

    Parameters
    ----------
    sys_tokens, ref_tokens, ctx_tokens : List[str]
        Preprocessed token lists (use :func:`utils.preprocessing.preprocess_text_bleu`).
    max_n : int
        Maximum n-gram order (default: 4).
    weights : List[float], optional
        Per-order weights. Defaults to uniform [1/max_n] * max_n.
    smoothing : float
        Smoothing for zero precisions in log-BLEU.

    Returns
    -------
    BleuNMISSResult
    """
    if weights is None:
        weights = [1.0 / max_n] * max_n

    # Stage 1 — Reference-based
    f_ref, prec_ref, lam1, sys_cnt_by_n, overlap_by_n = _bleu_match_space(
        sys_tokens, ref_tokens, max_n=max_n, weights=weights, smoothing=smoothing,
    )

    # Stage 2 — Context-based
    residual_by_n = _residual_counters(sys_cnt_by_n, overlap_by_n, max_n=max_n)
    f_cxt, prec_cxt, lam2 = _bleu_on_residual(
        residual_by_n, ctx_tokens, max_n=max_n, weights=weights, smoothing=smoothing,
    )

    # Stage 3 — NMISS combination (Equation 3)
    f_nmiss = nmiss_combine(f_ref, f_cxt, lam1, lam2)

    return BleuNMISSResult(
        bleu_ref=f_ref,
        bleu_cxt=f_cxt,
        bleu_nmiss=f_nmiss,
        precisions_ref=prec_ref,
        precisions_cxt=prec_cxt,
        lam1=lam1,
        lam2=lam2,
        max_n=max_n,
        weights=weights,
    )
