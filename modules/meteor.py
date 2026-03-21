"""
modules/meteor.py
-----------------
METEOR scorer extended with NMISS.

Algorithm 1 — Stage 1 (Reference-based):
    f_ref = METEOR(sys, ref)
    λ1    = clipped unigram overlap count between sys and ref tokens
            (proxy for |P_X| under METEOR matching; see note below).

Algorithm 1 — Stage 2 (Context-based):
    N     = G(sys) \\ P_X   (multiset difference on preprocessed unigrams)
    A     = N ∩ G(ctx)       →  λ2 = |A|
    f_cxt = METEOR(residual_tokens, ctx)

Algorithm 1 — Stage 3 (NMISS combination, Equation 3):
    f_NMISS = max(f_ref, (λ1·f_ref + λ2·f_cxt) / (λ1 + λ2))

Notes on the λ1 / λ2 proxy
---------------------------
METEOR's internal alignment combines exact match, stemming, and synonym
matching, making the true |P_X| difficult to extract without reimplementing
the aligner. Following the approximation disclosed in the paper
(Section 3.3.2, footnote), λ1 and λ2 are computed as clipped unigram
overlap counts on the preprocessed token lists.
"""

from collections import Counter
from dataclasses import dataclass
from typing import List

from nltk.translate.meteor_score import meteor_score

from modules.core import nmiss_combine
from utils.multiset import multiset_subtract, multiset_size, multiset_overlap, counter_to_tokens


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class MeteorNMISSResult:
    """
    All intermediate and final scores for one (system, reference, context)
    triple under METEOR-NMISS.

    Attributes
    ----------
    meteor_ref : float
        Baseline METEOR score  f_ref = METEOR(sys, ref).
    meteor_cxt : float
        Context METEOR score   f_cxt = METEOR(residual, ctx).
    meteor_nmiss : float
        Final NMISS score (Equation 3).
    lam1 : int
        λ1 proxy — clipped unigram overlap between sys and ref.
    lam2 : int
        λ2 = |A_X| — clipped unigram overlap between residual and ctx.
    residual_len : int
        Number of tokens in the residual sequence N(s_i, r_i).
    """
    meteor_ref:   float
    meteor_cxt:   float
    meteor_nmiss: float
    lam1:         int
    lam2:         int
    residual_len: int


# ---------------------------------------------------------------------------
# Public scorer
# ---------------------------------------------------------------------------

def meteor_nmiss(
    sys_toks: List[str],
    ref_toks: List[str],
    ctx_toks: List[str],
) -> MeteorNMISSResult:
    """
    Compute METEOR extended with NMISS (Algorithm 1, Equation 3).

    Parameters
    ----------
    sys_toks, ref_toks, ctx_toks : List[str]
        Preprocessed token lists for system, reference, and context.

    Returns
    -------
    MeteorNMISSResult

    Notes
    -----
    ``nltk.translate.meteor_score.meteor_score`` signature:
        meteor_score(references, hypothesis)
    The system answer is always the hypothesis; reference (or context)
    is always in the references list.
    """
    # Stage 1
    f_ref = meteor_score([ref_toks], sys_toks) if (ref_toks and sys_toks) else 0.0
    lam1, overlap_cnt = multiset_overlap(Counter(sys_toks), Counter(ref_toks))

    # Stage 2
    N_cnt    = multiset_subtract(Counter(sys_toks), overlap_cnt)
    residual = counter_to_tokens(N_cnt)
    lam2, _  = multiset_overlap(Counter(residual), Counter(ctx_toks))
    f_cxt    = meteor_score([ctx_toks], residual) if (ctx_toks and residual) else 0.0

    # Stage 3
    f_nmiss = nmiss_combine(f_ref, f_cxt, lam1, lam2)

    return MeteorNMISSResult(
        meteor_ref=f_ref,
        meteor_cxt=f_cxt,
        meteor_nmiss=f_nmiss,
        lam1=lam1,
        lam2=lam2,
        residual_len=len(residual),
    )
