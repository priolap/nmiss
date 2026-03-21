"""
modules/rouge.py
----------------
ROUGE-N (n=1,2) and ROUGE-L scorers extended with NMISS.

Each scorer follows the three-stage pipeline from Algorithm 1 of the paper:

    Stage 1 — Reference-based computation
        f_ref = ROUGE-X(sys, ref)
        P_X   = matched units  →  λ1 = |P_X|

    Stage 2 — Context-based computation
        N     = G(sys) \\ P_X
        A     = N ∩ G(ctx)     →  λ2 = |A|
        f_cxt = ROUGE-X(N, ctx)

    Stage 3 — NMISS combination (Equation 3)
        f_NMISS = max(f_ref, (λ1·f_ref + λ2·f_cxt) / (λ1 + λ2))

NMISS is applied independently to precision, recall, and F1.

Note on ROUGE-L residual (see footnote, Section 3.3.2 of the paper):
    ROUGE-L uses LCS rather than n-gram multisets. The residual is
    therefore defined by removing the token *positions* that participate
    in one LCS alignment, rather than by multiset subtraction.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

from modules.core import nmiss_combine
from utils.multiset import multiset_subtract, multiset_size


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class RougeNMISSResult:
    """
    All intermediate and final scores for one (system, reference, context)
    triple under a single ROUGE variant.

    Attributes
    ----------
    precision, recall, f1 : float
        Final NMISS scores (after Equation 3).
    f_ref : dict
        Reference-based scores {"precision", "recall", "f1"}.
    f_cxt : dict
        Context-based scores {"precision", "recall", "f1"}.
    lam1 : int
        λ1 = |P_X|.
    lam2 : int
        λ2 = |A_X|.
    """
    precision: float
    recall:    float
    f1:        float
    lam1:      int
    lam2:      int
    f_ref:     Dict[str, float]
    f_cxt:     Dict[str, float]


# ---------------------------------------------------------------------------
# ROUGE-N helpers
# ---------------------------------------------------------------------------

def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 0:
        raise ValueError("n must be >= 1")
    if len(tokens) < n:
        return []
    return [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


def _rouge_n_counts(cand_tokens: List[str], ref_tokens: List[str], n: int) -> Dict:
    """Standard ROUGE-N with clipped n-gram counts."""
    cand_ngr    = _ngrams(cand_tokens, n)
    ref_ngr     = _ngrams(ref_tokens,  n)
    cand_cnt    = Counter(cand_ngr)
    ref_cnt     = Counter(ref_ngr)
    overlap_cnt = cand_cnt & ref_cnt
    overlap     = sum(overlap_cnt.values())
    cand_total  = sum(cand_cnt.values())
    ref_total   = sum(ref_cnt.values())

    precision = overlap / cand_total if cand_total > 0 else 0.0
    recall    = overlap / ref_total  if ref_total  > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "precision":   precision,
        "recall":      recall,
        "f1":          f1,
        "overlap":     float(overlap),
        "cand_total":  float(cand_total),
        "ref_total":   float(ref_total),
        "cand_cnt":    cand_cnt,
        "ref_cnt":     ref_cnt,
        "overlap_cnt": overlap_cnt,
    }


# ---------------------------------------------------------------------------
# ROUGE-L helpers
# ---------------------------------------------------------------------------

def _lcs_length(a: List[str], b: List[str]) -> int:
    """LCS length via space-optimised DP (O(|a|·|b|) time, O(|b|) space)."""
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[n]


def _lcs_alignment_indices(a: List[str], b: List[str]) -> List[int]:
    """
    Return indices in *a* that participate in one LCS alignment with *b*
    (recovered via full DP backtracking, deterministic top-left preference).
    """
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return []

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

    i, j = m, n
    idxs: List[int] = []
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            idxs.append(i - 1)
            i -= 1
            j -= 1
        else:
            if dp[i - 1][j] >= dp[i][j - 1]:
                i -= 1
            else:
                j -= 1

    idxs.reverse()
    return idxs


def _rouge_l_scores(cand_tokens: List[str], ref_tokens: List[str]) -> Dict[str, float]:
    """Standard ROUGE-L (token-level LCS)."""
    lcs      = _lcs_length(cand_tokens, ref_tokens)
    cand_len = len(cand_tokens)
    ref_len  = len(ref_tokens)

    precision = lcs / cand_len if cand_len > 0 else 0.0
    recall    = lcs / ref_len  if ref_len  > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "lcs":       float(lcs),
        "cand_len":  float(cand_len),
        "ref_len":   float(ref_len),
    }


# ---------------------------------------------------------------------------
# Public scorers
# ---------------------------------------------------------------------------

def rouge_n_nmiss(
    sys_toks: List[str],
    ref_toks: List[str],
    ctx_toks: List[str],
    n: int,
) -> RougeNMISSResult:
    """
    Compute ROUGE-N extended with NMISS (Algorithm 1, Equation 3).

    Parameters
    ----------
    sys_toks, ref_toks, ctx_toks : List[str]
        Preprocessed token lists for system, reference, and context.
    n : int
        N-gram order (1 for ROUGE-1, 2 for ROUGE-2).

    Returns
    -------
    RougeNMISSResult
    """
    # Stage 1
    ref_scores = _rouge_n_counts(sys_toks, ref_toks, n)
    f_ref = {k: ref_scores[k] for k in ("precision", "recall", "f1")}
    P_cnt: Counter = ref_scores["overlap_cnt"]
    lam1 = multiset_size(P_cnt)

    # Stage 2
    N_cnt: Counter = multiset_subtract(ref_scores["cand_cnt"], P_cnt)
    ctx_cnt = Counter(_ngrams(ctx_toks, n))
    A_cnt: Counter = N_cnt & ctx_cnt
    lam2 = multiset_size(A_cnt)

    overlap    = multiset_size(A_cnt)
    cand_total = multiset_size(N_cnt)
    ref_total  = multiset_size(ctx_cnt)

    precision_c = overlap / cand_total if cand_total > 0 else 0.0
    recall_c    = overlap / ref_total  if ref_total  > 0 else 0.0
    f1_c = (2 * precision_c * recall_c / (precision_c + recall_c)
            if (precision_c + recall_c) > 0 else 0.0)
    f_cxt = {"precision": precision_c, "recall": recall_c, "f1": f1_c}

    # Stage 3
    p = nmiss_combine(f_ref["precision"], f_cxt["precision"], lam1, lam2)
    r = nmiss_combine(f_ref["recall"],    f_cxt["recall"],    lam1, lam2)
    f = nmiss_combine(f_ref["f1"],        f_cxt["f1"],        lam1, lam2)

    return RougeNMISSResult(precision=p, recall=r, f1=f,
                            lam1=lam1, lam2=lam2, f_ref=f_ref, f_cxt=f_cxt)


def rouge_l_nmiss(
    sys_toks: List[str],
    ref_toks: List[str],
    ctx_toks: List[str],
) -> RougeNMISSResult:
    """
    Compute ROUGE-L extended with NMISS (Algorithm 1, Equation 3).

    Parameters
    ----------
    sys_toks, ref_toks, ctx_toks : List[str]
        Preprocessed token lists for system, reference, and context.

    Returns
    -------
    RougeNMISSResult
    """
    # Stage 1
    ref_scores = _rouge_l_scores(sys_toks, ref_toks)
    f_ref = {k: ref_scores[k] for k in ("precision", "recall", "f1")}
    lam1 = int(ref_scores["lcs"])

    # Stage 2 — sequence-level residual (see footnote Section 3.3.2)
    lcs_idxs     = set(_lcs_alignment_indices(sys_toks, ref_toks))
    residual_seq  = [tok for i, tok in enumerate(sys_toks) if i not in lcs_idxs]

    A_cnt  = Counter(residual_seq) & Counter(ctx_toks)
    lam2   = multiset_size(A_cnt)

    cxt_scores = _rouge_l_scores(residual_seq, ctx_toks)
    f_cxt = {k: cxt_scores[k] for k in ("precision", "recall", "f1")}

    # Stage 3
    p = nmiss_combine(f_ref["precision"], f_cxt["precision"], lam1, lam2)
    r = nmiss_combine(f_ref["recall"],    f_cxt["recall"],    lam1, lam2)
    f = nmiss_combine(f_ref["f1"],        f_cxt["f1"],        lam1, lam2)

    return RougeNMISSResult(precision=p, recall=r, f1=f,
                            lam1=lam1, lam2=lam2, f_ref=f_ref, f_cxt=f_cxt)
