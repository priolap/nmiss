"""
modules/core.py
---------------
Core NMISS scoring formula (Equation 3 of the paper), shared by all
metric variants (ROUGE-1, ROUGE-2, ROUGE-L, METEOR).

Reference — Equation (3):

    f_NMISS(s_i, r_i, c_i) = max(
        f_ref(s_i, r_i),
        (λ1 · f_ref + λ2 · f_cxt) / (λ1 + λ2)
    )

    Edge case (Algorithm 1, lines 12–14):
        if λ1 + λ2 = 0  →  f_NMISS = f_ref
"""


def nmiss_combine(f_ref: float, f_cxt: float, lam1: int, lam2: int) -> float:
    """
    Compute the NMISS score given reference and context scalar components.

    Parameters
    ----------
    f_ref : float
        Reference-based score  f_ref(s_i, r_i) = X(s_i, r_i).
    f_cxt : float
        Context-based score  f_cxt(s_i, c_i) = X(N(s_i, r_i), c_i).
    lam1 : int
        λ1 = |P_X(s_i, r_i)| — cardinality of reference-matched units.
    lam2 : int
        λ2 = |A_X(s_i, c_i)| — cardinality of context-supported residual units.

    Returns
    -------
    float
        NMISS score in the same range as the underlying metric X.
    """
    if lam1 + lam2 <= 0:
        return f_ref
    weighted_avg = (lam1 * f_ref + lam2 * f_cxt) / (lam1 + lam2)
    return max(f_ref, weighted_avg)
