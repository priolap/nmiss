"""
utils/multiset.py
-----------------
Generic multiset operations based on collections.Counter.

All functions are metric-agnostic and reusable independently of NMISS.

Reference (NMISS paper, Section 3.3.2):
    G(·)            multiset of lexical n-gram units extracted from a text
    N(s_i, r_i)     = G(s_i) \\ P_X(s_i, r_i)   [multiset difference]
    A_X(s_i, c_i)   = N(s_i, r_i) ∩ G(c_i)       [multiset intersection]
    λ1 = |P_X|,     λ2 = |A_X|                    [cardinalities]
"""

from collections import Counter
from typing import List, Tuple


def multiset_subtract(a: Counter, b: Counter) -> Counter:
    """
    Multiset difference  a \\ b  (floors multiplicities at 0).

    Python's Counter subtraction already floors at zero and drops
    non-positive counts, which is exactly the multiset-difference semantics.

    Parameters
    ----------
    a, b : Counter

    Returns
    -------
    Counter
        Multiset difference with non-negative multiplicities.
    """
    return a - b


def multiset_size(cnt: Counter) -> int:
    """
    Cardinality of a multiset (sum of all multiplicities).

    Parameters
    ----------
    cnt : Counter

    Returns
    -------
    int
    """
    return int(sum(cnt.values()))


def multiset_overlap(a: Counter, b: Counter) -> Tuple[int, Counter]:
    """
    Clipped (multiset) intersection of two Counters.

    For each element the clipped count is min(a[e], b[e]), which
    corresponds to the standard n-gram clipping used in BLEU/ROUGE.

    Parameters
    ----------
    a, b : Counter

    Returns
    -------
    overlap_count : int
        Total number of clipped matches.
    overlap_cnt : Counter
        The intersection multiset.
    """
    overlap_cnt = a & b
    return multiset_size(overlap_cnt), overlap_cnt


def counter_to_tokens(cnt: Counter) -> List[str]:
    """
    Expand a Counter back into a flat token list, preserving multiplicities.

    Used to reconstruct the residual token list from its Counter
    representation before passing it to METEOR.

    Parameters
    ----------
    cnt : Counter

    Returns
    -------
    List[str]
    """
    return [tok for tok, k in cnt.items() for _ in range(k)]
