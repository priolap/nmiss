"""
utils/preprocessing.py
-----------------------
Text normalization and tokenization for Italian text.

Pipeline (order matters):
    1. Normalize whitespace
    2. Lowercase
    3. Tokenize (Italian, NLTK word_tokenize)
    4. Remove non-alphanumeric characters
    5. Stem tokens longer than 3 characters (SnowballStemmer, Italian)

Stopword removal is intentionally NOT applied so that token multisets
remain consistent with the multiset operations defined in the NMISS
formulation (Algorithm 1 of the paper).
"""

import re
from typing import Any, List

from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

_STEMMER      = SnowballStemmer("italian")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_text(text: Any) -> str:
    """
    Collapse multiple whitespace characters into a single space and strip
    leading/trailing whitespace.

    Non-string inputs are coerced to an empty string.

    Parameters
    ----------
    text : Any
        Raw input value (typically a spreadsheet cell).

    Returns
    -------
    str
        Whitespace-normalized string, or ``""`` if *text* is not a string.
    """
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().split())


def preprocess_text(text: str) -> List[str]:
    """
    Apply the full preprocessing pipeline and return a list of tokens.

    Parameters
    ----------
    text : str
        Input string, already whitespace-normalized.

    Returns
    -------
    List[str]
        Ordered list of preprocessed tokens (multiplicities preserved).
    """
    text = (text or "").lower()
    raw_tokens = word_tokenize(text, language="italian")

    tokens: List[str] = []
    for tok in raw_tokens:
        tok = _NON_ALNUM_RE.sub("", tok)
        if not tok:
            continue
        if len(tok) > 3:
            tok = _STEMMER.stem(tok)
        tokens.append(tok)

    return tokens


# ---------------------------------------------------------------------------
# BLEU-specific preprocessing
# ---------------------------------------------------------------------------

# Preserves accented Italian characters (e.g. città, perché) without stemming.
# BLEU traditionally does not apply stemming.
_BLEU_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+", re.UNICODE)


def preprocess_text_bleu(text: Any) -> List[str]:
    """
    Lowercase and tokenize using a Unicode-aware regex that preserves
    accented Italian characters. No stemming is applied.

    Used exclusively by the BLEU-NMISS scorer.

    Parameters
    ----------
    text : Any
        Raw input value.

    Returns
    -------
    List[str]
        Ordered list of lowercase tokens (multiplicities preserved).
    """
    if not isinstance(text, str):
        return []
    return _BLEU_TOKEN_RE.findall(text.lower())
