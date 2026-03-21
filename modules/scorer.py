"""
modules/scorer.py
-----------------
NMISSScorer: unified class for scoring single triples and full DataFrames.

Exposes:
    score_one(reference, system, context)
        Score a single (reference, system, context) triple.

    calculate_metrics_for_all_models(df)
        Iterate over multiple system columns and return a combined DataFrame.
"""

from typing import Any, Dict, List

import pandas as pd

from modules.rouge import RougeNMISSResult, rouge_n_nmiss, rouge_l_nmiss
from modules.meteor import MeteorNMISSResult, meteor_nmiss
from modules.bleu import BleuNMISSResult, bleu_nmiss
from utils.preprocessing import normalize_text, preprocess_text, preprocess_text_bleu


# ---------------------------------------------------------------------------
# Internal column-packing helpers
# ---------------------------------------------------------------------------

def _pack_rouge(prefix: str, obj: RougeNMISSResult) -> Dict[str, Any]:
    return {
        f"{prefix}_REF_PREC":   obj.f_ref["precision"],
        f"{prefix}_REF_REC":    obj.f_ref["recall"],
        f"{prefix}_REF_F1":     obj.f_ref["f1"],
        f"{prefix}_CXT_PREC":   obj.f_cxt["precision"],
        f"{prefix}_CXT_REC":    obj.f_cxt["recall"],
        f"{prefix}_CXT_F1":     obj.f_cxt["f1"],
        f"{prefix}_NMISS_PREC": obj.precision,
        f"{prefix}_NMISS_REC":  obj.recall,
        f"{prefix}_NMISS_F1":   obj.f1,
        f"{prefix}_LAMBDA1":    obj.lam1,
        f"{prefix}_LAMBDA2":    obj.lam2,
    }


def _pack_meteor(obj: MeteorNMISSResult) -> Dict[str, Any]:
    return {
        "METEOR":         obj.meteor_ref,
        "METEOR_CONTEXT": obj.meteor_cxt,
        "METEOR_NMISS":   obj.meteor_nmiss,
        "METEOR_LAMBDA1": obj.lam1,
        "METEOR_LAMBDA2": obj.lam2,
        "RESIDUAL_LEN":   obj.residual_len,
    }


def _pack_bleu(obj: BleuNMISSResult) -> Dict[str, Any]:
    return {
        "BLEU_REF":     obj.bleu_ref,
        "BLEU_CXT":     obj.bleu_cxt,
        "BLEU_NMISS":   obj.bleu_nmiss,
        "BLEU_LAMBDA1": obj.lam1,
        "BLEU_LAMBDA2": obj.lam2,
    }


# ---------------------------------------------------------------------------
# Public scorer class
# ---------------------------------------------------------------------------

class NMISSScorer:
    """
    Compute ROUGE-NMISS, METEOR-NMISS, and/or BLEU-NMISS scores for
    retrieval-augmented generation evaluation.

    Parameters
    ----------
    models : List[str]
        Column names in the input DataFrame that contain system-generated
        answers. NMISS does not call any language model internally.
    context_col : str
        Column name of the retrieved context passages.
    ref_col : str
        Column name of the reference (human) answers.
    compute_rouge : bool, optional
        Whether to compute ROUGE-NMISS (default: True).
    compute_meteor : bool, optional
        Whether to compute METEOR-NMISS (default: True).
    compute_bleu : bool, optional
        Whether to compute BLEU-NMISS (default: True).
    bleu_max_n : int, optional
        Maximum n-gram order for BLEU (default: 4).
    """

    def __init__(
        self,
        models: List[str],
        context_col: str,
        ref_col: str,
        compute_rouge:  bool = True,
        compute_meteor: bool = True,
        compute_bleu:   bool = True,
        bleu_max_n:     int  = 4,
    ) -> None:
        self.models         = models
        self.context_col    = context_col
        self.ref_col        = ref_col
        self.compute_rouge  = compute_rouge
        self.compute_meteor = compute_meteor
        self.compute_bleu   = compute_bleu
        self.bleu_max_n     = bleu_max_n

    def _prep(self, reference: str, system: str, context: str):
        """Preprocessing for ROUGE and METEOR (stemming, NLTK tokenizer)."""
        ref_toks = preprocess_text(normalize_text(reference))
        sys_toks = preprocess_text(normalize_text(system))
        ctx_toks = preprocess_text(normalize_text(context))
        return ref_toks, sys_toks, ctx_toks

    def _prep_bleu(self, reference: str, system: str, context: str):
        """Preprocessing for BLEU (regex-based, no stemming)."""
        ref_toks = preprocess_text_bleu(reference)
        sys_toks = preprocess_text_bleu(system)
        ctx_toks = preprocess_text_bleu(context)
        return ref_toks, sys_toks, ctx_toks

    def score_one(self, reference: str, system: str, context: str) -> Dict[str, Any]:
        """
        Score a single (reference, system, context) triple.

        Returns
        -------
        dict with keys:
            "rouge1", "rouge2", "rougeL"  →  RougeNMISSResult  (if compute_rouge)
            "meteor"                       →  MeteorNMISSResult (if compute_meteor)
            "bleu"                         →  BleuNMISSResult   (if compute_bleu)
        """
        out: Dict[str, Any] = {}

        if self.compute_rouge or self.compute_meteor:
            ref_toks, sys_toks, ctx_toks = self._prep(reference, system, context)

            if self.compute_rouge:
                out["rouge1"] = rouge_n_nmiss(sys_toks, ref_toks, ctx_toks, n=1)
                out["rouge2"] = rouge_n_nmiss(sys_toks, ref_toks, ctx_toks, n=2)
                out["rougeL"] = rouge_l_nmiss(sys_toks, ref_toks, ctx_toks)

            if self.compute_meteor:
                out["meteor"] = meteor_nmiss(sys_toks, ref_toks, ctx_toks)

        if self.compute_bleu:
            ref_b, sys_b, ctx_b = self._prep_bleu(reference, system, context)
            out["bleu"] = bleu_nmiss(sys_b, ref_b, ctx_b, max_n=self.bleu_max_n)

        return out

    def calculate_metrics_for_all_models(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score all system columns over all rows of *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns self.ref_col, self.context_col, and one
            column per system in self.models.

        Returns
        -------
        pd.DataFrame
            One row per (DataFrame row × system) pair.
        """
        rows: List[Dict[str, Any]] = []

        for _, row in df.iterrows():
            question      = row.get("Question", "")
            level         = row.get("LEVEL",    "")
            row_id        = row.get("ID",        "")
            context_raw   = row.get(self.context_col, "")
            reference_raw = row.get(self.ref_col,     "")

            for model in self.models:
                system_raw = row.get(model, "")
                scores     = self.score_one(reference_raw, system_raw, context_raw)

                out: Dict[str, Any] = {
                    "DOMANDA": question,
                    "ID":      row_id,
                    "LEVEL":   level,
                    "MODEL":   model,
                }

                if self.compute_rouge:
                    out["CONTEXT"] = context_raw
                    out["HUMAN"]   = reference_raw
                    out["SYSTEM"]  = system_raw
                    out.update(_pack_rouge("ROUGE1", scores["rouge1"]))
                    out.update(_pack_rouge("ROUGE2", scores["rouge2"]))
                    out.update(_pack_rouge("ROUGEL", scores["rougeL"]))

                if self.compute_meteor:
                    out.update(_pack_meteor(scores["meteor"]))

                if self.compute_bleu:
                    out.update(_pack_bleu(scores["bleu"]))

                rows.append(out)

        return pd.DataFrame(rows)
