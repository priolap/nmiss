"""
main.py
-------
Entry point for NMISS scoring.

Usage
-----
    python main.py

Edit the configuration block below to point to your data file and
select which models and metrics to compute.
"""

import pandas as pd
from modules.scorer import NMISS_Scorer
from utils.plot import plot_average_metrics

# ---------------------------------------------------------------------------
# Configuration — edit here
# ---------------------------------------------------------------------------

RAG_MODEL   = "paraphrase-multilingual-MiniLM-L12-v2"
TEST_CHOICE = "medita-news-chunk512-over64"
SPECS       = "k=1"
ROOT        = "D:/OneDrive_uniCA/PhD/Shared/RAG/output"

ROOT_PATH   = f"{ROOT}/{RAG_MODEL}/{TEST_CHOICE}/{SPECS}" if SPECS else \
              f"{ROOT}/{RAG_MODEL}/{TEST_CHOICE}/"
FILE_PATH   = f"{ROOT_PATH}/models_comparison.xlsx"
SHEET_NAME  = "news-chunks512_over64"

MODELS      = ["gpt4", "llama2", "llama3", "mistral", "gemma", "gemma2"]
CONTEXT_COL = "Context"
REF_COL     = "HUMAN"

COMPUTE_ROUGE  = True
COMPUTE_METEOR = True

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)

    scorer = NMISS_Scorer(
        models=MODELS,
        context_col=CONTEXT_COL,
        ref_col=REF_COL,
        compute_rouge=COMPUTE_ROUGE,
        compute_meteor=COMPUTE_METEOR,
    )

    metrics_df = scorer.calculate_metrics_for_all_models(df)

    output_file = f"{ROOT_PATH}/metrics_nmiss.xlsx"
    metrics_df.to_excel(output_file, index=False)
    print(f"Saved: {output_file}")
    print(metrics_df.head())

    # Uncomment to generate plots (ROUGE only)
    # for prefix in ["ROUGE1", "ROUGE2", "ROUGEL"]:
    #     for metric, ylabel in [("PREC", "Precision"), ("REC", "Recall"), ("F1", "F1 Score")]:
    #         plot_average_metrics(metrics_df, MODELS, metric=metric, ylabel=ylabel, prefix=prefix)