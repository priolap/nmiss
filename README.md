# NMISS — Negative Missing Information Scoring System

NMISS extends standard overlap-based metrics (ROUGE-1, ROUGE-2, ROUGE-L, METEOR) with a context-grounded correction component for retrieval-augmented generation (RAG) evaluation.

Traditional metrics penalize system answers that diverge lexically from the reference, even when the additional content is explicitly supported by the retrieved context. NMISS addresses this by assessing whether unmatched n-grams align with the external evidence, distinguishing grounded elaborations from hallucinations.

## Quickstart

```python
from modules.scorer import NMISS_Scorer

scorer = NMISS_Scorer(models=[], context_col="", ref_col="")

result = scorer.score_one(
    reference="I'm sorry, I cannot answer this question based on the available information.",
    system="I'm sorry, I cannot answer this question based on the available information. "
           "However, the document notes that regular physical activity and adequate rest "
           "are essential for maintaining good health.",
    context="Regular physical activity and adequate rest are key components of a healthy "
            "lifestyle. They contribute to cardiovascular health, mental well-being, and "
            "overall physical performance.",
)

print(f"ROUGE-1 Precision  baseline : {result['rouge1'].f_ref['precision']:.4f}")
print(f"ROUGE-1 Precision  NMISS    : {result['rouge1'].precision:.4f}")
print(f"METEOR             baseline : {result['meteor'].meteor_ref:.4f}")
print(f"METEOR             NMISS    : {result['meteor'].meteor_nmiss:.4f}")
```

## Motivating Example

A RAG system is asked a question outside the scope of the retrieved document. The reference answer is an abstention. The system correctly abstains but also adds recommendations explicitly supported by the retrieved context. ROUGE-1 Precision penalizes this as a mismatch; NMISS correctly rewards the grounded elaboration.

```bash
python examples/example_paper.py
```

## Structure

```
nmiss/
├── modules/
│   ├── core.py           # NMISS combination formula (Equation 3)
│   ├── rouge.py          # ROUGE-1, ROUGE-2, ROUGE-L extended with NMISS
│   ├── meteor.py         # METEOR extended with NMISS
│   └── scorer.py         # NMISS_Scorer: single-triple and batch scoring
├── utils/
│   ├── preprocessing.py  # Text normalization and tokenization
│   ├── multiset.py       # Generic multiset operations
│   └── plot.py           # Visualization utilities
├── examples/
│   └── example_paper.py  # Illustrative example of NMISS vs baseline metrics
├── main.py               # Entry point for batch scoring on your data
└── README.md
```

## Requirements

```
nltk
pandas
openpyxl
matplotlib
numpy
```

Download required NLTK data once:

```python
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("omw-1.4")
```

## Usage

### Score a single triple

```python
from modules.scorer import NMISS_Scorer

scorer = NMISS_Scorer(models=[], context_col="", ref_col="")

result = scorer.score_one(reference="...", system="...", context="...")

print(result["rouge1"].f1)            # ROUGE-1 NMISS F1
print(result["rouge1"].f_ref["f1"])   # ROUGE-1 baseline F1
print(result["meteor"].meteor_nmiss)  # METEOR NMISS
```

### Score a full DataFrame

NMISS expects a DataFrame where each row contains a retrieved context, a reference answer, and one or more system-generated answers — one column per system. It does **not** call any language model internally.

```python
# Expected DataFrame structure:
# | Context   | Reference | system_A  | system_B  |
# |-----------|-----------|-----------|-----------|
# | "..."     | "..."     | "..."     | "..."     |

import pandas as pd
from modules.scorer import NMISS_Scorer

df = pd.read_excel("your_data.xlsx")

scorer = NMISS_Scorer(
    models=["system_A", "system_B"],  # column names of system-generated answers
    context_col="Context",            # column name of retrieved context
    ref_col="Reference",              # column name of reference answer
)

scores_df = scorer.calculate_metrics_for_all_models(df)
scores_df.to_excel("nmiss_scores.xlsx", index=False)
```

### Run on your data

Edit the configuration block in `main.py` (file path, sheet name, model columns), then:

```bash
python main.py
```

## Equivalence Test

To verify numerical equivalence with the original implementation, place the two original scripts in the root folder and run:

```bash
python test_equivalence.py
```

Expected output:
```
Results: 12/12 test groups PASSED
ALL PASS — refactored code is numerically identical to original.
Results exported to: test_equivalence_results.csv
```

## Reference

If you use NMISS in your work, please cite:

```
@article{yourname2025nmiss,
  title   = {Your Paper Title},
  author  = {Your Name},
  journal = {Under review},
  year    = {2025},
  note    = {Preprint available upon request}
}
```