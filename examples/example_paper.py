"""
examples/example_paper.py
--------------------------
Illustrates the core motivation of NMISS using a fictional example.

Scenario:
    A RAG system is asked a question that falls outside the scope of the
    retrieved document. The reference answer is an abstention. The system
    correctly abstains but also adds recommendations explicitly supported
    by the retrieved context.

    This causes ROUGE-1 Precision to be low (the extra content does not
    match the reference), while NMISS correctly rewards the grounded
    elaboration by checking whether the unmatched tokens appear in the
    retrieved context.

Run from the project root:
    python examples/example_paper.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.scorer import NMISSScorer

# ---------------------------------------------------------------------------
# Fictional example (same structure as the paper's motivating example)
# ---------------------------------------------------------------------------

QUESTION = "What is the recommended daily water intake for adults?"

REFERENCE = (
    "I'm sorry, I cannot answer this question based on the available information."
)

SYSTEM = (
    "I'm sorry, I cannot answer this question based on the available information, "
    "as the document does not directly address daily water intake recommendations. "
    "However, the document notes that staying hydrated is important for maintaining "
    "physical performance, and that individuals engaged in physical activity should "
    "increase their fluid intake accordingly. It also mentions that environmental "
    "conditions such as heat and humidity can affect hydration needs."
)

CONTEXT = (
    "Hydration plays a central role in physical performance and cognitive function. "
    "Individuals engaged in regular physical activity should pay particular attention "
    "to their fluid intake, especially under conditions of heat and humidity, which "
    "significantly increase hydration needs. Dehydration of even 2% of body weight "
    "can impair both physical and mental performance. It is therefore important to "
    "maintain adequate fluid balance throughout the day, adjusting intake based on "
    "activity level and environmental conditions."
)

# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

scorer = NMISSScorer(
    models=[],
    context_col="",
    ref_col="",
    compute_rouge=True,
    compute_meteor=True,
)

result = scorer.score_one(
    reference=REFERENCE,
    system=SYSTEM,
    context=CONTEXT,
)

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------

print("=" * 60)
print("NMISS — Illustrative Example")
print(f"Question: {QUESTION}")
print("=" * 60)
print()
print("The system correctly abstains but adds context-grounded content.")
print("ROUGE-1 Precision penalizes the extra tokens as a mismatch.")
print("NMISS rewards them because they are supported by the context.")
print()

print("── ROUGE-1 ──────────────────────────────────────────────")
print(f"  Precision   baseline : {result['rouge1'].f_ref['precision']:.4f}")
print(f"  Precision   NMISS    : {result['rouge1'].precision:.4f}")
print(f"  Recall      baseline : {result['rouge1'].f_ref['recall']:.4f}")
print(f"  Recall      NMISS    : {result['rouge1'].recall:.4f}")
print(f"  F1          baseline : {result['rouge1'].f_ref['f1']:.4f}")
print(f"  F1          NMISS    : {result['rouge1'].f1:.4f}")
print(f"  λ1 = {result['rouge1'].lam1}   λ2 = {result['rouge1'].lam2}")

print()
print("── ROUGE-2 ──────────────────────────────────────────────")
print(f"  Precision   baseline : {result['rouge2'].f_ref['precision']:.4f}")
print(f"  Precision   NMISS    : {result['rouge2'].precision:.4f}")
print(f"  Recall      baseline : {result['rouge2'].f_ref['recall']:.4f}")
print(f"  Recall      NMISS    : {result['rouge2'].recall:.4f}")
print(f"  F1          baseline : {result['rouge2'].f_ref['f1']:.4f}")
print(f"  F1          NMISS    : {result['rouge2'].f1:.4f}")
print(f"  λ1 = {result['rouge2'].lam1}   λ2 = {result['rouge2'].lam2}")

print()
print("── ROUGE-L ──────────────────────────────────────────────")
print(f"  Precision   baseline : {result['rougeL'].f_ref['precision']:.4f}")
print(f"  Precision   NMISS    : {result['rougeL'].precision:.4f}")
print(f"  Recall      baseline : {result['rougeL'].f_ref['recall']:.4f}")
print(f"  Recall      NMISS    : {result['rougeL'].recall:.4f}")
print(f"  F1          baseline : {result['rougeL'].f_ref['f1']:.4f}")
print(f"  F1          NMISS    : {result['rougeL'].f1:.4f}")
print(f"  λ1 = {result['rougeL'].lam1}   λ2 = {result['rougeL'].lam2}")

print()
print("── METEOR ───────────────────────────────────────────────")
print(f"  METEOR      baseline : {result['meteor'].meteor_ref:.4f}")
print(f"  METEOR      NMISS    : {result['meteor'].meteor_nmiss:.4f}")
print(f"  λ1 = {result['meteor'].lam1}   λ2 = {result['meteor'].lam2}")

print()
print("=" * 60)
print("NMISS >= baseline for all metrics: grounded elaboration")
print("is rewarded rather than penalized.")
print("=" * 60)
