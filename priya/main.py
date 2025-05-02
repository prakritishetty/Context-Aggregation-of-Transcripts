# 1. Install dependencies (once):
#    pip install datasets transformers sentence-transformers evaluate openai

from datasets import load_dataset
from soap_note_evaluator import SOAPNoteEvaluator
import numpy as np
import pandas as pd

def main():
    # 2. Load your dataset
    #    Replace with your actual path or HF repo if different
    ds = load_dataset("ClinicianFOCUS/ACI-Bench-Refined", split="train")

    # 3. Pull out the three fields:
    dialogues = ds["dialogue"]         # raw medical dialogue
    summary_notes = ds["note"]         # free-text summary (NOT SOAP)
    augmented_notes = ds["augmented note"]   # your generated SOAP notes

    # 4. Initialize the evaluator
    evaluator = SOAPNoteEvaluator(
        dialogues=dialogues,
        generated_notes=augmented_notes,
        summary_notes=summary_notes,      # so we also evaluate vs the free-text note
        openai_api_key=None,  # omit or set None to skip LLM judgments
        device="cpu"                  # or "cpu" if you don't have a GPU
    )

    # 5. Run every metric (dialogue-based and summary-based)
    results = evaluator.run_all()

    # 6. Peek at a few of the results
    print("First example, QA vs dialogue:", results["qa_diag"][0])
    print("First example, QA vs summary :", results["qa_sum"][0] if "qa_sum" in results else "—")
    print("First example, structure    :", results["structure"][0])

    # 7. Aggregate into a DataFrame for analysis
    df = pd.DataFrame({
        # QA factuality
        "qa_vs_dialogue": [r["accuracy"] for r in results["qa_diag"]],
        "qa_vs_summary": [r["accuracy"] for r in results.get("qa_sum", [])],
        # Entailment & hallucination
        "entail_vs_dialogue": [r["entailment_score"] for r in results["entail_diag"]],
        "halluc_vs_dialogue": [r["hallucination_rate"] for r in results["halluc_diag"]],
        "entail_vs_summary": [r["entailment_score"] for r in results.get("entail_sum", [])],
        "halluc_vs_summary": [r["hallucination_rate"] for r in results.get("halluc_sum", [])],
        # Semantic similarity
        "sem_sim_diag": [r["semantic_similarity"] for r in results["sem_diag"]],
        "sem_sim_sum": [r["semantic_similarity"] for r in results.get("sem_sum", [])],
        # Clinical BERT relevance
        "clin_f1_diag": [r["bertscore_f1"] for r in results["clin_diag"]],
        "clin_sim_diag": [r["embedding_similarity"] for r in results["clin_diag"]],
        "clin_f1_sum": [r["bertscore_f1"] for r in results.get("clin_sum", [])],
        "clin_sim_sum": [r["embedding_similarity"] for r in results.get("clin_sum", [])],
    })

    print(df.describe())

if __name__ == "__main__":
    main()
