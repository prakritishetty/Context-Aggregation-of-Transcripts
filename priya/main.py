# 1. Install dependencies (once):
#    pip install datasets transformers sentence-transformers evaluate openai

from datasets import load_dataset
from soap_note_evaluator import SOAPNoteEvaluator
import numpy as np
import pandas as pd

def main():
    # 2. Load your dataset
    #    Replace with your actual path or HF repo if different
    ds_all = load_dataset("ClinicianFOCUS/ACI-Bench-Refined", split="train")

    # number of examples you want
    n = 2
    ds = ds_all.select(range(n))

    # 3. Pull out the three fields:
    dialogues = ds["dialogue"]       # raw medical dialogue
    summary_notes = ds["note"]       # free-text summary (NOT SOAP)
    augmented_notes = ds["augmented note"]   # your generated SOAP notes

    # 4. Initialize the evaluator
    evaluator = SOAPNoteEvaluator(
        dialogues=dialogues,
        generated_notes=augmented_notes,
        summary_notes=summary_notes,      # so we also evaluate vs the free-text note
        openai_api_key=None,  # omit or set None to skip LLM judgments
        device="cpu"          # or "cpu" if you don't have a GPU
    )

    # 5. Run every metric (dialogue-based and summary-based)
    results = evaluator.run_all()
    
    # 6. Convert results to a dataframe
    df_results = create_dataframe(results, n)
    
    # 7. Display the dataframe
    print(df_results)
    
    return df_results

def create_dataframe(results, n_samples):
    """
    Convert evaluation results to a pandas DataFrame
    """
    # Initialize a list to store data for each sample
    data = []
    
    # Process each sample
    for i in range(n_samples):
        sample_data = {
            'sample_id': i,
            # Structure metrics
            'section_presence_S': results['structure'][i]['section_presence']['S'],
            'section_presence_O': results['structure'][i]['section_presence']['O'],
            'section_presence_A': results['structure'][i]['section_presence']['A'],
            'section_presence_P': results['structure'][i]['section_presence']['P'],
            'order_correct': results['structure'][i]['order_correct'],
            # QA factuality metrics
            'qa_accuracy': results['qa_comb'][i]['accuracy'],
            'qa_n_questions': results['qa_comb'][i]['n_questions'],
            # Entailment metrics
            'entailment_score': results['entail_comb'][i]['entailment_score'],
            # Hallucination metrics
            'hallucination_rate': results['halluc_comb'][i]['hallucination_rate'],
            # Semantic similarity metrics
            'semantic_similarity': results['sem_comb'][i]['semantic_similarity'],
            # Clinical relevance metrics
            'bertscore_f1': results['clin_comb'][i]['bertscore_f1'],
            'embedding_similarity': results['clin_comb'][i]['embedding_similarity']
        }
        
        # Add LLM metrics if available
        if 'llm' in results:
            for criterion, score in results['llm'][i].items():
                sample_data[f'llm_{criterion}'] = score
        
        data.append(sample_data)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    main()


