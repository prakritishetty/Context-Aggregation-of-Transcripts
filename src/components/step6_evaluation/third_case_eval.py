# 1. Install dependencies (once):
#    pip install datasets transformers sentence-transformers evaluate openai

import json
from datasets import load_dataset
from src.components.step6_evaluation.soap_note_evaluator import SOAPNoteEvaluator
import numpy as np
import pandas as pd
import os
import openai
import time

def third_case_evaluation():
    # 1. Load your CSV file
    csv_path = "results_ensemble/csv/combined_soap_summaries.csv"
    df = pd.read_csv(csv_path)

    # 2. Use the entire dataset (remove n and head)
    # n = len(df)  # Not needed, use len(df) directly

    # 3. Pull out the three fields:
    dialogues = df["dialogue"].tolist()  # Adjust if your dialogue column has a different name
    summary_notes = df["note"].tolist()
    augmented_notes = df["combined_soap_summary"].tolist()  # Adjust if your generated notes column has a different name

    # Get HF token from environment variable
    hf_token = "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"

    # 4. Initialize the evaluator
    evaluator = SOAPNoteEvaluator(
        dialogues=dialogues,
        generated_notes=augmented_notes,
        summary_notes=summary_notes,      # Use combined_soap_summary as summary notes
        openai_api_key="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",  # omit or set None to skip LLM judgments
        device="cpu",         # or "cpu" if you don't have a GPU
        hf_token=hf_token,    # pass the HF token
        max_retries=5,        # More retries
        retry_delay=10         # Longer delay between retries
    )

    # 5. Run every metric (dialogue-based and summary-based)
    results = evaluator.run_all()
    
    # 6. Convert results to a dataframe
    df_results = create_dataframe(results, len(df))
    
    # 7. Display the dataframe
    print(df_results)
    
    # 8. Save results to JSON file
    output_dir = "results_eval/first_case"
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results to JSON-serializable format
    json_results = {
        'raw_results': results,
        'dataframe_results': df_results.to_dict(orient='records')
    }
    
    with open(os.path.join(output_dir, "eval_first.json"), 'w') as f:
        json.dump(json_results, f, indent=4)
    
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
            # Clinical relevance metrics
            'bertscore_f1': results['clin_comb'][i]['bertscore_f1'],
            'embedding_similarity': results['clin_comb'][i]['embedding_similarity']
        }
        
        # Add LLM metrics if available
        if 'llm' in results:
            for attempt in range(evaluator.max_retries):
                try:
                    for criterion, score in results['llm'][i].items():
                        sample_data[f'llm_{criterion}'] = score
                    break  # Success!
                except openai.error.RateLimitError as e:
                    print(f"Rate limit hit, sleeping for {evaluator.retry_delay} seconds...")
                    time.sleep(evaluator.retry_delay)
                    evaluator.retry_delay *= 2  # Exponential backoff
                except Exception as e:
                    raise e  # Other errors
        
        data.append(sample_data)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    main()
