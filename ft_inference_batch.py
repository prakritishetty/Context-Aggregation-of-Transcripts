from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import os
from tqdm import tqdm

prompt_header = """
**Role:** You are an AI assistant specialized in generating clinical SOAP notes.

**Task:** Generate a concise, accurate, and clinically relevant SOAP note based **STRICTLY AND SOLELY** on the provided doctor-patient interaction transcript.

**CRITICAL INSTRUCTIONS:**

1.  **Strict Transcript Adherence:** Generate the SOAP note using **ONLY** information **explicitly stated** within the provided transcript.
2.  **NO Assumptions or External Knowledge:** **DO NOT** infer information, add details not mentioned (even if clinically likely), make assumptions, or use external medical knowledge. Adherence to the transcript is paramount.
3.  **Standard SOAP Structure:** Organize the output clearly into the following sections using **EXACTLY** these headings:
    *   **S – Subjective**
    *   **O – Objective**
    *   **A – Assessment**
    *   **P – Plan**
4.  **NO Extraneous Text:** The output must contain **ONLY** the four section headings (S, O, A, P) and the corresponding content derived *directly* from the transcript. **DO NOT** include introductory sentences (e.g., "Here is the SOAP note:"), concluding remarks, disclaimers, notes about the generation process, metadata, or *any* other text before, between, or after the S/O/A/P sections.

**Formatting:**

*   Use clear headings for each SOAP section (as listed above).
*   Be concise but ensure all relevant details *from the transcript* are included under the correct heading.
*   Use standard medical abbreviations only if they are unambiguous and directly supported by the transcript's terminology.

**Input:** You will receive a doctor-patient transcript.

**Output:** Generate **ONLY** the structured SOAP note (S/O/A/P sections and content) based on the critical instructions above.
"""

def generate_soap_note(transcript, tokenizer, model):
    """Generate a SOAP note for a given transcript."""
    prompt = prompt_header + "\n\nTranscript: " + transcript
    
    # Prepare the model input
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Generate the response
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32768,
            temperature=0.3,
            top_p=0.9,
            top_k=20,
        )
    
    # Parse the response
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    try:
        # Find the end of thinking content
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    
    # Extract the actual content
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
    return content

def main():
    # Create output directory
    output_dir = "results_ensemble/ft_inference"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model and tokenizer
    model_name = "ClinicianFOCUS/Clinician-Note-2.0a"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load the dataset
    print("Loading dataset...")
    dataset = load_dataset("ClinicianFOCUS/ACI-Bench-Refined", split="train")
    
    # Initialize or load existing results
    output_path = os.path.join(output_dir, "ft_results.csv")
    try:
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            results_df = pd.read_csv(output_path)
            processed_indices = set(range(len(results_df)))
        else:
            results_df = pd.DataFrame(columns=['dialogue', 'note', 'augmented_note', 'generated_soap'])
            processed_indices = set()
    except Exception as e:
        print(f"Error reading existing results file: {str(e)}")
        print("Creating new results file...")
        results_df = pd.DataFrame(columns=['dialogue', 'note', 'augmented_note', 'generated_soap'])
        processed_indices = set()
    
    # Process each transcript
    for idx, sample in enumerate(tqdm(dataset, desc="Processing transcripts")):
        # Skip if already processed
        if idx in processed_indices:
            continue
            
        try:
            dialogue = sample['dialogue']
            note = sample['note']
            augmented_note = sample['augmented note']
            
            # Generate SOAP note
            soap_note = generate_soap_note(dialogue, tokenizer, model)
            
            # Create new row
            new_row = pd.DataFrame({
                'dialogue': [dialogue],
                'note': [note],
                'augmented_note': [augmented_note],
                'generated_soap': [soap_note]
            })
            
            # Append to existing DataFrame
            results_df = pd.concat([results_df, new_row], ignore_index=True)
            
            # Save after each entry
            try:
                results_df.to_csv(output_path, index=False)
                print(f"Saved results for sample {idx}")
            except Exception as e:
                print(f"Error saving results for sample {idx}: {str(e)}")
                # Try to save to a backup file
                backup_path = f"{output_path}.backup"
                results_df.to_csv(backup_path, index=False)
                print(f"Saved backup to {backup_path}")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            continue
    
    print(f"All results saved to {output_path}")

if __name__ == "__main__":
    main() 