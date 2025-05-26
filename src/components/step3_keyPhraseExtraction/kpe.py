import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
from huggingface_hub import notebook_login
from openai import OpenAI
import os 
from src.components.step2_normalisation.norm import normalize_medical_data

# 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

def extractKPE(dialogue, note):

    dialogue, note = normalize_medical_data(dialogue, note)

    # Combine dialogue and note for context
    context = f"""
    Dialogue: {dialogue}\n
    Note: {note}\n\n
    Extract key phrases that capture the main features and important information from this medical conversation:
    Remember that the key phrases should be exact sentences from the dialogue (or 99% similarity paraphrases).
    Do NOT simply replicate the actual dialogue.
    You can use note only to get more context, but the final key phrases have to come only from dialogue.
    """

    # client = OpenAI(
    #     base_url="https://integrate.api.nvidia.com/v1",
    #     api_key="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"
    # )

    # # Generate key phrases using NVIDIA API
    # completion = client.chat.completions.create(
    #     model="google/gemma-2-27b-it",
    #     messages=[{"role": "user", "content": context}],
    #     temperature=0.2,
    #     top_p=0.7,
    #     max_tokens=1024,
    #     stream=True
    # )


    client = OpenAI(
      base_url = "https://integrate.api.nvidia.com/v1",
      api_key = "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"
    )

    completion = client.chat.completions.create(
      model="qwen/qwq-32b",
      messages=[{"role": "user", "content": context}],
      temperature=0.6,
      top_p=0.7,
      max_tokens=4096,
      stream=True
    )


    key_phrases = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            key_phrases += chunk.choices[0].delta.content

    return key_phrases

# Process each entry in the dataset
# Input: None (uses global dataset variable)
# Return: List of dictionaries containing encounter_id and key_phrases

def extract_after_think(text):
    if isinstance(text, str) and '</think>' in text:
        return text.split('</think>', 1)[1].strip()
    return text

def main_extractKPE():

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.environ["HUGGINGFACE_TOKEN"] = "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"
 
    
    dataset = load_dataset("ClinicianFOCUS/ACI-Bench-Refined", split="train")

    results = []
    # print(f'{dataset[:5]}')
    for idx, entry in enumerate(tqdm(dataset)):  
        # print(f'{entry}')
        dialogue = entry['dialogue']
        note = entry['note']

        try:
            key_phrases = extractKPE(dialogue, note)
            results.append({
                'encounter_id': entry['encounter_id'],
                'key_phrases': key_phrases
            })
        except Exception as e:
            print(f"Error processing entry {idx}: {str(e)}")
            continue
        # if idx ==4:
        #   break

    # Save results to CSV
    # Input: results list of dictionaries
    # Return: None (saves to file)
    df_results = pd.DataFrame(results)
    df_results.to_csv('key_phrases_results.csv', index=False)
    

    df = pd.read_csv('key_phrases_results.csv')  
    df['cleaned_key_phrases'] = df['key_phrases'].apply(extract_after_think)
    df.to_csv('key_phrases_results.csv', index=False)
    print("Results saved to key_phrases_results.csv")


# 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
# %%
