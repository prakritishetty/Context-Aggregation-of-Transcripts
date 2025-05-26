import pandas as pd
from openai import OpenAI
import os
import re
from tqdm import tqdm

# Input and output file paths
INPUT_CSV = 'results_ensemble/csv/combined_soap_phrases.csv'
OUTPUT_CSV = 'results_ensemble/csv/combined_soap_summaries.csv'

# Initialize OpenAI client with NVIDIA endpoint
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

def extract_soap_sections(text):
    """
    Attempt to split the text into S, O, A, P sections using regex heuristics.
    Returns a dict with keys 'S', 'O', 'A', 'P'.
    """
    # Regex pattern to find S:, O:, A:, P: (case-insensitive, possibly with extra whitespace)
    pattern = re.compile(r'(S:|O:|A:|P:)', re.IGNORECASE)
    sections = {'S': '', 'O': '', 'A': '', 'P': ''}
    if not isinstance(text, str):
        return sections
    
    # Find all section headers and their positions
    matches = list(pattern.finditer(text))
    if not matches:
        # If no headers, return the whole text as S (subjective) by default
        sections['S'] = text.strip()
        return sections
    
    # Otherwise, split by found headers
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        section_key = match.group(1)[0].upper()  # 'S', 'O', 'A', or 'P'
        sections[section_key] = text[start:end].strip()
    return sections

def summarize_section(text, category, max_new_tokens=256):
    if not text.strip():
        return ''
    try:
        prompts = {
            'S': "Summarize the following subjective information (patient's reported symptoms and history) in third person perspective:\n",
            'O': "Summarize the following objective findings (physical examination and test results) in third person perspective:\n",
            'A': "Summarize the following assessment (diagnosis and clinical reasoning) in third person perspective:\n",
            'P': "Summarize the following plan (treatment and follow-up) in third person perspective:\n"
        }
        prompt = f"{prompts.get(category, 'Summarize the following in third person perspective:\n')}{text}\nSummary:"
        completion = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct-v0.2",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            top_p=1,
            max_tokens=max_new_tokens,
            stream=False
        )
        generated = completion.choices[0].message.content
        if 'Summary:' in generated:
            return generated.split('Summary:')[-1].strip()
        return generated.strip()
    except Exception as e:
        print(f"Error processing {category} section: {str(e)}")
        return ''

def main():
    try:
        df = pd.read_csv(INPUT_CSV)
        # For each row, extract S, O, A, P sections and summarize each
        summaries_S, summaries_O, summaries_A, summaries_P = [], [], [], []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Summarizing rows"):
            soap_sections = extract_soap_sections(row['combined_soap'])
            summaries_S.append(summarize_section(soap_sections['S'], 'S'))
            summaries_O.append(summarize_section(soap_sections['O'], 'O'))
            summaries_A.append(summarize_section(soap_sections['A'], 'A'))
            summaries_P.append(summarize_section(soap_sections['P'], 'P'))
        df['summary_S'] = summaries_S
        df['summary_O'] = summaries_O
        df['summary_A'] = summaries_A
        df['summary_P'] = summaries_P
        # Combine all summaries into one SOAP summary column
        df['combined_soap_summary'] = (
            'S: ' + df['summary_S'].fillna('') + '\n' +
            'O: ' + df['summary_O'].fillna('') + '\n' +
            'A: ' + df['summary_A'].fillna('') + '\n' +
            'P: ' + df['summary_P'].fillna('')
        )
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Summarized results saved to {OUTPUT_CSV}")
    except Exception as e:
        print(f"Error in main processing: {str(e)}")

if __name__ == "__main__":
    main()

# Previous implementation using transformers pipeline
"""
from transformers import pipeline

MODEL_NAME = 'mistralai/Mistral-7B-v0.1'  # Open-access base model

def summarize_third_person(text, summarizer, max_new_tokens=256):
    if not isinstance(text, str) or not text.strip():
        return ''
    prompt = f"Summarize the following in third person perspective, not as doctor or patient:\n{text}\nSummary:"
    result = summarizer(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    # Extract the summary after 'Summary:' if present
    generated = result[0]['generated_text']
    if 'Summary:' in generated:
        return generated.split('Summary:')[-1].strip()
    return generated.strip()

def main():
    # Load the combined SOAP phrases
    df = pd.read_csv(INPUT_CSV)

    # Initialize the text-generation pipeline with Mistral 7B
    summarizer = pipeline(
        "text-generation",
        model=MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
    )

    # Summarize each combined_soap entry
    df['summary_third_person'] = df['combined_soap'].apply(lambda x: summarize_third_person(x, summarizer))

    # Save the results
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Summarized results saved to {OUTPUT_CSV}")
"""
