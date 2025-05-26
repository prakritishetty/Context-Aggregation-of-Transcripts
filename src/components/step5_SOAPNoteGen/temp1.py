import pandas as pd
from collections import defaultdict
from transformers import pipeline

# Placeholder mapping: update as needed to match your actual cluster-to-SOAP mapping
# Example: {0: 'S', 1: 'O', 2: 'A', 3: 'P'}
cluster_to_soap = {
    0: 'S',
    1: 'O',
    2: 'A',
    3: 'P',
}

soap_headings = {
    'S': 'Subjective',
    'O': 'Objective',
    'A': 'Assessment',
    'P': 'Plan',
}

def summarize_with_hf(phrases, section, summarizer=None, max_input_tokens=1024):
    """
    Summarize a list of phrases into a concise paragraph for the given SOAP section using HuggingFace Transformers.
    """
    if not phrases:
        return ""
    text = ' '.join(phrases)
    # Truncate if too long for the model
    if len(text.split()) > max_input_tokens:
        text = ' '.join(text.split()[:max_input_tokens])
    prompt = f"Summarize the following medical phrases into a concise, coherent paragraph suitable for the {section} section of a SOAP note. Do not simply list the phrases; synthesize them into a readable summary.\n\nPhrases:\n{text}"
    result = summarizer(prompt, max_length=256, min_length=30, do_sample=False)
    return result[0]['summary_text'].strip()

def generate_soap_note_from_clusters(csv_path, output_path, model_name="facebook/bart-large-cnn"):
    df = pd.read_csv(csv_path)
    # Group phrases by cluster
    clusters = defaultdict(list)
    for _, row in df.iterrows():
        cluster = row['cluster_labels']
        phrase = row['phrases']
        clusters[cluster].append(phrase)

    # Assign clusters to SOAP headings
    soap_sections = defaultdict(list)
    for cluster, phrases in clusters.items():
        soap = cluster_to_soap.get(cluster, 'Other')
        soap_sections[soap].extend(phrases)

    # Summarize (here: just join) phrases for each section
    summarizer = pipeline("summarization", model=model_name)
    soap_note = {}
    for soap, phrases in soap_sections.items():
        heading = soap_headings.get(soap, soap)
        summary = summarize_with_hf(phrases, heading, summarizer=summarizer)
        soap_note[heading] = summary

    # Save to CSV (one row, columns: Subjective, Objective, Assessment, Plan)
    note_df = pd.DataFrame([soap_note])
    note_df.to_csv(output_path, index=False)
    print(f"SOAP note saved to {output_path}")

if __name__ == "__main__":
    generate_soap_note_from_clusters(
        csv_path="clustering_results.csv",
        output_path="generated_soap_note.csv",
        model_name="facebook/bart-large-cnn"  # You can change this to any summarization model you prefer
    )
