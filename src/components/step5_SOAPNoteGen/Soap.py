import pandas as pd
from transformers import pipeline
import os

def load_soap_data(csv_path):
    """Load and process SOAP data from CSV."""
    df = pd.read_csv(csv_path)
    
    # Group by SOAP category and join phrases
    soap_groups = df.groupby('soap_category')['phrase'].agg(lambda x: ' '.join(x)).reset_index()
    
    # Rename columns for clarity
    soap_groups.columns = ['category', 'phrases']
    
    return soap_groups

def generate_summaries(df):
    """Generate summaries using a small LLM in third person perspective."""
    # Initialize the summarization pipeline with a small model
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    summaries = []
    for text in df['phrases']:
        # Truncate text if too long (BART has a max input length)
        if len(text) > 1024:
            text = text[:1024]
        
        # Add instruction for third person perspective
        prompt = f"Summarize the following in third person perspective: {text}"
        
        # Generate summary
        summary = summarizer(prompt, max_length=130, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    
    df['summary'] = summaries
    return df

def main_soap():
    # Path to the CSV file
    csv_path = "soap_notes.csv"
    
    # Load and process data
    df = load_soap_data(csv_path)
    
    # Generate summaries
    df = generate_summaries(df)
    
    # Save the results
    output_path = "results_soap/soap_processed_3.csv"
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    return df

if __name__ == "__main__":
    main_soap()
