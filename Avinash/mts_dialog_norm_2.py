#%%
from transformers import pipeline
import pandas as pd
import re
from datetime import datetime, timedelta
import os

# Initialize text normalization model (example using Hugging Face)
normalizer = pipeline("text2text-generation", 
                     model="mrm8488/t5-base-finetuned-common_gen")  # Replace with Liu et al.'s model

def normalize_text(utterance):
    # Apply text normalization
    normalized = normalizer(f"normalize: {utterance}", max_length=512)[0]['generated_text']
    
    # Grammar correction
    corrected = normalizer(f"grammar: {normalized}", max_length=512)[0]['generated_text']
    return corrected

#%%
def reduce_spoken_features(text):
    # Remove fillers and repetitions
    text = re.sub(r'\b(um|uh|er|ah|like|you know)\b', '', text, flags=re.IGNORECASE)
    
    # Handle pauses and interruptions
    text = re.sub(r'(\.\.\.|--|―)', '', text)  # Remove ellipses and dashes[4]
    
    # Reduce repeated phrases
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)  # Remove immediate repetitions
    return text.strip()

#%%
def parse_entry(entry):
    """
    Parse a dialogue entry from the CSV format.
    Expected format: [speaker, text, start_time, end_time]
    """
    if isinstance(entry, tuple) and len(entry) == 4:
        speaker, text, start, end = entry
        return speaker, text, start, end
    else:
        # Handle different formats or provide default values
        return "Unknown", str(entry), "00:00:00", "00:00:00"

def format_output(speaker, text, start_time, end_time):
    """
    Format the output in a consistent way.
    """
    return (speaker, text, start_time, end_time)

def merge_utterances(transcript):
    merged = []
    current_speaker = None
    current_text = []
    start_time = end_time = None

    for entry in transcript:
        speaker, text, start, end = parse_entry(entry)
        
        if speaker == current_speaker:
            current_text.append(text)
            end_time = end
        else:
            if current_speaker:
                merged.append(format_output(current_speaker, 
                                          ' '.join(current_text),
                                          start_time,
                                          end_time))
            current_speaker = speaker
            current_text = [text]
            start_time = start
    
    # Add last entry
    if current_speaker:
        merged.append(format_output(current_speaker,
                                  ' '.join(current_text),
                                  start_time,
                                  end_time))
    return merged

#%%
def process_transcript(raw_transcript):
    """
    Process a transcript from the CSV format.
    raw_transcript: List of tuples (speaker, text, start_time, end_time)
    """
    processed = []
    
    # Extract components
    speakers = [entry[0] for entry in raw_transcript]
    utterances = [entry[1] for entry in raw_transcript]
    timestamps = [(entry[2], entry[3]) for entry in raw_transcript]
    
    # Step 1: Text normalization
    normalized = [normalize_text(utterance) for utterance in utterances]
    
    # Step 2: Spoken feature reduction
    cleaned = [reduce_spoken_features(text) for text in normalized]
    
    # Step 3: Utterance consolidation
    merged = merge_utterances(zip(speakers, cleaned, [t[0] for t in timestamps], [t[1] for t in timestamps]))
    
    return merged

#%%
def calculate_duration(start, end):
    fmt = "%H:%M:%S.%f"
    return datetime.strptime(end, fmt) - datetime.strptime(start, fmt)

#%%
def process_csv_dialogues(csv_path, output_path=None):
    """
    Process dialogues from a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        output_path: Path to save the processed dialogues (optional)
    
    Returns:
        DataFrame with processed dialogues
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    results = []
    
    for idx, row in df.iterrows():
        # Extract dialogue from the row
        dialogue = row['dialogue']
        
        # Split the dialogue into turns (assuming format like "Speaker: Text")
        turns = re.split(r'\n', dialogue)
        transcript = []
        
        for turn in turns:
            if not turn.strip():
                continue
                
            # Extract speaker and text
            match = re.match(r'([^:]+):\s*(.+)', turn)
            if match:
                speaker, text = match.groups()
                # For simplicity, we'll use dummy timestamps
                start_time = "00:00:00.000"
                end_time = "00:00:05.000"
                transcript.append((speaker.strip(), text.strip(), start_time, end_time))
        
        # Process the transcript
        processed = process_transcript(transcript)
        
        # Format the processed transcript back to a string
        processed_dialogue = "\n".join([f"{speaker}: {text}" for speaker, text, _, _ in processed])
        
        results.append({
            'ID': row['ID'],
            'section_header': row['section_header'],
            'original_dialogue': dialogue,
            'normalized_dialogue': processed_dialogue
        })
    
    # Create a DataFrame with the results
    results_df = pd.DataFrame(results)
    
    # Save to CSV if output path is provided
    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    
    return results_df

#%%
# Example usage
if __name__ == "__main__":
    # Path to the CSV file
    csv_path = "MTS-Dialog-TrainingSet.csv"
    
    # Process the dialogues
    results_df = process_csv_dialogues(csv_path, "normalized_dialogues.csv")
    
    # Display sample results
    print("\nSample results:")
    print(f'Results: {results_df["normalized_dialogue"][0]}')
    df= pd.read_csv(csv_path)
    print(f'Original: {df["dialogue"][0]}')

#%%
# Validate normalization improvements
assert normalize_text("gimme da file") == "Please provide the document"

# Test feature reduction
assert reduce_spoken_features("Um... I need-- need the report") == "I need the report"

# Verify utterance merging
test_input = [('DrA', 'Hello', '00:00:00', '00:00:02'),
              ('DrA', 'How are you?', '00:00:03', '00:00:05')]
assert len(merge_utterances(test_input)) == 1




