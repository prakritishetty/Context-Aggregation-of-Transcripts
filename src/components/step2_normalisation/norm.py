import re

def normalize_text(text):
    """
    Normalize medical dialogue text by removing filler words and cleaning up common speech patterns.
    
    Args:
        text (str): Input dialogue text
    Returns:
        str: Cleaned dialogue text
    """
    # Common filler words and expressions to remove
    filler_words = [
        r'\b(um|uh|uhh|hmm|err|like|you know|i mean|basically|actually|literally|sort of|kind of)\b',
        r'\b(yeah|yep|nope|okay|ok|right|so|well|now|then|anyway|anyways)\b',
        r'\b(i guess|i think|i suppose|you see|you get|gonna|wanna|gotta)\b',
        r'\b(its-+its|thats-+thats|hes-+hes|shes-+shes)\b',  # Repetitive stutters
        r'(\w+)-\1',  # General pattern for word-word repetitions
        r'\s+',  # Multiple spaces
    ]
    
    # Convert to lowercase for consistent processing
    text = text.lower()
    
    # Remove filler patterns
    for pattern in filler_words:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    
    # Clean up spacing
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'\s*\n+\s*', '\n', text)  # Clean up newlines
    
    # Capitalize sentences
    text = '. '.join(s.strip().capitalize() for s in text.split('.') if s.strip())
    
    # Additional medical-specific cleanup
    text = re.sub(r'(\d+)\s*/\s*(\d+)', r'\1/\2', text)  # Fix spacing in fractions/measurements
    text = re.sub(r'(\d+)\s+mg', r'\1mg', text)  # Fix spacing in measurements
    text = re.sub(r'(\d+)\s+ml', r'\1ml', text)  # Fix spacing in measurements
    
    # Final cleanup
    text = text.strip()
    text = re.sub(r'\s+\.', '.', text)  # Remove spaces before periods
    text = re.sub(r'\s+,', ',', text)  # Remove spaces before commas
    
    return text

def normalize_medical_data(dialogue, note):
    """
    Normalize both dialogue and clinical note while preserving medical terminology.
    
    Args:
        dialogue (str): Medical dialogue text
        note (str): Clinical note text
    Returns:
        tuple: (normalized_dialogue, normalized_note)
    """
    # Normalize dialogue with more aggressive cleaning
    normalized_dialogue = normalize_text(dialogue)
    
    # Normalize note with less aggressive cleaning (preserve medical formatting)
    normalized_note = note.strip()
    normalized_note = re.sub(r'\s+', ' ', normalized_note)
    normalized_note = re.sub(r'\s*\n+\s*', '\n', normalized_note)
    
    return normalized_dialogue, normalized_note
