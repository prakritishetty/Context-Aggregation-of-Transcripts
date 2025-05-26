# Text Normalization Module

This module is responsible for cleaning and normalizing medical dialogue text and clinical notes. It removes filler words, cleans up speech patterns, and preserves important medical terminology while improving text consistency.

## Overview

The normalization process involves:
1. Removing common filler words and speech patterns
2. Cleaning up spacing and formatting
3. Preserving medical terminology and measurements
4. Applying different normalization rules for dialogue vs. clinical notes

## Main Components

### 1. `normalize_text(text)`
- **Input**: Raw text (typically dialogue)
- **Process**:
  - Removes common filler words and expressions (e.g., "um", "uh", "like", "you know")
  - Cleans up repetitive stutters and word repetitions
  - Standardizes spacing and newlines
  - Capitalizes sentences
  - Fixes spacing in medical measurements (e.g., "10 mg" → "10mg")
  - Preserves medical terminology while cleaning up formatting

### 2. `normalize_medical_data(dialogue, note)`
- **Input**: 
  - dialogue: Medical conversation text
  - note: Clinical note text
- **Process**:
  - Applies more aggressive cleaning to dialogue text
  - Applies minimal cleaning to clinical notes to preserve medical formatting
  - Returns both normalized texts as a tuple

## Normalization Rules

### Dialogue Cleaning
- Removes filler words and expressions
- Cleans up speech patterns and stutters
- Standardizes spacing and punctuation
- Preserves medical measurements and terminology

### Clinical Note Cleaning
- Minimal cleaning to preserve medical formatting
- Standardizes spacing and newlines
- Preserves all medical terminology and measurements

## Examples of Normalization

### Input Dialogue:
```
"Um, like, the patient has been taking, you know, 10 mg of medication, um, twice daily."
```

### Normalized Output:
```
"The patient has been taking 10mg of medication twice daily."
```

## Usage
```python
from norm import normalize_medical_data

# Normalize both dialogue and note
normalized_dialogue, normalized_note = normalize_medical_data(dialogue_text, note_text)

# Or normalize just text
from norm import normalize_text
cleaned_text = normalize_text(input_text)
```

## Notes
- The module preserves important medical terminology while removing conversational elements
- Different cleaning rules are applied to dialogue vs. clinical notes
- Medical measurements and numbers are preserved and standardized
- The normalization process is designed to improve text consistency while maintaining medical accuracy 