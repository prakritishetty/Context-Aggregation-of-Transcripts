# Key Phrase Extraction (KPE) Module

This module is responsible for extracting key phrases from medical dialogues and their corresponding notes. It uses advanced language models to identify and extract the most important information from medical conversations.

## Overview

The key phrase extraction process involves:
1. Normalizing the input dialogue and note
2. Using a language model to extract key phrases
3. Processing and cleaning the extracted phrases
4. Saving the results to a CSV file

## Main Components

### 1. `extractKPE(dialogue, note)`
- **Input**: Raw dialogue text and corresponding medical note
- **Process**:
  - Normalizes the input using the normalization module
  - Combines dialogue and note into a context for the language model
  - Uses the NVIDIA API with the Qwen/QWQ-32B model to generate key phrases
  - Returns the extracted key phrases as a string

### 2. `extract_after_think(text)`
- **Input**: Text that may contain a `<think>` tag
- **Process**: Removes any content before and including the `</think>` tag
- **Output**: Cleaned text containing only the key phrases

### 3. `main_extractKPE()`
- **Process**:
  - Loads the ClinicianFOCUS/ACI-Bench-Refined dataset
  - Processes each entry in the dataset
  - Extracts key phrases for each dialogue-note pair
  - Saves results to 'key_phrases_results.csv'
  - Cleans the results by removing any thinking process text
  - Updates the CSV file with cleaned results

## Dependencies
- torch
- datasets
- transformers
- pandas
- tqdm
- huggingface_hub
- openai

## Output
The module generates a CSV file ('key_phrases_results.csv') containing:
- encounter_id: Identifier for each medical encounter
- key_phrases: Extracted key phrases from the dialogue
- cleaned_key_phrases: Final cleaned version of the key phrases

## Usage
To use this module, simply call the `main_extractKPE()` function. It will process the entire dataset and save the results automatically.

## Notes
- The module uses GPU if available, falling back to CPU if not
- Error handling is implemented to continue processing even if individual entries fail
- The extraction process uses a temperature of 0.6 and top_p of 0.7 for balanced creativity and accuracy 