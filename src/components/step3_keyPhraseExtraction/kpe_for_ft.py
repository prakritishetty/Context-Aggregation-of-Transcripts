import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import pandas as pd
from tqdm import tqdm
from huggingface_hub import login
import os
from dotenv import load_dotenv
# from src.components.step2_normalisation.norm import normalize_medical_data

# Load environment variables
load_dotenv()

# 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

def extractKPE(dialogue, note):
    dialogue, note = normalize_medical_data(dialogue, note)

    # Combine dialogue and note for context
    context = f"""
    Dialogue: {dialogue}\n
    Note: {note}\n\n
    Generate a concise medical note that captures the key clinical information from this doctor-patient interaction.
    Focus on extracting the most important medical details, symptoms, diagnoses, and treatment plans.
    The note should be clear, professional, and follow standard medical documentation practices.
    """

    # Initialize the model and tokenizer with optimizations
    model_name = "ClinicianFOCUS/Clinician-Note-2.0a"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Configure quantization using BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )

    # Load model with quantization config
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    # Create text generation pipeline with optimized settings
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=0.3,
        top_p=0.7,
        max_length=8192,
        device_map="auto",
        batch_size=1,
        truncation=True  # Explicitly enable truncation
    )

    # Generate the medical note
    generated_text = generator(context, do_sample=True)[0]['generated_text']
    print(generated_text)
    key_phrases = generated_text[len(context):].strip()

    # Clear CUDA cache after each generation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return key_phrases

# Process each entry in the dataset
# Input: None (uses global dataset variable)
# Return: List of dictionaries containing encounter_id and key_phrases

def extract_after_think(text):
    if isinstance(text, str) and '</think>' in text:
        return text.split('</think>', 1)[1].strip()
    return text

def main_extractKPE_for_ft():
    # Get Hugging Face token from environment variable
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError(
            "Hugging Face token not found. Please set the HF_TOKEN environment variable.\n"
            "You can get your token from https://huggingface.co/settings/tokens\n"
            "Then set it using: export HF_TOKEN=your_token_here"
        )

    try:
        # Login to Hugging Face Hub
        login(token=hf_token)

        # Load dataset with authentication
        dataset = load_dataset(
            "ClinicianFOCUS/ACI-Bench-Refined",
            split="train",
            token=hf_token
        )

        results = []
        for idx, entry in enumerate(tqdm(dataset)):
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

        # Save results to CSV
        df_results = pd.DataFrame(results)
        df_results.to_csv('key_phrases_results_for_ft.csv', index=False)

        df = pd.read_csv('key_phrases_results.csv')
        df['cleaned_key_phrases'] = df['key_phrases'].apply(extract_after_think)
        df.to_csv('key_phrases_results.csv', index=False)
        print("Results saved to key_phrases_results.csv")

    except Exception as e:
        print(f"Authentication error: {str(e)}")
        print("\nPlease make sure you have:")
        print("1. A valid Hugging Face token from https://huggingface.co/settings/tokens")
        print("2. Set the token in your environment: export HF_TOKEN=your_token_here")
        print("3. Have the necessary permissions to access the dataset")
        raise

if __name__ == "__main__":
    main_extractKPE_for_ft()