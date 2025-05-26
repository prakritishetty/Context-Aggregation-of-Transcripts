from datasets import load_dataset
import pandas as pd
import os
import json
import requests
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login

class SOAPNoteGenerator:
    def __init__(
        self,
        hf_token: str = "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",
        model_name: str = "ClinicianFOCUS/Clinician-Note-2.0a",
        output_folder: str = "results_ft",
        output_filename: str = "soap_notes_output_.csv",
        x_title: str = "SOAP Note Generator"
    ):
        # Login to Hugging Face
        login(token=hf_token)
        
        self.model_name = model_name
        self.output_folder = output_folder
        self.output_filename = output_filename
        self.x_title = x_title
        
        # Initialize the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token
        )
        print(f'Model loaded: {self.model}')

    def format_input(self, dialogue: str, notes: Optional[str] = None) -> str:
        """Format the input prompt for the model."""
        prompt = f"""
                You are a clinical documentation assistant. Generate a SOAP note from the medical dialogue below.

                Follow this structure and guidance:

                ---

                **SOAP Note Format:**

                **S (Subjective):**  
                Include what the client reports about their symptoms and medical history, including the chief complaint, history of present illness, relevant past medical/surgical/family/social history, medications, allergies, and review of systems. This section captures the patient's perspective.

                **O (Objective):**  
                Include measurable or observed findings from the session: vital signs, physical exam results, lab tests, imaging findings, and observed appearance or behavior. This section captures clinical data collected during the encounter.

                **A (Assessment):**  
                Interpret and assess the situation based on subjective and objective data. List diagnoses, clinical impressions, and your reasoning. Summarize key medical problems and how they interrelate.

                **P (Plan):**  
                Outline the treatment and management plan, including tests, medications, referrals, therapies, lifestyle modifications, patient education, short- and long-term goals, and follow-up plans.

                ---

                Reply with the SOAP note only, using the format above. Do not include explanations, commentary, or any other text.

                Dialogue:
                {dialogue}
                """
        if notes:
            prompt += f"\n\nAdditional Doctor Notes:\n{notes}\n"
        print(f'Prompt: {prompt.strip()}')

        return prompt.strip()

    def generate_soap_note(self, dialogue: str, notes: Optional[str] = None) -> str:
        """Generate a SOAP note for a given dialogue."""
        prompt = self.format_input(dialogue, notes)
        
        # Tokenize the input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate the response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        # Decode and return the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f'Response: {response}')
        return response.replace(prompt, "").strip()

    def process_dataset(self, dataset_name: str = "har1/MTS_Dialogue-Clinical_Note", split: str = "train") -> pd.DataFrame:
        """Process the dataset and generate SOAP notes."""
        print("Loading dataset...")
        dataset = load_dataset(dataset_name, split=split)
        
        results = []
        for i, sample in enumerate(dataset):
            print(f"Generating SOAP note for sample {i+1}...")
            dialogue = sample.get("dialogue", "")
            notes = sample.get("section_text", "")
            
            soap_note = self.generate_soap_note(dialogue, notes)
            
            results.append({
                "dialogue": dialogue,
                "doctor_notes": notes,
                "generated_soap_note": soap_note
            })
        
        return pd.DataFrame(results)

    def save_results(self, df: pd.DataFrame) -> None:
        """Save the results to a CSV file."""
        os.makedirs(self.output_folder, exist_ok=True)
        output_path = os.path.join(self.output_folder, self.output_filename)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

def main():
    # Initialize the SOAP note generator
    generator = SOAPNoteGenerator(
        hf_token="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",
        model_name="ClinicianFOCUS/Clinician-Note-2.0a",
        output_folder="/work/pi_hzamani_umass_edu/psshetty_umass_edu/Context-Aggregation-of-Transcripts/results_ft",
        output_filename="soap_notes_output_baseline.csv",
        x_title="SOAP Note Generator"
    )
    
    # Process the dataset
    results_df = generator.process_dataset()
    print(results_df)
    # Save the results
    generator.save_results(results_df)

if __name__ == "__main__":
    main()
