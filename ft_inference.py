from datasets import load_dataset
import pandas as pd
import os
import json
import requests
from typing import Dict, List, Optional

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
            token=hf_tokens
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

    # def generate_soap_note(self, dialogue: str, notes: Optional[str] = None) -> str:
    #     """Generate a SOAP note for a given dialogue."""
    #     prompt = self.format_input(dialogue, notes)
        
    #     headers = {
    #         "Authorization": f"Bearer {self.api_key}",
    #         "Content-Type": "application/json",
    #         "HTTP-Referer": self.http_referer,
    #         "X-Title": self.x_title
    #     }
        
    #     data = {
    #         "model": self.model_name,
    #         "messages": [
    #             {
    #                 "role": "user",
    #                 "content": prompt
    #             }
    #         ]
    #     }

    #     try:
    #         response = requests.post(
    #             "https://openrouter.ai/api/v1/chat/completions",
    #             headers=headers,
    #             data=json.dumps(data)
    #         )

    #         if response.status_code == 200:
    #             try:
    #                 print(str(response.json()))
    #                 return ""
    #                 # return response.json()["choices"][0]["message"]["content"].strip()
    #             except KeyError:
    #                 print("ERROR: Unexpected response structure")
    #                 return ""
    #         else:
    #             print(f"ERROR {response.status_code}: {response.text}")
    #             return ""
                
    #     except Exception as e:
    #         print(f"Error generating SOAP note: {str(e)}")
    #         return ""

    def process_dataset(self, dataset_name: str = "har1/MTS_Dialogue-Clinical_Note", split: str = "train") -> pd.DataFrame:
        """Process the dataset and generate SOAP notes."""
        try:
            print("Loading dataset...")
            dataset = load_dataset(dataset_name, split=split)
            
            results = []
            for i, sample in enumerate(dataset):
                try:
                    print(f"Generating SOAP note for sample {i+1}...")
                    dialogue = sample.get("dialogue", "")
                    notes = sample.get("section_text", "")
                    
                    soap_note = self.generate_soap_note(dialogue, notes)
                    
                    results.append({
                        "dialogue": dialogue,
                        "doctor_notes": notes,
                        "generated_soap_note": soap_note
                    })
                except Exception as e:
                    print(f"Error processing sample {i+1}: {str(e)}")
                    continue
            
            return pd.DataFrame(results)
        except Exception as e:
            print(f"Error loading or processing dataset: {str(e)}")
            return pd.DataFrame()

    def save_results(self, df: pd.DataFrame) -> None:
        """Save the results to a CSV file."""
        try:
            os.makedirs(self.output_folder, exist_ok=True)
            output_path = os.path.join(self.output_folder, self.output_filename)
            df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")

def main():
    try:
        # Initialize the SOAP note generator
        generator = SOAPNoteGenerator(
            api_key="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",
            model_name="qwen/qwen3-0.6b-04-28:free",
            output_folder="/work/pi_hzamani_umass_edu/psshetty_umass_edu/Context-Aggregation-of-Transcripts/results_ft",
            output_filename="soap_notes_output_baseline.csv",
            x_title="SOAP Note Generator"
        )
        
        # Process the dataset
        results_df = generator.process_dataset()
        if not results_df.empty:
            print(results_df)
            # Save the results
            generator.save_results(results_df)
        else:
            print("No results to save - dataset processing failed")
    except Exception as e:
        print(f"An error occurred in main: {str(e)}")

if __name__ == "__main__":
    main()
