#%%
from google.colab import drive
drive.mount('/content/drive')

#%%
import pandas

df=pandas.read_csv('/content/drive/MyDrive/MTSdata/MTS-Dialog-TrainingSet.csv')

df.head()   

#%%
import torch

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())

# Check CUDA version
print("CUDA version:", torch.version.cuda)

# Get the name of the current GPU
if torch.cuda.is_available():
    print("GPU Device:", torch.cuda.get_device_name(0))


#%%
from vllm import LLM, SamplingParams
import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import re




#%%
def format_prompt(dialogue):
    return f"""
    Given this conversation:
    {dialogue}

    Task: Convert this dialogue into a clear, professional format by:
    1. Removing filler words (um, ah, la)
    2. Converting informal speech into clear, complete sentences
    3. Maintaining the core meaning and intent
    4. Using consistent speaker labels (Agent/Patient)
    5. Making the language more direct and professional
    
    Rules:
    - Keep the same turn-taking structure
    - Preserve all important medical information
    - Make the language more accessible
    - Remove hesitations and repetitions
    - Keep the same meaning but make it clearer

    Format each line as:
    Agent: [Clear professional statement]
    Patient: [Clear response]
    """
    
    
#%%

# print(f"Loading from: {model_path}")
monitor_gpu_memory("Before initialization")
# Initialize vLLM engine
# Calculate available GPU memory
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
total_memory = torch.cuda.get_device_properties(0).total_memory
allocated_memory = torch.cuda.memory_allocated()
free_memory = total_memory - allocated_memory

llm = LLM(
    model=model_name,
    trust_remote_code=True,
    dtype="float16",
    gpu_memory_utilization=0.9,
    tensor_parallel_size=1,
    max_num_batched_tokens=4096,  # Increased to match max_model_len
    max_num_seqs=64,
    max_model_len=4096,
)

monitor_gpu_memory("After initialization")

#%%
sampling_params = SamplingParams(
    temperature=0.3,  # Reduced temperature for more focused outputs
    max_tokens=512,   # Reduced as we don't need as many tokens
    top_p=0.9,
    stop=["</think>", "```", "\n\n\n"],  # Added extra newlines as stop token
)

#%%

# Function to process dialogues
def process_dialogues(df, llm, sampling_params):
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        prompt = format_prompt(row['dialogue'])
        
        try:
            outputs = llm.generate(prompt, sampling_params)
            normalized_dialogue = outputs[0].outputs[0].text.strip()
            results.append({
                'ID': row['ID'],
                'section_header': row['section_header'],
                'original_dialogue': row['dialogue'],
                'normalized_dialogue': normalized_dialogue
            })
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            results.append({
                'ID': row['ID'],
                'section_header': row['section_header'],
                'original_dialogue': row['dialogue'],
                'normalized_dialogue': 'ERROR'
            })
    
    return pd.DataFrame(results)

#%%
# Process the dialogues
print("Starting dialogue processing...")
results_df = process_dialogues(df, llm, sampling_params)

#%%
output_path = '/content/drive/MyDrive/MTSdata/normalized_dialogues.csv'
results_df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")

# Display sample results
print("\nSample results:")
print(results_df["normalized_dialogue"][0])



