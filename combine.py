import pandas as pd
import glob
import os
from datasets import load_dataset
import csv

def combine_soap_phrases():
    # Load ACI benchmark dataset
    ds_all = load_dataset("ClinicianFOCUS/ACI-Bench-Refined", split="train")
    
    # Get all SOAP categorized phrases CSV files
    csv_files = glob.glob('results_ensemble/csv/soap_categorized_phrases_phrase_set_*.csv')
    
    # Create a dictionary to store dataframes by file number
    dfs_dict = {}
    
    # Read each CSV file and store in dictionary with file number as key
    for file in sorted(csv_files):
        try:
            # Extract file number from filename
            file_num = int(os.path.basename(file).split('_')[-1].split('.')[0])
            # Read CSV with proper quoting to handle commas in fields
            df = pd.read_csv(file, quoting=csv.QUOTE_ALL, escapechar='\\')
            df['source_file'] = os.path.basename(file)
            dfs_dict[file_num] = df
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue

    # Get the maximum file number
    max_file_num = max(dfs_dict.keys()) if dfs_dict else 0
    
    # Initialize result rows
    result_rows = []
    
    # Process each file number in sequence
    for i in range(max_file_num + 1):
        # Get the first two columns from the ACI benchmark dataset
        aci_row = ds_all[i]
        first_two_cols = {
            'dialogue': aci_row['dialogue'],
            'note': aci_row['note']
        }
        
        # If file exists, process it; otherwise use empty string
        if i in dfs_dict:
            group = dfs_dict[i]
            soap_phrases = []
            for category in ['S', 'O', 'A', 'P']:
                category_phrases = group[group['soap_category'] == category]['phrase'].tolist()
                if category_phrases:
                    soap_phrases.append(f"{category}: {' '.join(category_phrases)}")
            combined_soap = ' | '.join(soap_phrases)
        else:
            combined_soap = ""
        
        # Create a row with first two columns and combined SOAP phrases
        row = pd.Series({
            'dialogue': first_two_cols['dialogue'],
            'note': first_two_cols['note'],
            'combined_soap': combined_soap
        })
        result_rows.append(row)

    # Create the final dataframe
    result_df = pd.DataFrame(result_rows)

    # Save the combined results
    output_file = 'results_ensemble/csv/combined_soap_phrases.csv'
    result_df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)

    print(f"Combined results saved to {output_file}")
    print("\nSummary of combined phrases:")
    print(result_df)

if __name__ == "__main__":
    combine_soap_phrases()
