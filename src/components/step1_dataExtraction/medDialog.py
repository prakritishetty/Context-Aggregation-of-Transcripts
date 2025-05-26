from datasets import load_dataset
import pandas as pd

class MedDialog:

    def process_utterances(utterance_list):
        patient_utterances = [u.split(": ", 1)[1] for u in utterance_list if u.lower().startswith("patient")]
        cleaned_utterances = [u.split(": ", 1)[1] if ": " in u else u for u in utterance_list]
        return patient_utterances[0] if patient_utterances else None, cleaned_utterances

    def extractMedDialog():

        dataset = load_dataset("UCSD26/medical_dialog", "processed.en")

        df_train = pd.DataFrame(dataset['train'])
        df_val = pd.DataFrame(dataset['validation'])
        df_test = pd.DataFrame(dataset['test'])

        df_train["patient_utterance"], df_train["cleaned_utterances"] = zip(*df_train["utterances"].apply(process_utterances))
        df_train = df_train.drop(['description','utterances'], axis=1)
        df_train = df_train.rename(columns={
            "patient_utterance": "patient_text",
            "cleaned_utterances": "text"
        })