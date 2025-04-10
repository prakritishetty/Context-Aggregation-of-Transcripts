import spacy
import pandas as pd

class clean_transcripts:
    def __init__(self, transcripts, min_length=50):
        self.transcripts = transcripts
        self.cleaned_transcripts = []
        self.nlp = spacy.load("en_core_web_sm")
    def remove_short_entries(self, min_length=50):
        """
        Remove entries that are too short.
        """
        for entry in self.transcripts:
            sentence = entry.split()[-1]
            doc = self.nlp(sentence)
            if len(doc) >= min_length:
                self.cleaned_transcripts.append(entry)
        print(f"Number of Raw Transcripts: {len(self.transcripts)}, Number of Cleaned Transcripts: {len(self.cleaned_transcripts)}")
        return self.cleaned_transcripts

if __name__ == "__main__":
    transcripts = pd.read_csv("MTS-Dialog-TrainingSet").tolist()
    cleaner = clean_transcripts(transcripts)
    cleaned = cleaner.remove_short_entries(min_length=20)
    print(cleaned)