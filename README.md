# Hierarchical Context Aggregation for Quality Automated Medical Notes

## Overview
This project presents a lightweight, modular NLP pipeline for generating clinically accurate and structurally coherent SOAP-format medical notes from raw doctor-patient dialogue transcripts. Unlike traditional LLM-only solutions, our approach integrates domain-specific embeddings, hierarchical clustering, and structured post-processing to reduce hallucinations and improve alignment with clinical documentation standards.

## Motivation
Manual clinical note-taking is time-consuming and error-prone, leading to reduced physician efficiency and potential inconsistencies in medical records. While LLMs can automate this process, they often struggle with factual grounding, structured formatting, and interpretability. This project proposes a novel solution to address these limitations using open-source, instruction-tuned models.

## Key Features
- **End-to-End Pipeline**: Automates note generation from transcript normalization to structured SOAP note output.
- **Hybrid Framework**: Combines key phrase extraction, BiomedBERT-based hierarchical clustering, and ensemble SOAP classification for discourse-aware structuring.
- **LLM-agnostic**: Works with baseline and fine-tuned LLMs (e.g., Qwen 3) without requiring domain-specific retraining.
- **Modular Components**:
  - Text normalization
  - Key phrase extraction (LLM-prompted)
  - Sentence embedding (BiomedBERT)
  - Hierarchical + k-means clustering
  - SOAP classification using ensemble of embedding match, topic modeling, and zero-shot BART-MNLI

## Architecture
Raw Transcript → Text Normalization → Key Phrase Extraction →
Embedding (BiomedBERT) → Hierarchical Clustering + Subclustering →
SOAP Section Classification → Final Note Assembly


## Results
- **SOAP Section Accuracy**: 91%
- **Factual Grounding**: +15% over baseline LLM
- **Clinical Coherence Score**: ↑16% (LLM-as-a-judge)
- **Fluency**: ↓12% perplexity using GPT-2 scoring

## Datasets Used
- **ACI-Bench**: Simulated doctor-patient transcripts and expert-authored SOAP notes.
- **MTS-Dialog**: Used for fine-tuning in comparative LLM evaluation.

## Evaluation Metrics
- SOAP structure verification
- LLM-as-a-Judge scoring (DeepSeek-R1)
- QA-based factual consistency (T5 + DistilBERT)
- Clinical relevance (BERTScore + BioBERT cosine similarity)
- Fluency (GPT-2 perplexity)

## Team Members
- Prakriti Shetty
- Priya Balakrishnan  
- Avinash Nandyala  
- Donald Winkleman  

Avinash's Files: 


https://colab.research.google.com/drive/1uKQLWV3Q0UUje_oaz-TEyrgvLBS2Jm8n?usp=sharing
https://colab.research.google.com/drive/1RUk3vxfFjgpOHa05LkuClurMNM5cRuJm?usp=sharing
