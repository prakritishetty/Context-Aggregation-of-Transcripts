#%%
import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def extract_key_phrases(document, model="gpt-4"):
    """
    Extract key phrases from a document using OpenAI's API.
    
    Args:
        document (str): The input document text
        model (str): The OpenAI model to use (default: "gpt-4")
        
    Returns:
        list: List of extracted key phrases
        
    Raises:
        Exception: If there's an error with the API call
    """
    try:
        prompt = f"""
        Extract 3-5 key phrases from this document that represent its main themes.
        Return only comma-separated phrases without explanations.
        
        Document: {document}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100
        )
        return [phrase.strip() for phrase in response.choices[0].message.content.split(",")]
    except Exception as e:
        print(f"Error extracting key phrases: {str(e)}")
        return []

# Example usage

# Read the first row from the CSV file
csv_path = "MTS-Dialog-TrainingSet.csv"
df = pd.read_csv(csv_path)
document = df['dialogue'].iloc[0]  # Get the first dialogue

# Extract key phrases from the document
key_phrases = extract_key_phrases(document)
print("Document:", document)
print("Extracted key phrases:", key_phrases)


#%%
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-mpnet-base-v2')

def validate_coherence(document, key_phrases, threshold=0.1):
    doc_embedding = model.encode([document])
    phrase_embeddings = model.encode(key_phrases)
    
    similarities = cosine_similarity(doc_embedding, phrase_embeddings)
    return [phrase for phrase, score in zip(key_phrases, similarities[0]) if score >= threshold]

# Validate coherence of extracted key phrases
coherent_phrases = validate_coherence(document, key_phrases)
print("Coherent key phrases:", coherent_phrases)


#%%
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

def optimal_clustering(embeddings, max_clusters=10):
    best_score = -1
    optimal_k = 2
    
    for k in range(2, max_clusters+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        
        if score > best_score:
            best_score = score
            optimal_k = k
            
    return optimal_k
# Test optimal clustering with sample embeddings
test_embeddings = model.encode(coherent_phrases)
k = optimal_clustering(test_embeddings)
print(f"Optimal number of clusters: {k}")


#%%
def cluster_phrases(key_phrases):
    embeddings = model.encode(key_phrases)
    optimal_k = optimal_clustering(embeddings)
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    return {
        "labels": clusters,
        "centroids": kmeans.cluster_centers_,
        "silhouette_score": silhouette_score(embeddings, clusters)
    }

# Test clustering on coherent phrases
cluster_results = cluster_phrases(coherent_phrases)
print("Clustering results:")
print(f"Number of clusters: {len(set(cluster_results['labels']))}")
print(f"Silhouette score: {cluster_results['silhouette_score']:.3f}")

# Print phrases by cluster
for cluster_id in range(len(set(cluster_results['labels']))):
    cluster_phrases = [phrase for phrase, label in zip(coherent_phrases, cluster_results['labels']) if label == cluster_id]
    print(f"\nCluster {cluster_id}:")
    print(", ".join(cluster_phrases))

#%%
def topic_coherence(topics, documents):
    # Implementation of normalized pointwise mutual information (Eq 3)
    pass 

def topic_diversity(topics):
    unique_words = len(set([word for topic in topics for word in topic]))
    total_words = sum(len(topic) for topic in topics)
    return unique_words / total_words


#%%
def evaluate_topics(topics, documents):
    coherence = topic_coherence(topics, documents)
    diversity = topic_diversity(topics)
    
    return {
        "coherence": coherence,
        "diversity": diversity
    }









