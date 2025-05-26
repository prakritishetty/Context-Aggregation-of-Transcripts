import pandas as pd
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import seaborn as sns
from tqdm import tqdm
import json
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from bert_score import BERTScorer
# from bleurt import score
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the key phrases data
def load_key_phrases(file_path):
    """
    Load key phrases from CSV file and preprocess them.

    Args:
        file_path (str): Path to the CSV file containing key phrases

    Returns:
        list: List of preprocessed key phrases
    """
    df = pd.read_csv(file_path)

    # Extract individual key phrases from the text
    all_phrases = []
    for _, row in df.iterrows():
        # Extract phrases between quotes
        phrases = re.findall(r'"([^"]*)"', row['cleaned_key_phrases'])
        # Clean up phrases
        phrases = [p.strip() for p in phrases if len(p.strip()) > 10]  # Filter out very short phrases
        all_phrases.append(phrases)
    print(f'All phrases: {len(all_phrases)}')
    return all_phrases

# Load the model and tokenizer for embeddings
def load_model_and_tokenizer(model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"):
    """
    Load the model and tokenizer for generating embeddings.

    Args:
        model_name (str): Name of the model to use

    Returns:
        tuple: (tokenizer, model)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    return tokenizer, model

# Generate prompt-aware embeddings
def generate_embeddings(phrases, tokenizer, model, prompt_type):
    """
    Generate embeddings for phrases using prompt-aware approach.

    Args:
        phrases (list): List of phrases to embed
        tokenizer: Tokenizer for the model
        model: Model for generating embeddings
        prompt_type (str): Type of prompt to use (S, O, A, P)

    Returns:
        numpy.ndarray: Array of embeddings
    """
    # Define prompts for each SOAP category
    prompts = {
        'S': "Subjective: This section captures the patient's personal experiences and feelings. It includes the chief complaint, history of present illness, and any other relevant personal or family medical history. Example: 'The patient reports a persistent headache for the past three days.'",
        'O': "Objective: This section records measurable or observable data from the patient encounter, such as vital signs, physical examination findings, and laboratory results. Example: 'Blood pressure is 140/90 mmHg, and the patient has a temperature of 37.5°C.'",
        'A': "Assessment: This section provides a medical diagnosis or assessment based on the subjective and objective information. It includes the clinician's interpretation and analysis of the patient's condition. Example: 'The patient is diagnosed with hypertension based on elevated blood pressure readings.'",
        'P': "Plan: This section outlines the treatment strategy, including medications, therapies, and follow-up appointments. It details the steps to manage the patient's condition. Example: 'Prescribe lisinopril 10 mg daily and schedule a follow-up in two weeks.'"
    }
    

    embeddings = []

    for phrase in tqdm(phrases, desc=f"Generating {prompt_type} embeddings"):
        # Add the appropriate prompt
        prompted_phrase = f"Generate an embedding for {phrase} based on the context {prompts[prompt_type]}, since this the type of phrase it is. Understand from the example and "

        # Tokenize and generate embeddings
        inputs = tokenizer(prompted_phrase, return_tensors="pt", truncation=True, max_length=2048, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Use the [CLS] token embedding as the sentence embedding
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(embedding[0])

    return np.array(embeddings)

# Perform hierarchical clustering
def perform_hierarchical_clustering(embeddings, n_clusters=4):
    """
    Perform hierarchical clustering on embeddings.

    Args:
        embeddings (numpy.ndarray): Array of embeddings
        n_clusters (int): Number of clusters to form

    Returns:
        tuple: (clustering model, cluster labels, linkage matrix)
    """
    # Adjust number of clusters if there are fewer samples than requested clusters
    n_samples = len(embeddings)
    if n_samples < n_clusters:
        print(f"Warning: Only {n_samples} samples available, reducing number of clusters from {n_clusters} to {n_samples}")
        n_clusters = n_samples

    # Compute linkage matrix
    linkage_matrix = linkage(embeddings, method='ward')

    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    cluster_labels = clustering.fit_predict(embeddings)

    return clustering, cluster_labels, linkage_matrix

# Perform subclustering for each SOAP category
def perform_subclustering(embeddings, cluster_labels, cluster_to_soap, n_subclusters=3):
    """
    Perform subclustering for each SOAP category.

    Args:
        embeddings (numpy.ndarray): Array of embeddings
        cluster_labels (numpy.ndarray): Cluster labels
        cluster_to_soap (dict): Mapping of clusters to SOAP categories
        n_subclusters (int): Number of subclusters to form for each SOAP category

    Returns:
        dict: Mapping of (cluster, subcluster) to subcluster label
    """
    subcluster_labels = {}

    # Group embeddings by SOAP category
    soap_embeddings = defaultdict(list)
    soap_indices = defaultdict(list)

    for i, label in enumerate(cluster_labels):
        soap_category = cluster_to_soap[label]
        soap_embeddings[soap_category].append(embeddings[i])
        soap_indices[soap_category].append(i)

    # Perform subclustering for each SOAP category
    for soap_category, category_embeddings in soap_embeddings.items():
        if len(category_embeddings) < n_subclusters:
            # Not enough samples for subclustering, assign all to subcluster 0
            print(f"Warning: Only {len(category_embeddings)} samples available for {soap_category}, assigning all to subcluster 0")
            for idx in soap_indices[soap_category]:
                subcluster_labels[idx] = 0
            continue

        # Convert to numpy array
        category_embeddings = np.array(category_embeddings)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_subclusters, random_state=42)
        subcluster_results = kmeans.fit_predict(category_embeddings)

        # Map back to original indices
        for i, idx in enumerate(soap_indices[soap_category]):
            subcluster_labels[idx] = subcluster_results[i]

    return subcluster_labels

def evaluate_clustering(embeddings, cluster_labels):
    """
    Evaluate clustering using multiple metrics.

    Args:
        embeddings (numpy.ndarray): Array of embeddings
        cluster_labels (numpy.ndarray): Cluster labels

    Returns:
        dict: Dictionary of evaluation metrics
    """
    n_samples = len(embeddings)
    n_labels = len(np.unique(cluster_labels))

    # Initialize metrics dictionary
    metrics = {}

    # Only calculate silhouette score if we have enough samples and labels
    if n_samples > 2 and 1 < n_labels < n_samples:
        # Silhouette score
        metrics['silhouette_score'] = silhouette_score(embeddings, cluster_labels)
    else:
        # For small samples, use a simple distance-based metric instead
        print(f"Warning: Not enough samples ({n_samples}) or labels ({n_labels}) for silhouette score. Using alternative metric.")
        # Calculate average distance between samples in the same cluster
        same_cluster_distances = []
        for label in np.unique(cluster_labels):
            mask = cluster_labels == label
            if np.sum(mask) > 1:  # Only if there are at least 2 samples in this cluster
                cluster_embeddings = embeddings[mask]
                # Calculate pairwise distances
                distances = np.zeros((len(cluster_embeddings), len(cluster_embeddings)))
                for i in range(len(cluster_embeddings)):
                    for j in range(i+1, len(cluster_embeddings)):
                        dist = np.linalg.norm(cluster_embeddings[i] - cluster_embeddings[j])
                        distances[i, j] = distances[j, i] = dist
                # Average distance for this cluster
                same_cluster_distances.append(np.mean(distances[distances > 0]))

        # Use the average of same-cluster distances as a proxy for silhouette score
        metrics['silhouette_score'] = np.mean(same_cluster_distances) if same_cluster_distances else 0.0

    # Calinski-Harabasz index
    if n_samples > 2 and n_labels > 1:
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(embeddings, cluster_labels)
    else:
        metrics['calinski_harabasz_score'] = 0.0

    # Davies-Bouldin index
    if n_samples > 2 and n_labels > 1:
        metrics['davies_bouldin_score'] = davies_bouldin_score(embeddings, cluster_labels)
    else:
        metrics['davies_bouldin_score'] = 0.0

    return metrics

# Calculate semantic similarity between clusters
def calculate_cluster_similarity(embeddings, cluster_labels):
    """
    Calculate semantic similarity between clusters.

    Args:
        embeddings (numpy.ndarray): Array of embeddings
        cluster_labels (numpy.ndarray): Cluster labels

    Returns:
        numpy.ndarray: Similarity matrix between clusters
    """
    # Calculate centroid for each cluster
    unique_labels = np.unique(cluster_labels)
    centroids = {}

    for label in unique_labels:
        mask = cluster_labels == label
        centroids[label] = np.mean(embeddings[mask], axis=0)

    # Calculate similarity between centroids
    similarity_matrix = np.zeros((len(unique_labels), len(unique_labels)))
    for i, label1 in enumerate(unique_labels):
        for j, label2 in enumerate(unique_labels):
            similarity = cosine_similarity([centroids[label1]], [centroids[label2]])[0][0]
            similarity_matrix[i, j] = similarity

    return similarity_matrix

# Calculate semantic coherence of clusters
def calculate_semantic_coherence(embeddings, cluster_labels, phrases):
    """
    Calculate semantic coherence of clusters using BERTScore.

    Args:
        embeddings (numpy.ndarray): Array of embeddings
        cluster_labels (numpy.ndarray): Cluster labels
        phrases (list): List of phrases

    Returns:
        dict: Dictionary of coherence scores for each cluster
    """
    # Initialize BERTScore
    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)

    # Calculate coherence for each cluster
    unique_labels = np.unique(cluster_labels)
    coherence_scores = {}

    for label in unique_labels:
        mask = cluster_labels == label
        cluster_phrases = [phrases[i] for i in range(len(phrases)) if mask[i]]

        if len(cluster_phrases) < 2:
            coherence_scores[label] = 0.0
            continue

        # Calculate pairwise similarities
        similarities = []
        for i in range(len(cluster_phrases)):
            for j in range(i+1, len(cluster_phrases)):
                score = bert_scorer.score([cluster_phrases[i]], [cluster_phrases[j]])[2].mean()
                similarities.append(score)

        # Average similarity as coherence score
        coherence_scores[label] = np.mean(similarities)

    return coherence_scores

# Visualize clustering results
def visualize_clustering(linkage_matrix, cluster_labels, phrases, output_dir="visualizations"):
    """
    Visualize clustering results.

    Args:
        linkage_matrix (numpy.ndarray): Linkage matrix
        cluster_labels (numpy.ndarray): Cluster labels
        phrases (list): List of phrases
        output_dir (str): Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Plot dendrogram
    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix, labels=cluster_labels, leaf_rotation=90)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dendrogram.png"))
    plt.close()

    # Plot cluster distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x=cluster_labels)
    plt.title("Distribution of Clusters")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cluster_distribution.png"))
    plt.close()

# Visualize embeddings using dimensionality reduction
def visualize_embeddings(embeddings, cluster_labels, phrases, output_dir="visualizations"):
    """
    Visualize embeddings using dimensionality reduction techniques.

    Args:
        embeddings (numpy.ndarray): Array of embeddings
        cluster_labels (numpy.ndarray): Cluster labels
        phrases (list): List of phrases
        output_dir (str): Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_embeddings = tsne.fit_transform(embeddings)

    # Plot t-SNE
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=cluster_labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title("t-SNE Visualization of Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tsne_visualization.png"))
    plt.close()

    # Apply UMAP
    umap_reducer = umap.UMAP(random_state=42)
    umap_embeddings = umap_reducer.fit_transform(embeddings)

    # Plot UMAP
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=cluster_labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title("UMAP Visualization of Embeddings")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "umap_visualization.png"))
    plt.close()

    # Apply PCA
    pca = PCA(n_components=2)
    pca_embeddings = pca.fit_transform(embeddings)

    # Plot PCA
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], c=cluster_labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title("PCA Visualization of Embeddings")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_visualization.png"))
    plt.close()

# Map clusters to SOAP categories
def map_clusters_to_soap(cluster_labels, embeddings, phrases):
    """
    Map clusters to SOAP categories based on semantic similarity.

    Args:
        cluster_labels (numpy.ndarray): Cluster labels
        embeddings (numpy.ndarray): Array of embeddings
        phrases (list): List of phrases

    Returns:
        dict: Mapping of clusters to SOAP categories
    """
    # Define SOAP embeddings (using a few representative phrases for each category)
    soap_representatives = {
        'S': ["patient reports", "patient states", "patient feels", "patient describes"],
        'O': ["physical examination", "vital signs", "lab results", "imaging shows"],
        'A': ["diagnosis", "assessment", "impression", "evaluation"],
        'P': ["plan", "treatment", "recommendation", "follow-up"]
    }

    # Generate embeddings for SOAP representatives
    tokenizer, model = load_model_and_tokenizer()
    soap_embeddings = {}

    for category, reps in soap_representatives.items():
        category_embeddings = []
        for rep in reps:
            inputs = tokenizer(rep, return_tensors="pt", truncation=True, max_length=2048, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            category_embeddings.append(embedding[0])
        # print(f'Category embeddings: {len(category_embeddings)}')
        soap_embeddings[category] = np.mean(category_embeddings, axis=0)
    # print(f'Soap embeddings: {soap_embeddings}, {len(soap_embeddings["S"])}')

    # Calculate similarity between clusters and SOAP categories
    unique_labels = np.unique(cluster_labels)
    cluster_centroids = {}
    # print(f'Unique labels: {unique_labels}')
    for label in unique_labels:
        mask = cluster_labels == label
        cluster_centroids[label] = np.mean(embeddings[mask], axis=0)

    # Map clusters to SOAP categories
    cluster_to_soap = {}
    for label in unique_labels:
        similarities = {}
        for category, embedding in soap_embeddings.items():
            # print(f'cluster_centroids[label].shape: {cluster_centroids[label].shape}')
            # print(f'embedding: {embedding.shape}')
            similarity = cosine_similarity([cluster_centroids[label]], [embedding])[0][0]
            similarities[category] = similarity

        # Assign to the most similar SOAP category
        best_category = max(similarities, key=similarities.get)
        cluster_to_soap[label] = best_category

    return cluster_to_soap

def process_phrase_set(phrases, phrase_set_index, tokenizer, model, output_base_dir="results_new1"):
    """
    Process a single set of phrases and save all results.

    Args:
        phrases (list): List of phrases to process
        phrase_set_index (int): Index of the phrase set
        tokenizer: Tokenizer for the model
        model: Model for generating embeddings
        output_base_dir (str): Base directory for output files

    Returns:
        dict: Dictionary containing all results for this phrase set
    """
    # Create output directories
    os.makedirs(f"{output_base_dir}/dendrograms", exist_ok=True)
    os.makedirs(f"{output_base_dir}/embeddings", exist_ok=True)
    os.makedirs(f"{output_base_dir}/json", exist_ok=True)
    os.makedirs(f"{output_base_dir}/csv", exist_ok=True)

    print(f"Processing phrase set {phrase_set_index} with {len(phrases)} phrases")

    # Generate embeddings for each SOAP category
    all_embeddings = {}
    for prompt_type in tqdm(['S', 'O', 'A', 'P'], desc="Generating embeddings", position=0, leave=True):
        print(f"Generating embeddings for {prompt_type}...")
        embeddings = generate_embeddings(phrases, tokenizer, model, prompt_type)
        all_embeddings[prompt_type] = embeddings

    # Combine embeddings from all prompt types
    combined_embeddings = np.mean([emb for emb in all_embeddings.values()], axis=0)

    # Perform hierarchical clustering
    print("Performing hierarchical clustering...")
    clustering, cluster_labels, linkage_matrix = perform_hierarchical_clustering(combined_embeddings)

    # Map clusters to SOAP categories
    cluster_to_soap = map_clusters_to_soap(cluster_labels, combined_embeddings, phrases)

    # Perform subclustering for each SOAP category
    print("Performing subclustering...")
    subcluster_labels = perform_subclustering(combined_embeddings, cluster_labels, cluster_to_soap)

    # Evaluate clustering
    evaluation_metrics = evaluate_clustering(combined_embeddings, cluster_labels)

    # Calculate cluster similarity
    similarity_matrix = calculate_cluster_similarity(combined_embeddings, cluster_labels)

    # Calculate semantic coherence
    coherence_scores = calculate_semantic_coherence(combined_embeddings, cluster_labels, phrases)

    # Visualize clustering with custom output directory
    visualize_clustering(
        linkage_matrix,
        cluster_labels,
        phrases,
        output_dir=f"{output_base_dir}/dendrograms/phrase_set_{phrase_set_index}"
    )

    # Visualize embeddings with custom output directory
    # visualize_embeddings(
    #     combined_embeddings,
    #     cluster_labels,
    #     phrases,
    #     output_dir=f"{output_base_dir}/embeddings/phrase_set_{phrase_set_index}"
    # )

    # Create DataFrame for results
    results_df = pd.DataFrame({
        'phrases': phrases,
        'cluster_labels': cluster_labels,
    })
    results_df.to_csv(f'{output_base_dir}/csv/clustering_results_phrase_set_{phrase_set_index}.csv', index=False)

    # Group phrases by SOAP category and subcluster and save to CSV
    soap_phrases_list = []
    for i, label in enumerate(cluster_labels):
        soap_category = cluster_to_soap[label]
        subcluster = subcluster_labels[i]
        soap_phrases_list.append({
            'soap_category': soap_category,
            'main_cluster': label,
            'subcluster': subcluster,
            'phrase': phrases[i]
        })

    soap_phrases_df = pd.DataFrame(soap_phrases_list)
    soap_phrases_df.to_csv(f'{output_base_dir}/csv/soap_categorized_phrases_phrase_set_{phrase_set_index}.csv', index=False)

    # Organize phrases by main cluster and subcluster
    cluster_phrases = defaultdict(lambda: defaultdict(list))
    for i, label in enumerate(cluster_labels):
        main_cluster = label
        subcluster = subcluster_labels[i]
        cluster_phrases[str(main_cluster)][str(subcluster)].append(phrases[i])

    # Prepare results for JSON
    results = {
        'phrases': phrases,
        'cluster_labels': cluster_labels.tolist(),
        'subcluster_labels': subcluster_labels,
        'cluster_to_soap': {str(k): v for k, v in cluster_to_soap.items()},  # Convert keys to strings
        'evaluation_metrics': evaluation_metrics,
        'similarity_matrix': similarity_matrix.tolist(),
        'coherence_scores': {str(k): v for k, v in coherence_scores.items()},  # Convert keys to strings
        'cluster_phrases': cluster_phrases  # Organized by main cluster and subcluster
    }

    # Save results to JSON
    def default_serializer(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(f'{output_base_dir}/json/results_phrase_set_{phrase_set_index}.json', 'w') as f:
        json.dump(results, f, indent=2, default=default_serializer)

    print(f"Results for phrase set {phrase_set_index} saved to {output_base_dir}/")

    return results

def main_hc():
    # Load key phrases
    all_phrases = load_key_phrases("key_phrases_results.csv")

    print(f"Loaded {len(all_phrases)} phrase sets")

    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer()

    # Process each phrase set
    all_results = []
    for i, phrases in tqdm(enumerate(all_phrases[0:4]), desc="Processing phrase sets"):
        results = process_phrase_set(phrases, i, tokenizer, model)
        all_results.append(results)

    # Save summary of all results
    summary = {
        'total_phrase_sets': len(all_phrases),
        'phrase_set_sizes': [len(result['phrases']) for result in all_results],
        'average_evaluation_metrics': {
            metric: np.mean([result['evaluation_metrics'][metric] for result in all_results])
            for metric in all_results[0]['evaluation_metrics'].keys()
        }
    }

    with open('results/json/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("All phrase sets processed. Summary saved to results/json/summary.json")

