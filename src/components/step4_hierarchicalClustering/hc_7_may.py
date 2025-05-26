import pandas as pd
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
import os
import json
from tqdm import tqdm
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
from scipy.optimize import linear_sum_assignment


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
        tuple: (clustering model, cluster labels, linkage matrix) or None if embeddings is empty
    """
    # Check if embeddings array is empty
    if len(embeddings) == 0:
        print("Warning: Empty embeddings array, skipping hierarchical clustering")
        return None

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

# Perform subclustering for each main cluster, ensuring subcluster numbers are unique and consecutive within each main cluster
def perform_subclustering(embeddings, cluster_labels, cluster_to_soap, n_subclusters=3):
    """
    Perform subclustering for each main cluster, ensuring subcluster numbers are unique and consecutive within each main cluster.
    Returns a list of subcluster labels (same order as phrases/embeddings).
    """
    subcluster_labels = np.zeros(len(cluster_labels), dtype=int)
    unique_main_clusters = np.unique(cluster_labels)
    for main_cluster in unique_main_clusters:
        indices = np.where(cluster_labels == main_cluster)[0]
        if len(indices) < n_subclusters:
            for idx in indices:
                subcluster_labels[idx] = 0
            continue
        cluster_embeds = embeddings[indices]
        kmeans = KMeans(n_clusters=n_subclusters, random_state=42)
        sub_labels = kmeans.fit_predict(cluster_embeds)
        # Remap sub_labels to be consecutive starting from 0 within this main cluster
        unique_subs, new_labels = np.unique(sub_labels, return_inverse=True)
        for i, idx in enumerate(indices):
            subcluster_labels[idx] = new_labels[i]
    return subcluster_labels.tolist()

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

# Visualize clustering results with:
# - SOAP label at the cut for 4 clusters (on the vertical lines that intersect the red line, using color order)
# - Subcluster numbers at internal nodes (unique within main cluster)
# - Phrases at leaves
def visualize_clustering(linkage_matrix, cluster_labels, phrases, subcluster_labels, cluster_to_soap, output_dir="visualizations"):
    """
    Visualize clustering results with:
    - SOAP label at the cut for 4 clusters (on the vertical lines that intersect the red line, using color order)
    - Subcluster numbers at internal nodes (unique within main cluster)
    - Phrases at leaves
    """
    os.makedirs(output_dir, exist_ok=True)
    Z = linkage_matrix
    n = len(phrases)
    leaf_labels = [phrases[i] for i in range(n)]
    node_subcluster = {}
    for i in range(n):
        node_subcluster[i] = subcluster_labels[i]
    next_subcluster = max(subcluster_labels) + 1 if len(subcluster_labels) > 0 else 0
    for i in range(Z.shape[0]):
        node_id = i + n
        left = int(Z[i, 0])
        right = int(Z[i, 1])
        if node_subcluster[left] == node_subcluster[right]:
            node_subcluster[node_id] = node_subcluster[left]
        else:
            node_subcluster[node_id] = next_subcluster
            next_subcluster += 1
    # Find the cut height for 4 clusters
    cut_height = Z[-4, 2] if Z.shape[0] >= 4 else Z[-1, 2]
    plt.figure(figsize=(15, 10))
    dendro = dendrogram(
        Z,
        labels=leaf_labels,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=cut_height,
        above_threshold_color='black',
        orientation='top',
        show_leaf_counts=False
    )
    # Add subcluster number at each internal node
    icoord = dendro['icoord']
    dcoord = dendro['dcoord']
    color_list = dendro['color_list']
    # Draw subcluster numbers
    for i, (xs, ys) in enumerate(zip(icoord, dcoord)):
        x = 0.5 * (xs[1] + xs[2])
        y = ys[1]
        node_id = i + n
        label = str(node_subcluster[node_id])
        plt.text(x, y, label, va='bottom', ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    # Draw horizontal line at cut height
    plt.axhline(y=cut_height, color='red', linestyle='--', linewidth=2, label='Main cluster cut')
    # Place SOAP labels on the vertical lines that intersect the red line, using color order
    from scipy.cluster.hierarchy import fcluster
    main_cluster_assignments = fcluster(Z, t=cut_height, criterion='distance')
    # Find the x positions and colors of the main cluster roots
    soap_labels = list(cluster_to_soap.values())
    color_to_soap = {}
    # Get the color order for the 4 main clusters
    main_cluster_colors = []
    main_cluster_xs = []
    for i, (xs, ys) in enumerate(zip(icoord, dcoord)):
        if ys[1] >= cut_height and ys[2] >= cut_height:
            # This is a main cluster root
            color = color_list[i]
            x = 0.5 * (xs[1] + xs[2])
            main_cluster_colors.append(color)
            main_cluster_xs.append(x)
    # Sort by x to match left-to-right order
    sorted_clusters = sorted(zip(main_cluster_xs, main_cluster_colors, soap_labels), key=lambda t: t[0])
    for x, color, soap in sorted_clusters:
        plt.text(x, cut_height + 0.1, soap, va='bottom', ha='center', fontsize=14, color='blue', bbox=dict(facecolor='white', alpha=0.9, edgecolor='blue'))
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Phrases")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dendrogram.png"), dpi=300, bbox_inches='tight')
    plt.close()
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


def map_clusters_to_soap(cluster_labels, embeddings, phrases):
    """
    Ensemble method for mapping clusters to SOAP categories using:
    1. Prompt-based embedding comparison
    2. Topic modeling
    3. Zero/few-shot LLM classification
    Returns per-method and majority-vote assignments.
    Ensures one-to-one mapping between clusters and SOAP categories for each method.
    """
    # --- 1. Prompt-based Embedding Comparison ---
    soap_prompts = {
        'S': "Subjective: This section captures the patient's personal experiences and feelings. It includes the chief complaint, history of present illness, and any other relevant personal or family medical history. Example: 'The patient reports a persistent headache for the past three days.'",
        'O': "Objective: This section records measurable or observable data from the patient encounter, such as vital signs, physical examination findings, and laboratory results. Example: 'Blood pressure is 140/90 mmHg, and the patient has a temperature of 37.5°C.'",
        'A': "Assessment: This section provides a medical diagnosis or assessment based on the subjective and objective information. It includes the clinician's interpretation and analysis of the patient's condition. Example: 'The patient is diagnosed with hypertension based on elevated blood pressure readings.'",
        'P': "Plan: This section outlines the treatment strategy, including medications, therapies, and follow-up appointments. It details the steps to manage the patient's condition. Example: 'Prescribe lisinopril 10 mg daily and schedule a follow-up in two weeks.'"
    }
    tokenizer, model = load_model_and_tokenizer()
    prompt_embeds = {}
    for k, v in soap_prompts.items():
        inputs = tokenizer(v, return_tensors="pt", truncation=True, max_length=256, padding=True)
        inputs = {k2: v2.to(model.device) for k2, v2 in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        prompt_embeds[k] = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
    # Compute cluster centroids
    unique_labels = np.unique(cluster_labels)
    cluster_centroids = {label: np.mean(embeddings[cluster_labels == label], axis=0) for label in unique_labels}
    soap_keys = list(soap_prompts.keys())
    n_clusters = len(unique_labels)
    n_soap = len(soap_keys)
    # --- Prompt-based Hungarian assignment ---
    prompt_score_matrix = np.zeros((n_clusters, n_soap))
    for i, label in enumerate(unique_labels):
        centroid = cluster_centroids[label]
        for j, soap in enumerate(soap_keys):
            prompt_score_matrix[i, j] = cosine_similarity([centroid], [prompt_embeds[soap]])[0][0]
    # Hungarian: maximize similarity (minimize negative similarity)
    row_ind, col_ind = linear_sum_assignment(-prompt_score_matrix)
    prompt_assignments = {unique_labels[i]: soap_keys[j] for i, j in zip(row_ind, col_ind)}

    # --- 2. Topic Modeling (NMF) Hungarian assignment ---
    topic_keywords = {
        'S': ['report', 'feel', 'describe', 'complain', 'history', 'symptom'],
        'O': ['exam', 'vital', 'lab', 'result', 'sign', 'physical', 'imaging'],
        'A': ['diagnosis', 'assessment', 'impression', 'evaluation'],
        'P': ['plan', 'treatment', 'recommend', 'prescribe', 'follow-up', 'therapy']
    }
    topic_score_matrix = np.zeros((n_clusters, n_soap))
    for i, label in enumerate(unique_labels):
        cluster_phrases = [phrases[k] for k in range(len(phrases)) if cluster_labels[k] == label]
        if len(cluster_phrases) < 2:
            topic_score_matrix[i, :] = 0
            topic_score_matrix[i, 0] = 1  # fallback: S
            continue
        vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        X = vectorizer.fit_transform(cluster_phrases)
        nmf = NMF(n_components=1, random_state=42)
        W = nmf.fit_transform(X)
        H = nmf.components_[0]
        top_indices = H.argsort()[::-1][:5]
        top_words = [vectorizer.get_feature_names_out()[idx] for idx in top_indices]
        for j, soap in enumerate(soap_keys):
            topic_score_matrix[i, j] = sum(word in topic_keywords[soap] for word in top_words)
    row_ind, col_ind = linear_sum_assignment(-topic_score_matrix)
    topic_assignments = {unique_labels[i]: soap_keys[j] for i, j in zip(row_ind, col_ind)}

    # --- 3. Zero/Few-shot LLM Classification Hungarian assignment ---
    llm_score_matrix = np.zeros((n_clusters, n_soap))
    try:
        zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        candidate_labels = ["Subjective", "Objective", "Assessment", "Plan"]
        for i, label in enumerate(unique_labels):
            cluster_phrases = [phrases[k] for k in range(len(phrases)) if cluster_labels[k] == label]
            rep_phrase = max(cluster_phrases, key=len)
            result = zero_shot(rep_phrase, candidate_labels)
            for j, soap in enumerate(soap_keys):
                # Use the score for the corresponding label
                label_idx = result['labels'].index({
                    'S': 'Subjective', 'O': 'Objective', 'A': 'Assessment', 'P': 'Plan'
                }[soap])
                llm_score_matrix[i, j] = result['scores'][label_idx]
    except Exception as e:
        print(f"LLM zero-shot classification failed: {e}")
        llm_score_matrix[:, 0] = 1  # fallback: all S
    row_ind, col_ind = linear_sum_assignment(-llm_score_matrix)
    llm_assignments = {unique_labels[i]: soap_keys[j] for i, j in zip(row_ind, col_ind)}

    # --- 4. Ensemble Hungarian assignment ---
    # For ensemble, use majority voting as a score matrix
    ensemble_score_matrix = np.zeros((n_clusters, n_soap))
    for i, label in enumerate(unique_labels):
        votes = [prompt_assignments[label], topic_assignments[label], llm_assignments[label]]
        for j, soap in enumerate(soap_keys):
            ensemble_score_matrix[i, j] = votes.count(soap)
    row_ind, col_ind = linear_sum_assignment(-ensemble_score_matrix)
    ensemble_assignments = {unique_labels[i]: soap_keys[j] for i, j in zip(row_ind, col_ind)}

    # --- Return all assignments for interpretability ---
    return {
        'prompt': prompt_assignments,
        'topic': topic_assignments,
        'llm': llm_assignments,
        'ensemble': ensemble_assignments
    }

def convert_keys_to_str(d):
    if isinstance(d, dict):
        return {str(k): convert_keys_to_str(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_keys_to_str(i) for i in d]
    else:
        return d

def process_phrase_set(phrases, phrase_set_index, tokenizer, model, output_base_dir="results_ensemble"):
    os.makedirs(f"{output_base_dir}/dendrograms", exist_ok=True)
    os.makedirs(f"{output_base_dir}/embeddings", exist_ok=True)
    os.makedirs(f"{output_base_dir}/json", exist_ok=True)
    os.makedirs(f"{output_base_dir}/csv", exist_ok=True)
    print(f"Processing phrase set {phrase_set_index} with {len(phrases)} phrases")
    # Generate embeddings for each SOAP category
    all_embeddings = {}
    for prompt_type in tqdm(['S', 'O', 'A', 'P'], desc="Generating embeddings", position=0, leave=True):
        embeddings = generate_embeddings(phrases, tokenizer, model, prompt_type)
        all_embeddings[prompt_type] = embeddings
    combined_embeddings = np.mean([emb for emb in all_embeddings.values()], axis=0)
    
    # Perform hierarchical clustering
    clustering_result = perform_hierarchical_clustering(combined_embeddings)
    if clustering_result is None:
        print(f"Skipping phrase set {phrase_set_index} due to empty embeddings")
        return None
        
    clustering, cluster_labels, linkage_matrix = clustering_result
    soap_assignments = map_clusters_to_soap(cluster_labels, combined_embeddings, phrases)
    # Use ensemble for downstream, but save all
    cluster_to_soap = soap_assignments['ensemble']
    subcluster_labels = perform_subclustering(combined_embeddings, cluster_labels, cluster_to_soap)
    evaluation_metrics = evaluate_clustering(combined_embeddings, cluster_labels)
    similarity_matrix = calculate_cluster_similarity(combined_embeddings, cluster_labels)
    coherence_scores = calculate_semantic_coherence(combined_embeddings, cluster_labels, phrases)
    visualize_clustering(
        linkage_matrix,
        cluster_labels,
        phrases,
        subcluster_labels,
        cluster_to_soap,
        output_dir=f"{output_base_dir}/dendrograms/phrase_set_{phrase_set_index}"
    )
    results_df = pd.DataFrame({
        'phrases': phrases,
        'cluster_labels': cluster_labels,
    })
    results_df.to_csv(f'{output_base_dir}/csv/clustering_results_phrase_set_{phrase_set_index}.csv', index=False)
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
    cluster_phrases = defaultdict(lambda: defaultdict(list))
    for i, label in enumerate(cluster_labels):
        main_cluster = label
        subcluster = subcluster_labels[i]
        cluster_phrases[str(main_cluster)][str(subcluster)].append(phrases[i])
    results = {
        'phrases': phrases,
        'cluster_labels': cluster_labels.tolist(),
        'subcluster_labels': subcluster_labels,
        'soap_assignments': convert_keys_to_str(soap_assignments),
        'cluster_to_soap': {str(k): v for k, v in cluster_to_soap.items()},
        'evaluation_metrics': evaluation_metrics,
        'similarity_matrix': similarity_matrix.tolist(),
        'coherence_scores': {str(k): v for k, v in coherence_scores.items()},
        'cluster_phrases': cluster_phrases
    }
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
    for i, phrases in tqdm(enumerate(all_phrases), desc="Processing phrase sets"):
        if i < 93:
            continue
        try:
            results = process_phrase_set(phrases, i, tokenizer, model)
            if results is not None:
                all_results.append(results)
            else:
                # Add empty result for skipped set
                empty_result = {
                    'phrases': [],
                    'cluster_labels': [],
                    'subcluster_labels': [],
                    'soap_assignments': {'prompt': {}, 'topic': {}, 'llm': {}, 'ensemble': {}},
                    'cluster_to_soap': {},
                    'evaluation_metrics': {},
                    'similarity_matrix': [],
                    'coherence_scores': {},
                    'cluster_phrases': {}
                }
                all_results.append(empty_result)
                print(f"Added empty result for phrase set {i} due to empty embeddings")
        except Exception as e:
            # Add empty result for error case
            empty_result = {
                'phrases': [],
                'cluster_labels': [],
                'subcluster_labels': [],
                'soap_assignments': {'prompt': {}, 'topic': {}, 'llm': {}, 'ensemble': {}},
                'cluster_to_soap': {},
                'evaluation_metrics': {},
                'similarity_matrix': [],
                'coherence_scores': {},
                'cluster_phrases': {}
            }
            all_results.append(empty_result)
            print(f"Error processing phrase set {i}: {str(e)}")

    # Skip summary if no results were processed
    if not all_results:
        print("No phrase sets were processed")
        return

    # Save summary of all results
    summary = {
        'total_phrase_sets': len(all_phrases),
        'processed_phrase_sets': len(all_results),
        'phrase_set_sizes': [len(result['phrases']) for result in all_results],
        'average_evaluation_metrics': {
            metric: np.mean([result['evaluation_metrics'][metric] for result in all_results if result['evaluation_metrics']])
            for metric in ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']
        }
    }

    # Create results directory if it doesn't exist
    os.makedirs('results/json', exist_ok=True)

    with open('results/json/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"All phrase sets processed. Summary saved to results/json/summary.json")
    print(f"Processed {len(all_results)} sets")


