"""
Utility functions for analytics, clustering, and visualization.
"""
from typing import List, Dict, Any, Optional, Tuple

# Import logging utilities
from src.utils.logging_utils import get_logger

# Initialize logger
logger = get_logger(__name__)

# Qdrant client and models might be needed if fetch_vectors_for_clustering interacts directly
# For now, assuming it takes a pre-configured client if needed, or relies on qdrant_utils for direct interactions.
from qdrant_client import AsyncQdrantClient # For type hinting if client is passed

# Import create_qdrant_filter from the new qdrant_utils module
from src.utils.qdrant.retrieval import create_qdrant_filter

# Optional imports for analytics, attempt gracefully
try:
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.manifold import TSNE
    import plotly.graph_objects as go
    import plotly.express as px
    import nltk
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
    # Download stopwords if not already present (run once)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            logger.info("NLTK stopwords not found, attempting to download...")
            nltk.download('stopwords', quiet=True)
            logger.info("NLTK stopwords downloaded successfully.")
        except Exception as download_e: # Catch potential errors during download
            logger.warning(f"Failed to download NLTK stopwords: {download_e}")
            # NLTK_AVAILABLE = False # Consider if NLTK should be disabled if stopwords fail
            pass # Keep NLTK_AVAILABLE true, other NLTK fns might work
except ImportError as e:
    logger.warning(f"Analytics libraries (numpy, scikit-learn, plotly, nltk) not fully available: {e}. Some analytics functions may be limited.")
    np = None
    KMeans = None
    silhouette_score = None
    CountVectorizer = None
    TSNE = None
    go = None
    px = None
    nltk = None
    stopwords = None
    NLTK_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    logger.warning("Plotly not available. Visualization features will be limited.")
    PLOTLY_AVAILABLE = False


async def fetch_vectors_for_clustering(
    client: AsyncQdrantClient,
    collection_name: str,
    filter_condition: Optional[Dict[str, Any]] = None,
    sample_size: int = 1000,
    with_payload_fields: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Fetch vectors and their payloads from Qdrant for clustering analysis.
    
    Args:
        client: AsyncQdrantClient instance
        collection_name: Name of the collection
        filter_condition: Optional filter to apply (e.g., by source)
        sample_size: Maximum number of vectors to fetch
        with_payload_fields: Specific payload fields to include
    
    Returns:
        Dictionary with keys:
        - "vectors": List of vectors
        - "payloads": List of corresponding payloads
        - "ids": List of corresponding IDs
        - "actual_sample_size": Actual number of vectors fetched
        - "total_vectors_in_source": Placeholder for total vectors in source, ideally a separate count query
    """
    try:
        # Use helper function from qdrant_utils to create filter
        query_filter = create_qdrant_filter(filter_condition=filter_condition)
        
        # Default payload fields if none specified
        if with_payload_fields is None:
            with_payload_fields = ["url", "text", "source", "crawl_type", "headers"]
        
        # Fetch points using scroll - this is more efficient for getting many points
        # Ensure client.scroll is called as an async operation if the client supports it,
        # or run it in a thread if it's synchronous.
        # Assuming qdrant_client's scroll method might be synchronous based on original utils.py context
        # However, QdrantClient methods are generally synchronous unless explicitly using an async client.
        # For consistency with other async utils, let's assume it could be blocking.
        # If your QdrantClient is already async, asyncio.to_thread is not needed.
        
        # The qdrant-client is synchronous by default. To use it in an async context properly:
        import asyncio
        points_response, _ = await asyncio.to_thread(
            client.scroll,
            collection_name=collection_name,
            limit=sample_size,
            scroll_filter=query_filter, # scroll_filter is the correct param name for scroll
            with_payload=with_payload_fields,
            with_vectors=True
        )
        
        points = points_response # client.scroll returns (points, next_offset_or_none)

        if not points:
            logger.warning(f"No points found in collection '{collection_name}' with the given filter.")
            return {
                "vectors": [], 
                "payloads": [], 
                "ids": [], 
                "actual_sample_size": 0,
                "total_vectors_in_source": 0 # Or None, requires a count query for true total
            }
        
        vectors = []
        payloads = []
        ids = []
        
        for point in points:
            if point.vector:
                vectors.append(point.vector)
                payloads.append(point.payload if point.payload is not None else {})
                ids.append(point.id)
        
        logger.info(f"Fetched {len(vectors)} vectors for clustering from collection '{collection_name}'.")
        return {
            "vectors": vectors, 
            "payloads": payloads, 
            "ids": ids, 
            "actual_sample_size": len(vectors),
            "total_vectors_in_source": len(vectors) # Placeholder, ideally a separate count query
        }
    
    except Exception as e:
        logger.error(f"Error fetching vectors for clustering: {e}")
        return {
            "vectors": [], 
            "payloads": [], 
            "ids": [], 
            "actual_sample_size": 0,
            "total_vectors_in_source": 0,
            "error": str(e)
        }

async def perform_clustering(
    vectors: List[List[float]],
    num_clusters: Optional[int] = None,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Perform K-means clustering on a set of vectors.
    Calculates an optimal number of clusters if num_clusters is not provided.
    
    Args:
        vectors: List of vectors to cluster
        num_clusters: Optional number of clusters to create. If None, tries to find an optimal number.
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with clustering results:
            "success": bool,
            "num_clusters": int,
            "labels": List[int] (cluster label for each input vector),
            "cluster_sizes": Dict[str, int] (size of each cluster),
            "silhouette_score": Optional[float],
            "error": Optional[str]
    """
    if not NLTK_AVAILABLE or not np or not KMeans or not silhouette_score:
        return {
            "success": False,
            "error": "Required libraries (numpy, scikit-learn, nltk) not installed. Install them for clustering.",
            "num_clusters": 0,
            "labels": [],
            "cluster_sizes": {}
        }
    
    if not vectors or len(vectors) < 2: # Need at least 2 samples for clustering
        return {
            "success": False,
            "error": f"Not enough vectors ({len(vectors) if vectors else 0}) provided for clustering. At least 2 are required.",
            "num_clusters": 0,
            "labels": [],
            "cluster_sizes": {}
        }
    
    X = np.array(vectors)
    
    actual_num_clusters = num_clusters
    if actual_num_clusters is None:
        # Determine optimal number of clusters using silhouette score (example range)
        # This can be computationally expensive for large datasets or many k values
        range_n_clusters = range(2, min(11, len(X))) 
        if not range_n_clusters: # handles len(X) < 2, though already checked
             actual_num_clusters = len(X) if len(X) >=1 else 1 # should not happen due to earlier check
        else:
            silhouette_scores = []
            best_k = -1
            highest_score = -1 # Silhouette scores range from -1 to 1

            # Ensure we don't try more clusters than samples
            k_upper_bound = min(11, len(X)) 
            if k_upper_bound <= 1: # If only 1 sample, or k_upper_bound forces 1
                 return {
                     "success": False, 
                     "error": f"Cannot determine optimal k for {len(X)} samples. Need at least 2.",
                     "num_clusters": 0,
                     "labels": [],
                     "cluster_sizes": {}
                    }


            for k_candidate in range(2, k_upper_bound):
                try:
                    kmeans_temp = KMeans(n_clusters=k_candidate, random_state=random_seed, n_init='auto')
                    cluster_labels_temp = await asyncio.to_thread(kmeans_temp.fit_predict, X)
                    if len(set(cluster_labels_temp)) > 1: # Silhouette score requires more than 1 cluster
                        score = await asyncio.to_thread(silhouette_score, X, cluster_labels_temp)
                        silhouette_scores.append(score)
                        if score > highest_score:
                            highest_score = score
                            best_k = k_candidate
                    else: # Only one cluster formed
                        silhouette_scores.append(-1) # Assign a low score
                except Exception as e_k:
                    logger.warning(f"Could not calculate silhouette for k={k_candidate}: {e_k}")
                    silhouette_scores.append(-1) # Assign a low score

            if best_k != -1:
                actual_num_clusters = best_k
            elif len(X) >= 2 : # Fallback if no best_k found (e.g., all scores were -1)
                actual_num_clusters = max(2, len(X) // 2) if len(X) >= 4 else 2
            else: # Should be caught by initial check
                 return {
                     "success": False, 
                     "error": "Insufficient data for automatic k selection.",
                     "num_clusters": 0,
                     "labels": [],
                     "cluster_sizes": {}
                    }
            logger.info(f"Determined optimal number of clusters: {actual_num_clusters} (Silhouette scores: {silhouette_scores})")

    elif len(X) < actual_num_clusters:
        if len(X) < 2:
            return {
                "success": False,
                "error": f"Not enough samples ({len(X)}) for clustering. At least 2 are required for {actual_num_clusters} clusters.",
                "num_clusters": actual_num_clusters,
                "labels": [],
                "cluster_sizes": {}
            }
        logger.warning(f"Number of samples ({len(X)}) is less than requested num_clusters ({actual_num_clusters}). Adjusting num_clusters.")
        actual_num_clusters = max(2, len(X) // 2) if len(X) >= 4 else 2
        if len(X) == 2 or len(X) == 3: actual_num_clusters = 2


    try:
        kmeans = KMeans(n_clusters=actual_num_clusters, random_state=random_seed, n_init='auto')
        cluster_labels_list = (await asyncio.to_thread(kmeans.fit_predict, X)).tolist() # Ensure it's a list
        
        current_silhouette_avg = None
        if actual_num_clusters > 1 and len(X) > actual_num_clusters: # Check if silhouette score can be calculated
            try:
                current_silhouette_avg = float(await asyncio.to_thread(silhouette_score, X, cluster_labels_list))
            except ValueError as sve:
                logger.warning(f"Could not calculate silhouette score: {sve}")
                current_silhouette_avg = None
            except Exception as e_sil: # Catch any other exception during silhouette calculation
                logger.error(f"Error calculating silhouette score: {e_sil}")
                current_silhouette_avg = None
        
        cluster_sizes_dict = {str(i): int(np.sum(np.array(cluster_labels_list) == i)) for i in range(actual_num_clusters)}
        
        # The `cluster_content` tool does not currently expect `clusters_with_ids` or detailed item info from this util.
        # It reconstructs what it needs. The primary outputs are labels, num_clusters, and sizes.
        return {
            "success": True,
            "num_clusters": actual_num_clusters,
            "labels": cluster_labels_list, # cluster_content expects this key
            "cluster_sizes": cluster_sizes_dict, # cluster_content expects this key
            "silhouette_score": current_silhouette_avg
            # "clusters_with_ids": clusters # Not directly used by cluster_content tool as per its current logic
        }
    
    except Exception as e:
        logger.error(f"Error performing K-means clustering: {e}")
        return {
            "success": False,
            "error": str(e),
            "num_clusters": actual_num_clusters,
            "labels": [],
            "cluster_sizes": {}
        }

async def extract_cluster_themes(text: str, max_themes: int = 5) -> List[str]:
    """
    Extract potential themes or keywords from a given text, typically representing content from a cluster.
    Uses CountVectorizer to find frequent terms (unigrams and bigrams).

    Args:
        text: The input text from which to extract themes.
        max_themes: The maximum number of themes/keywords to return. Defaults to 5.

    Returns:
        A list of strings, where each string is an extracted theme/keyword.
        Returns an empty list if NLTK/Scikit-learn are unavailable, text is empty, or an error occurs.
    """
    if not NLTK_AVAILABLE or not CountVectorizer or not stopwords:
        logger.warning("NLTK or Scikit-learn not available, cannot extract themes.")
        return []
    
    if not text.strip():
        return []
        
    try:
        stop_words_set = set(stopwords.words('english'))
        additional_stops = {'https', 'http', 'www', 'com', 'html', 'the', 'and', 'for', 'with', 'this', 'that', 'not'}
        stop_words_set.update(additional_stops)
        
        vectorizer = CountVectorizer(
            max_features=max_themes*3, 
            stop_words=list(stop_words_set),
            ngram_range=(1, 2) 
        )
        
        X_counts = await asyncio.to_thread(vectorizer.fit_transform, [text])
        words = vectorizer.get_feature_names_out()
        counts = X_counts.toarray()[0]
        
        word_counts = list(zip(words, counts))
        word_counts.sort(key=lambda x: x[1], reverse=True)
        
        themes = [word for word, count in word_counts[:max_themes] if count > 1] # Only themes with count > 1
        
        return themes
    
    except Exception as e:
        logger.error(f"Error extracting cluster themes: {e}")
        return []

async def generate_cluster_visualization(
    vectors: List[List[float]], 
    cluster_labels: List[int]
) -> Optional[str]:
    """
    Generate an HTML string for cluster visualization using Plotly and t-SNE.
    
    Args:
        vectors: List of vectors that were clustered
        cluster_labels: List of cluster labels corresponding to the vectors
        
    Returns:
        HTML string of the Plotly chart, or None if visualization fails.
    """
    if not NLTK_AVAILABLE or not np or not TSNE or not go or not px:
        logger.warning("Required libraries (numpy, scikit-learn, plotly, nltk) not installed for visualization.")
        return None # Or raise an error, or return a message
    
    if not vectors or not cluster_labels or len(vectors) != len(cluster_labels):
        logger.warning("Invalid input for visualization: vectors or labels missing/mismatched.")
        return None
    
    if len(vectors) < 2: # t-SNE and plotting require at least 2 points
        logger.warning("Not enough data points (need at least 2) for t-SNE visualization.")
        return None

    X_np = np.array(vectors)
    
    try:
        # Ensure perplexity is less than n_samples for t-SNE
        tsne_perplexity = min(30, len(X_np) - 1)
        if tsne_perplexity <=0: # if len(X_np) was 1, this would be 0 or less
            tsne_perplexity = 5 # a small default if only few points
            if len(X_np) <=5 : tsne_perplexity = max(1, len(X_np)-1)


        tsne = TSNE(n_components=2, random_state=42, perplexity=tsne_perplexity, n_iter=300, learning_rate='auto', init='pca')
        X_tsne = await asyncio.to_thread(tsne.fit_transform, X_np)
        
        # Create a Plotly figure
        # Using plotly express for simplicity
        # Ensure cluster_labels are strings for discrete colors in px
        labels_str = [str(label) for label in cluster_labels]

        fig = px.scatter(
            x=X_tsne[:, 0], 
            y=X_tsne[:, 1], 
            color=labels_str, # Use string labels for discrete colors
            title="Cluster Visualization (t-SNE)",
            labels={'color': 'Cluster'},
            # hover_data potentially could include item IDs or snippets if passed through
        )
        
        fig.update_layout(
            xaxis_title="t-SNE Component 1",
            yaxis_title="t-SNE Component 2",
            legend_title_text='Cluster'
        )
        
        # Convert figure to HTML string
        html_output = fig.to_html(full_html=False, include_plotlyjs='cdn')
        return html_output

    except Exception as e:
        logger.error(f"Error generating cluster visualization: {e}")
        return None

# Helper to calculate optimal K if not provided, used within perform_clustering
# This is integrated into perform_clustering now.
# async def _calculate_optimal_k(X: np.ndarray, k_range: Tuple[int, int] = (2, 10), random_seed: int = 42) -> int: ... 