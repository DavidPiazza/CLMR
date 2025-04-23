import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import hdbscan
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import shutil
import json
from datetime import datetime
from joblib import Parallel, delayed

from clmr.datasets import get_dataset
from clmr.data import ContrastiveDataset
from clmr.models import SampleCNN
from clmr.modules import ContrastiveLearning, LinearEvaluation
from clmr.utils import yaml_config_hook, load_encoder_checkpoint

class PrunedMAGNATAGATUNE(Dataset):
    """Pruned version of the MagnaTagATune dataset."""
    
    def __init__(self, original_dataset, indices_to_keep):
        self.original_dataset = original_dataset
        self.indices_to_keep = indices_to_keep
        self.n_classes = original_dataset.n_classes
        self._validate_and_filter_indices()
        
    def _validate_and_filter_indices(self):
        """Validate indices and filter out any that are out of bounds."""
        max_idx = len(self.original_dataset)
        invalid_indices = [idx for idx in self.indices_to_keep if idx >= max_idx]
        
        if invalid_indices:
            print(f"Warning: Found {len(invalid_indices)} invalid indices that are out of bounds of the original dataset (size: {max_idx})")
            print("Filtering out invalid indices...")
            # Filter out invalid indices
            self.indices_to_keep = self.indices_to_keep[self.indices_to_keep < max_idx]
            print(f"Remaining valid indices: {len(self.indices_to_keep)}")
    
    def __getitem__(self, idx):
        if idx >= len(self.indices_to_keep):
            raise IndexError(f"Index {idx} is out of bounds for pruned dataset with size {len(self.indices_to_keep)}")
        
        original_idx = self.indices_to_keep[idx]
        try:
            return self.original_dataset[original_idx]
        except IndexError:
            raise IndexError(f"Original dataset index {original_idx} is out of bounds")
    
    def __len__(self):
        return len(self.indices_to_keep)
    
    def load(self, idx):
        if idx >= len(self.indices_to_keep):
            raise IndexError(f"Index {idx} is out of bounds for pruned dataset with size {len(self.indices_to_keep)}")
        
        original_idx = self.indices_to_keep[idx]
        try:
            return self.original_dataset.load(original_idx)
        except IndexError:
            raise IndexError(f"Original dataset index {original_idx} is out of bounds")
    
    def file_path(self, idx):
        if idx >= len(self.indices_to_keep):
            raise IndexError(f"Index {idx} is out of bounds for pruned dataset with size {len(self.indices_to_keep)}")
        
        original_idx = self.indices_to_keep[idx]
        try:
            return self.original_dataset.file_path(original_idx)
        except IndexError:
            raise IndexError(f"Original dataset index {original_idx} is out of bounds")
    
    def target_file_path(self, idx):
        if idx >= len(self.indices_to_keep):
            raise IndexError(f"Index {idx} is out of bounds for dataset with size {len(self.indices_to_keep)}")
        original_idx = self.indices_to_keep[idx]
        return self.original_dataset.target_file_path(original_idx)
    
    def preprocess(self, idx, sample_rate):
        if idx >= len(self.indices_to_keep):
            raise IndexError(f"Index {idx} is out of bounds for dataset with size {len(self.indices_to_keep)}")
        original_idx = self.indices_to_keep[idx]
        return self.original_dataset.preprocess(original_idx, sample_rate)


def extract_embeddings(encoder, dataloader, device):
    """Extract embeddings from the encoder."""
    encoder = encoder.to(device)
    encoder.eval()
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Extracting embeddings"):
            x = x.to(device)
            # Get the output from the sequential part, before applying FC
            features = encoder.sequential(x)
            # Reshape just like in the forward method
            features = features.reshape(x.shape[0], features.size(1) * features.size(2))
            all_embeddings.append(features.cpu())
            all_labels.append(y.cpu())
    
    return torch.cat(all_embeddings), torch.cat(all_labels)


def save_embeddings(embeddings, labels, output_path):
    """Save embeddings and labels to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        'embeddings': embeddings,
        'labels': labels
    }, output_path)
    print(f"Embeddings saved to {output_path}")


def load_embeddings(embeddings_path):
    """Load embeddings and labels from disk."""
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found at {embeddings_path}")
    
    print(f"Loading pre-extracted embeddings from {embeddings_path}")
    data = torch.load(embeddings_path)
    return data['embeddings'], data['labels']


def jensen_shannon_divergence(p, q):
    """Calculate Jensen-Shannon divergence between two distributions with improved numerical stability."""
    # Make sure distributions sum to 1 and handle zeros
    p = np.maximum(p, 1e-10)
    q = np.maximum(q, 1e-10)
    
    p_sum = np.sum(p)
    q_sum = np.sum(q)
    
    if p_sum == 0 or q_sum == 0:
        return 1.0  # Maximum divergence if either is all zeros
        
    p = p / p_sum
    q = q / q_sum
    
    m = 0.5 * (p + q)
    
    # Use a more stable calculation with explicit handling of zeros
    divergence1 = 0.0
    for p_i, m_i in zip(p, m):
        if p_i > 0 and m_i > 0:
            divergence1 += p_i * np.log(p_i / m_i)
    
    divergence2 = 0.0
    for q_i, m_i in zip(q, m):
        if q_i > 0 and m_i > 0:
            divergence2 += q_i * np.log(q_i / m_i)
    
    return 0.5 * (divergence1 + divergence2)


def precompute_distance_summaries(distances):
    """Precompute summary statistics for each sample's distances to others."""
    n_samples = len(distances)
    # For each sample, compute summary stats of its distances to all others
    summaries = np.zeros((n_samples, 5))  # 5 summary statistics
    
    for i in range(n_samples):
        dist_to_others = distances[i, :]
        summaries[i, 0] = np.mean(dist_to_others)  # mean distance
        summaries[i, 1] = np.std(dist_to_others)   # standard deviation
        summaries[i, 2] = np.min(dist_to_others[dist_to_others > 0]) if np.any(dist_to_others > 0) else 0  # min non-zero
        summaries[i, 3] = np.median(dist_to_others)  # median
        summaries[i, 4] = np.percentile(dist_to_others, 75)  # 75th percentile
    
    return summaries


def compute_pruning_objective(r, distances, t, original_summary_hist):
    """Compute objective function for optimizing pruning fraction using summary statistics."""
    # Verify r is in reasonable range
    if r <= 0.01 or r >= 0.99:
        return 100.0  # High penalty for extreme values
    
    # Ensure r is a scalar - extract first element if it's an array
    r_scalar = float(r[0]) if hasattr(r, "__len__") else float(r)
    
    n_samples = len(distances)
    n_keep = max(2, int(n_samples * (1 - r_scalar)))
    
    # Get summary stats for all samples
    sample_summaries = precompute_distance_summaries(distances)
    
    # Sort based on centrality (mean distance to others)
    # Lower mean distance = more central/representative
    sorted_indices = np.argsort(sample_summaries[:, 0])
    keep_indices = sorted_indices[:n_keep]
    
    # Get summaries of kept samples
    kept_summaries = sample_summaries[keep_indices]
    
    # Create histograms for each summary statistic
    hist_bins = 10  # Number of bins in histogram
    kept_hists = []
    
    # Create histograms for each of the 5 summary statistics
    for i in range(sample_summaries.shape[1]):
        # Get min/max across all samples for consistent binning
        min_val = np.min(sample_summaries[:, i])
        max_val = np.max(sample_summaries[:, i])
        
        # Handle edge case where min equals max
        if min_val == max_val:
            kept_hist = np.ones(hist_bins) / hist_bins
        else:
            bin_edges = np.linspace(min_val, max_val, hist_bins + 1)
            kept_hist, _ = np.histogram(kept_summaries[:, i], bins=bin_edges, density=True)
            # Ensure histogram sums to 1
            if np.sum(kept_hist) > 0:
                kept_hist = kept_hist / np.sum(kept_hist)
            else:
                kept_hist = np.ones(hist_bins) / hist_bins
        
        kept_hists.append(kept_hist)
    
    # Combine histograms into a single feature vector
    kept_hist_combined = np.concatenate(kept_hists)
    
    # Make sure both histograms have the same length before JSD calculation
    if len(kept_hist_combined) != len(original_summary_hist):
        print(f"Warning: Histogram length mismatch: {len(kept_hist_combined)} vs {len(original_summary_hist)}")
        # Pad the shorter one with zeros or truncate the longer one
        if len(kept_hist_combined) < len(original_summary_hist):
            kept_hist_combined = np.pad(kept_hist_combined, 
                                       (0, len(original_summary_hist) - len(kept_hist_combined)))
        else:
            kept_hist_combined = kept_hist_combined[:len(original_summary_hist)]
    
    # Create target distribution as interpolation between uniform and original distribution
    uniform_dist = np.ones_like(original_summary_hist) / len(original_summary_hist)
    target_dist = t * original_summary_hist + (1-t) * uniform_dist
    # Normalize to ensure it sums to 1
    target_dist = target_dist / np.sum(target_dist)
    
    # Compare with target distribution - use a more stable implementation
    try:
        jsd = jensen_shannon_divergence(kept_hist_combined, target_dist)
    except Exception as e:
        print(f"JSD calculation error: {e}")
        # Fallback to a simple squared difference
        jsd = np.sum((kept_hist_combined - target_dist)**2)
    
    # Log values for debugging - make sure to convert values to scalars for formatting
    print(f"r={float(r_scalar):.3f}, n_keep={int(n_keep)}, JSD={float(jsd):.6f}, t={float(t):.2f}")
    
    return float(jsd)  # Ensure we return a scalar


def grid_search_optimal_removal(distances, t, original_summary_hist,
                                r_min=0.4, r_max=0.90, n_points=20):
    """Find optimal removal fraction using grid search within [r_min, r_max]."""
    assert 0.0 <= r_min < r_max <= 1.0, "Invalid r_min / r_max bounds"
    grid_points = np.linspace(r_min, r_max, n_points)
    best_r = 0.4  # Default fallback
    best_score = float('inf')
    
    print("Starting grid search for optimal removal fraction...")
    results = []
    
    for r in grid_points:
        try:
            score = compute_pruning_objective(r, distances, t, original_summary_hist)
            results.append((r, score))
            if score < best_score:
                best_score = score
                best_r = r
        except Exception as e:
            print(f"Error at r={r:.2f}: {e}")
    
    # Sort and display results for clarity
    results.sort(key=lambda x: x[1])  # Sort by score
    print("\nTop 5 best removal fractions:")
    for r, score in results[:5]:
        print(f"r={r:.2f}, score={score:.6f}")
    
    print(f"\nBest removal fraction: {best_r:.2f} with score: {best_score:.6f}")
    return best_r


# ==================  Diversity‑aware sample selection  ==================
# This helper is used once the pruning fraction for a cluster has been
# decided (inter‑cluster balance via the JSD term).  It is responsible
# for picking which concrete samples are kept inside a cluster
# (intra‑cluster coverage).

def select_diverse_samples(embeddings_cluster: np.ndarray,
                           k: int,
                           method: str = "mean_distance",
                           random_state: int = 42) -> np.ndarray:
    """Select *k* diverse samples from *embeddings_cluster*.

    Parameters
    ----------
    embeddings_cluster : np.ndarray(shape=(n_samples, n_features))
        Embeddings belonging to one cluster (already on CPU / numpy).
    k : int
        Number of samples to return (``k <= n_samples``).
    method : {"mean_distance", "k_medoids", "dpp"}
        Strategy used for diversity selection.
        * ``mean_distance`` ‑ greedily picks points with the highest mean
          pairwise distance to all others (fast, no extra deps).
        * ``k_medoids`` ‑ uses the medoid indices from the k‑Medoids
          algorithm (requires ``sklearn‑extra``).
        * ``dpp`` ‑ farthest‑point heuristic that approximates k‑DPP
          sampling.
    random_state : int
        Seed used by stochastic algorithms.

    Returns
    -------
    np.ndarray(shape=(k,))
        Indices **local to the cluster** of the selected samples.
    """

    n_samples = embeddings_cluster.shape[0]
    if k >= n_samples:
        # Nothing to prune – keep everything.
        return np.arange(n_samples)

    method = method.lower()

    # Pre‑compute pairwise distances once, they are reused by several
    # strategies.  For very large clusters this could be memory hungry; in
    # that case the caller should fall back to a simpler random strategy.
    distances = pairwise_distances(embeddings_cluster, metric="euclidean")

    if method == "mean_distance":
        # Leave‑one‑out mean: exclude self‑distance (always 0) to avoid
        # compressing the score range.
        mean_dist = distances.sum(axis=1) / (distances.shape[1] - 1)
        selected = np.argsort(-mean_dist)[:k]

    elif method == "k_medoids":
        try:
            from sklearn_extra.cluster import KMedoids  # type: ignore
            kmedoids = KMedoids(
                n_clusters=k,
                metric="precomputed",
                init="k-medoids++",
                random_state=random_state,
            )
            kmedoids.fit(distances)
            selected = np.array(kmedoids.medoid_indices_)
        except ImportError:
            print(
                "sklearn_extra is not installed – falling back to "
                "'mean_distance' strategy for diversity selection."
            )
            mean_dist = np.mean(distances, axis=1)
            selected = np.argsort(-mean_dist)[:k]

    elif method == "dpp":
        # Greedy farthest‑point algorithm (approximates a k‑DPP). Start
        # with the most *distant* point to maximise subsequent spread
        first = int(np.argmax(np.mean(distances, axis=1)))
        selected = [first]

        while len(selected) < k:
            remaining = list(set(range(n_samples)) - set(selected))
            # For each remaining point compute its distance to the closest
            # already selected point.
            min_dist_to_selected = np.min(distances[remaining][:, selected], axis=1)
            # Choose the point that maximises this distance.
            next_idx = remaining[int(np.argmax(min_dist_to_selected))]
            selected.append(next_idx)

        selected = np.array(selected)

    else:
        raise ValueError(
            "Unknown diversity selection method: {}. Choose from "
            "'mean_distance', 'k_medoids', or 'dpp'.".format(method)
        )

    return selected


# ----------------------------------------------------------------------
#  Decide *how many* points to keep in a cluster (inter‑cluster balance)
#  using the JSD‑based objective from the original implementation, then
#  pass the decision to a diversity‑aware selector for the actual sample
#  choice (intra‑cluster coverage).
# ----------------------------------------------------------------------


def optimize_cluster_removal(embeddings_cluster, t, r_min, r_max,
                             selection_method="mean_distance"):
    """Find the optimal fraction of samples to remove from a cluster using grid search."""
    print(f"Starting optimization for cluster with {len(embeddings_cluster)} samples")
    
    # Compute pairwise distances
    print("Computing pairwise distances...")
    distances = pairwise_distances(embeddings_cluster, n_jobs=-1, metric='euclidean')  # Use euclidean on normalized vectors
    
    # Precompute summary statistics
    print("Computing distance summary statistics...")
    sample_summaries = precompute_distance_summaries(distances)
    
    # Create original histograms for each summary statistic
    hist_bins = 10  # Number of bins in histogram
    original_hists = []
    
    # Create histograms for each of the 5 summary statistics
    for i in range(sample_summaries.shape[1]):
        min_val = np.min(sample_summaries[:, i])
        max_val = np.max(sample_summaries[:, i])
        
        # Handle edge case where min equals max
        if min_val == max_val:
            orig_hist = np.ones(hist_bins) / hist_bins
        else:
            bin_edges = np.linspace(min_val, max_val, hist_bins + 1)
            orig_hist, _ = np.histogram(sample_summaries[:, i], bins=bin_edges, density=True)
            # Ensure histogram sums to 1
            if np.sum(orig_hist) > 0:
                orig_hist = orig_hist / np.sum(orig_hist)
            else:
                orig_hist = np.ones(hist_bins) / hist_bins
        
        original_hists.append(orig_hist)
    
    # Combine histograms into a single feature vector
    original_hist_combined = np.concatenate(original_hists)
    
    # Find optimal removal fraction using grid search instead of optimizer
    try:
        # Use grid search instead of minimize
        optimal_r = grid_search_optimal_removal(
            distances, t, original_hist_combined,
            r_min=r_min, r_max=r_max
        )
        print(f"Grid search successful. Optimal removal fraction: {optimal_r:.4f}")
    except Exception as e:
        print(f"Grid search failed: {e}")
        # Fallback to default removal fraction
        optimal_r = 0.3
        print(f"Using fallback removal fraction: {optimal_r:.4f}")
    
    # ------------------------------------------------------------------
    #  Intra‑cluster selection – choose *which* samples to keep.
    # ------------------------------------------------------------------

    print("Selecting concrete samples to keep with '{}' strategy...".format(selection_method))

    n_samples = len(embeddings_cluster)
    n_keep = max(2, int(n_samples * (1 - optimal_r)))

    keep_indices = select_diverse_samples(
        embeddings_cluster, k=n_keep, method=selection_method
    )

    return keep_indices, optimal_r


# ----------------------------------------------------------------------
#  Wrapper for parallel execution that combines the inter‑cluster JSD
#  decision with the intra‑cluster diversity selector.
# ----------------------------------------------------------------------


def process_single_cluster(
    cluster_id,
    cluster_indices,
    normalized_embeddings,
    min_cluster_size,
    r_min,
    r_max,
    t,
    selection_method="mean_distance",
):
    """Process a single cluster - for parallel execution."""
    # Treat very small clusters as indivisible, but process the noise
    # cluster (‑1) like any other to avoid over‑representing easy negatives.
    if cluster_id != -1 and len(cluster_indices) <= min_cluster_size:
        print(f"Keeping all {len(cluster_indices)} samples from small cluster {cluster_id}")
        return cluster_indices, 0.0
    
    print(f"Processing cluster {cluster_id} with {len(cluster_indices)} samples")
    
    # For very large clusters, apply a simple sampling strategy instead of optimization
    if len(cluster_indices) > 5000:
        print(f"Large cluster detected with {len(cluster_indices)} samples. Using simplified pruning.")
        # Keep a fixed ratio but still pick diverse samples within the cluster.
        # Here we prune 60 % (r = 0.6) and **keep** the remaining 40 %.
        keep_ratio = 0.4  # fraction of items to keep
        keep_count = max(min_cluster_size, int(len(cluster_indices) * keep_ratio))

        # Use the diversity‑aware selector even for the simplified path to
        # preserve intra‑cluster coverage.
        cluster_embeddings = normalized_embeddings[cluster_indices].numpy()
        keep_indices_local = select_diverse_samples(
            cluster_embeddings,
            k=keep_count,
            method=selection_method,
        )
        optimal_r = 1.0 - keep_ratio
    else:
        # Get embeddings for this cluster
        cluster_embeddings = normalized_embeddings[cluster_indices].numpy()
        
        # Optimize pruning for this cluster – obtain the number of items
        # to keep via the JSD term, then pick actual samples with the
        # requested diversity‑aware selector.
        try:
            keep_indices_local, optimal_r = optimize_cluster_removal(
                cluster_embeddings, t,
                r_min, r_max,
                selection_method=selection_method
            )
        except Exception as e:
            print(f"Error in optimization for cluster {cluster_id}: {e}")
            # Fallback to a simple strategy
            keep_ratio = 0.6  # Keep 60% as a fallback
            keep_count = max(min_cluster_size, int(len(cluster_indices) * keep_ratio))
            # Use the diversity‑aware selector instead of taking the first
            # *keep_count* samples in order to maintain coverage.
            cluster_embeddings = normalized_embeddings[cluster_indices].numpy()
            keep_indices_local = select_diverse_samples(
                cluster_embeddings,
                k=keep_count,
                method=selection_method,
            )
            optimal_r = 1.0 - keep_ratio
    
    # Map local indices back to global indices
    keep_indices_global = cluster_indices[keep_indices_local]
    
    # Validate that all indices are within bounds
    max_idx = len(normalized_embeddings)
    invalid_indices = [idx for idx in keep_indices_global if idx >= max_idx]
    if invalid_indices:
        print(f"Warning: Found {len(invalid_indices)} invalid indices in cluster {cluster_id}")
        # Filter out invalid indices
        keep_indices_global = keep_indices_global[keep_indices_global < max_idx]
    
    return keep_indices_global, optimal_r


# ----------------------------------------------------------------------
#  Main driver.  Adds `selection_method` so the user can control the
#  intra‑cluster selector from the CLI.
# ----------------------------------------------------------------------


def prune_dataset(
    embeddings,
    labels,
    min_cluster_size,
    min_samples,
    t,
    r_min,
    r_max,
    selection_method="mean_distance",
):
    """Apply HDBSCAN clustering and prune each cluster."""
    print("Starting dataset pruning process...")
    # Normalize embeddings
    print("Normalizing embeddings...")
    normalized_embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
    
    # Apply HDBSCAN clustering
    print(f"Applying HDBSCAN clustering with min_cluster_size={min_cluster_size} and min_samples={min_samples}...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',  # Using euclidean on normalized embeddings is equivalent to cosine distance
        cluster_selection_method='eom',
        core_dist_n_jobs=-1  # Use all available cores
    )
    # Using euclidean on normalized embeddings is mathematically equivalent to cosine distance
    print("Using euclidean distance on normalized vectors (mathematically equivalent to cosine distance)")
    cluster_labels = clusterer.fit_predict(normalized_embeddings.numpy())
    
    unique_clusters = np.unique(cluster_labels)
    print(f"Found {len(unique_clusters)} clusters, including noise cluster (-1)")
    
    # Process each cluster
    all_indices_to_keep = []
    cluster_stats = {}
    
    # Prepare cluster data for parallel processing
    cluster_data = []
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_data.append((cluster_id, cluster_indices))
    
    # Process clusters in parallel
    results = Parallel(n_jobs=-1)(
        delayed(process_single_cluster)(
            cluster_id,
            cluster_indices,
            normalized_embeddings,
            min_cluster_size,
            r_min,
            r_max,
            t,
            selection_method,
        )
        for cluster_id, cluster_indices in cluster_data
    )
    
    for i, (keep_indices_global, optimal_r) in enumerate(results):
        cluster_id = cluster_data[i][0]
        cluster_indices = cluster_data[i][1]
        print(f"Cluster {i+1}/{len(unique_clusters)}: Kept {len(keep_indices_global)}/{len(cluster_indices)} samples")
        all_indices_to_keep.extend(keep_indices_global)
        cluster_stats[str(cluster_id)] = {
            "original_size": int(len(cluster_indices)),
            "kept_size": int(len(keep_indices_global)),
            "removal_fraction": float(optimal_r)
        }
    
    print(f"Keeping {len(all_indices_to_keep)} samples out of {len(embeddings)}")
    
    return np.array(all_indices_to_keep), cluster_stats, cluster_labels


def random_baseline_pruning(cluster_labels, cluster_stats):
    """Perform random baseline pruning using the same reduction ratios as the optimized method."""
    all_indices_to_keep = []
    
    for cluster_id, stats in cluster_stats.items():
        cluster_id = int(cluster_id)
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if cluster_id == -1 or len(cluster_indices) <= 10:  # Keep all noise/small clusters
            all_indices_to_keep.extend(cluster_indices)
            continue
            
        # Use the same reduction ratio as the optimized method
        keep_ratio = 1.0 - stats["removal_fraction"]
        keep_count = max(10, int(len(cluster_indices) * keep_ratio))
        
        # Randomly sample indices
        np.random.seed(42 + cluster_id)  # for reproducibility
        keep_indices = np.random.choice(len(cluster_indices), keep_count, replace=False)
        all_indices_to_keep.extend(cluster_indices[keep_indices])
    
    return np.array(all_indices_to_keep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset pruning using redundancy removal")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    
    # Add pruning-specific arguments
    parser.add_argument("--min_cluster_size", type=int, default=10, 
                        help="Minimum cluster size for HDBSCAN")
    parser.add_argument("--min_samples", type=int, default=5,
                        help="Minimum number of samples in neighborhood for HDBSCAN")
    parser.add_argument("--t", type=float, default=0.5, 
                        help="Interpolation parameter for target distribution")
    parser.add_argument("--output_dir", type=str, default="./pruned_data",
                        help="Directory to save pruned dataset info")
    parser.add_argument("--n_random_runs", type=int, default=5,
                        help="Number of random baseline runs for statistical significance")
    parser.add_argument("--embeddings_path", type=str, default=None,
                        help="Path to pre-extracted embeddings (skip extraction if provided)")
    
    # --- NEW: clamp the per-cluster removal fraction -------------------
    parser.add_argument("--r_min", type=float, default=0.4,
                        help="Minimum removal fraction per cluster [0–1]")
    parser.add_argument("--r_max", type=float, default=0.90,
                        help="Maximum removal fraction per cluster [0–1, > r_min]")

    # Intra‑cluster diversity selection strategy
    parser.add_argument("--selection_method", type=str, default="mean_distance",
                        choices=["mean_distance", "k_medoids", "dpp"],
                        help="Strategy for intra-cluster diverse sample selection")
    
    # Deprecated: embedding_distance_metric used to exist for cosine/euclidean
    # distance choice. The selector now works on normalized embeddings with
    # Euclidean distance (equivalent to cosine) so we keep the argument but
    # hide it to avoid breaking existing scripts.
    parser.add_argument("--embedding_distance_metric", type=str, default="cosine",
                        help=argparse.SUPPRESS)
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"pruned_{args.dataset}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    train_dataset = get_dataset(args.dataset, args.dataset_dir, subset="train")
    valid_dataset = get_dataset(args.dataset, args.dataset_dir, subset="valid")
    test_dataset = get_dataset(args.dataset, args.dataset_dir, subset="test")
    
    # Check if embeddings path is provided
    if args.embeddings_path:
        # Load pre-extracted embeddings
        embeddings, labels = load_embeddings(args.embeddings_path)
    else:
        # Setup dataloaders for embedding extraction
        contrastive_train_dataset = ContrastiveDataset(
            train_dataset,
            input_shape=(1, args.audio_length),
            transform=None  # No transformations for embedding extraction
        )
        
        train_loader = DataLoader(
            contrastive_train_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            shuffle=False  # Don't shuffle to keep track of indices
        )
        
        # Load encoder
        encoder = SampleCNN(
            strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
            supervised=args.supervised,
            out_dim=train_dataset.n_classes,
        )
        
        state_dict = load_encoder_checkpoint(args.checkpoint_path, train_dataset.n_classes)
        encoder.load_state_dict(state_dict)
        
        # Extract embeddings
        device = "mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu"
        embeddings, labels = extract_embeddings(encoder, train_loader, device)
        
        # Save embeddings for future use
        embeddings_path = os.path.join(output_dir, "extracted_embeddings.pt")
        save_embeddings(embeddings, labels, embeddings_path)
    
    # Prune dataset with optimized method
    indices_to_keep, cluster_stats, cluster_labels = prune_dataset(
        embeddings, 
        labels, 
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        t=args.t,
        r_min=args.r_min,
        r_max=args.r_max,
        selection_method=args.selection_method
    )

    # Validate optimized indices against the loaded train_dataset size before saving
    original_indices_count_opt = len(indices_to_keep)
    indices_to_keep = np.array(indices_to_keep) # Ensure it's a numpy array for filtering
    max_valid_index = len(train_dataset)
    indices_to_keep = indices_to_keep[indices_to_keep < max_valid_index]
    filtered_count_opt = original_indices_count_opt - len(indices_to_keep)
    if filtered_count_opt > 0:
        print(f"Warning: Removed {filtered_count_opt} indices (>= {max_valid_index}) from optimized set before saving.")

    # Save optimized indices and statistics
    np.save(os.path.join(output_dir, "indices_to_keep.npy"), indices_to_keep)
    np.save(os.path.join(output_dir, "cluster_labels.npy"), cluster_labels)
    
    # Generate random baseline prunings
    random_indices_list = []
    for run in range(args.n_random_runs):
        random_indices = random_baseline_pruning(cluster_labels, cluster_stats)
        # Validate indices before saving
        original_indices_count_random = len(random_indices)
        max_valid_index = len(train_dataset) # Use the same max index as before
        random_indices = random_indices[random_indices < max_valid_index]
        filtered_count_random = original_indices_count_random - len(random_indices)
        if filtered_count_random > 0:
            print(f"Warning: Removed {filtered_count_random} indices (>= {max_valid_index}) from random run {run} set before saving.")
        random_indices_list.append(random_indices)
        np.save(os.path.join(output_dir, f"random_indices_run_{run}.npy"), random_indices)
    
    # Save metadata
    metadata = {
        "original_size": len(train_dataset),
        "pruned_size": len(indices_to_keep),
        "reduction_percentage": (1 - len(indices_to_keep) / len(train_dataset)) * 100,
        "parameters": {
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
            "t": args.t,
            "n_random_runs": args.n_random_runs
        },
        "cluster_stats": cluster_stats
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Pruned dataset information saved to {output_dir}")
    print(f"Original size: {len(train_dataset)}, Pruned size: {len(indices_to_keep)}")
    print(f"Reduction: {metadata['reduction_percentage']:.2f}%")
    print(f"Generated {args.n_random_runs} random baseline prunings")
    
    # Create pruned dataset
    pruned_train_dataset = PrunedMAGNATAGATUNE(train_dataset, indices_to_keep)
    
    # Save pruned dataset indices for evaluation
    with open(os.path.join(output_dir, "pruned_dataset_path.txt"), "w") as f:
        f.write(output_dir) 