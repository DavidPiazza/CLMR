import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import hdbscan
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import json
from datetime import datetime
from joblib import Parallel, delayed
#
# ----------------------------------------------------------------------
#  Second‑pass clustering for the original noise bucket (label ‑1)
# ----------------------------------------------------------------------
def recluster_noise(indices, embeddings, min_cluster_size=30, min_samples=2):
    """
    Run HDBSCAN again on the items that were labelled ‑1 in the first pass.
    Returns a tuple (updated_labels, next_label_id).

    Items that remain noise in the second pass keep label ‑1.
    """
    if len(indices) == 0:
        return {}, -1  # nothing to do

    sub_embeddings = embeddings[indices]
    sub_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        core_dist_n_jobs=-1,
    )
    sub_labels = sub_clusterer.fit_predict(sub_embeddings)

    # Map sub‑labels 0,1,… to NEW global labels starting at current max+1
    global cluster_labels  # needed to access current global cluster labels
    label_mapping = {}
    next_global = np.max(cluster_labels) + 1
    for sub_id in np.unique(sub_labels):
        if sub_id == -1:
            label_mapping[sub_id] = -1  # keep noise
        else:
            label_mapping[sub_id] = next_global
            next_global += 1

    updated = {idx: label_mapping[sub_lab] for idx, sub_lab in zip(indices, sub_labels)}
    return updated, next_global - 1

# Threshold for dense clusters
DENSE_THRESHOLD = 500  # >500 items = dense

from clmr.datasets import get_dataset
from clmr.data import ContrastiveDataset
from clmr.models import SampleCNN
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

def expected_coverage_gain(k: int, distances: np.ndarray) -> float:
    """
    Greedy farthest-point approximation of facility-location coverage.

    Parameters
    ----------
    k : int
        Number of exemplars to keep.
    distances : ndarray, shape (n, n)
        Pair-wise distance matrix in [0, 1].

    Returns
    -------
    cov : float
        Expected coverage (1 – mean minimal distance to an exemplar).
    """
    n = distances.shape[0]
    if k >= n:
        return 1.0

    # Seed with the medoid (smallest mean distance)
    current = [int(np.argmin(distances.mean(axis=1)))]
    min_d = distances[current[0]].copy()

    while len(current) < k:
        next_idx = int(np.argmax(min_d))
        current.append(next_idx)
        min_d = np.minimum(min_d, distances[next_idx])

    return 1.0 - min_d.mean() 

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
            from sklearn_extra.cluster import KMedoids
            kmedoids = KMedoids(
                n_clusters=k,
                metric="precomputed",
                init="k-medoids++",
                random_state=random_state,
            )
            kmedoids.fit(distances)
            selected = np.array(kmedoids.medoid_indices_)
        except (ImportError, NameError) as e:
            print(f"Error using KMedoids: {str(e)}")
            print("Falling back to 'mean_distance' strategy for diversity selection.")
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

def optimize_cluster_removal(embeddings_cluster, eps,
                             selection_method="mean_distance"):
    """Choose keep-size so facility-location coverage drops ≤ eps."""

    n_samples = embeddings_cluster.shape[0]

    # Pairwise distances scaled to [0,1]
    distances = pairwise_distances(embeddings_cluster, n_jobs=-1, metric="euclidean")
    distances /= distances.max() + 1e-12

    target_cov = 1.0 - eps

    # Binary search over k
    k_low, k_high = 1, n_samples
    while k_low < k_high:
        k_mid = (k_low + k_high) // 2
        if expected_coverage_gain(k_mid, distances) >= target_cov:
            k_high = k_mid
        else:
            k_low = k_mid + 1

    k_opt = k_low
    r_opt = 1.0 - k_opt / n_samples
    print(f"Facility-location keep {k_opt}/{n_samples} (r={r_opt:.2f})")

    keep_indices = select_diverse_samples(
        embeddings_cluster, k=k_opt, method=selection_method
    )

    return keep_indices, r_opt

def process_single_cluster(
    cluster_id,
    cluster_indices,
    normalized_embeddings,
    min_cluster_size,
    eps,
    selection_method="mean_distance",
):
    """Process a single cluster - for parallel execution."""
    # Treat very small clusters as indivisible, but process the noise
    # cluster (‑1) like any other to avoid over‑representing easy negatives.
    if cluster_id != -1 and len(cluster_indices) <= min_cluster_size:
        print(f"Keeping all {len(cluster_indices)} samples from small cluster {cluster_id}")
        return cluster_indices, 0.0
    
    print(f"Processing cluster {cluster_id} with {len(cluster_indices)} samples")

    # Decide selection strategy based on density
    if len(cluster_indices) > DENSE_THRESHOLD:
        method_local = "k_medoids"      # central‑medoid for dense clusters
    else:
        method_local = selection_method # keep user choice (e.g. "dpp")

    # Get embeddings for this cluster
    cluster_embeddings = normalized_embeddings[cluster_indices].numpy()

    try:
        keep_indices_local, optimal_r = optimize_cluster_removal(
            cluster_embeddings, eps,
            selection_method=method_local
        )
    except Exception as e:
        print(f"Error in optimization for cluster {cluster_id}: {e}")
        # Fallback to a simple strategy
        keep_ratio = 0.6  # Keep 60% as a fallback
        keep_count = max(min_cluster_size, int(len(cluster_indices) * keep_ratio))
        keep_indices_local = select_diverse_samples(
            cluster_embeddings,
            k=keep_count,
            method=method_local,
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

def prune_dataset(
    embeddings,
    labels,
    min_cluster_size,
    min_samples,
    eps,
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
    global cluster_labels
    cluster_labels = clusterer.fit_predict(normalized_embeddings.numpy())

    # ---- Re‑cluster items originally labelled ‑1 ---------------------
    noise_indices = np.where(cluster_labels == -1)[0]
    if len(noise_indices) > 0:
        print(f"Re‑clustering {len(noise_indices)} noise items (label ‑1)")
        updated_map, _ = recluster_noise(
            noise_indices, normalized_embeddings.numpy(),
            min_cluster_size=max(10, min_cluster_size // 2),
            min_samples=max(2, min_samples // 2),
        )
        for idx, new_lab in updated_map.items():
            cluster_labels[idx] = new_lab

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
            eps,
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
    parser.add_argument("--eps", type=float, default=0.05,
                        help="Allowed loss of coverage inside each cluster (0–1); "
                             "smaller = keep more samples")
    parser.add_argument("--output_dir", type=str, default="./pruned_data",
                        help="Directory to save pruned dataset info")
    parser.add_argument("--n_random_runs", type=int, default=5,
                        help="Number of random baseline runs for statistical significance")
    parser.add_argument("--embeddings_path", type=str, default=None,
                        help="Path to pre-extracted embeddings (skip extraction if provided)")
    
    # Intra‑cluster diversity selection strategy
    parser.add_argument("--selection_method", type=str, default="mean_distance",
                        choices=["mean_distance", "k_medoids", "dpp"],
                        help="Strategy for intra-cluster diverse sample selection")
    
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
        eps=args.eps,
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
            "eps": args.eps,
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