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


def jensen_shannon_divergence(p, q):
    """Calculate Jensen-Shannon divergence between two distributions."""
    # Make sure distributions sum to 1
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log(p / m + 1e-10)) + np.sum(q * np.log(q / m + 1e-10)))


def compute_pruning_objective(r, distances, t, original_dist):
    """Compute objective function for optimizing pruning fraction."""
    # Add logging for debugging
    objective_values = []
    
    # Verify r is in reasonable range
    if r <= 0.01 or r >= 0.99:
        return 100.0  # High penalty for extreme values
    
    n_samples = len(distances)
    n_keep = max(2, int(n_samples * (1 - r)))
    
    # Sort and get indices
    sorted_indices = np.argsort(distances.sum(axis=1))
    keep_indices = sorted_indices[:n_keep]
    
    # Compute distributions with better numerical stability
    pruned_distances = distances[keep_indices][:, keep_indices]
    pruned_similarities = 1 / (1 + pruned_distances + 1e-10)
    
    # Normalize with higher precision
    pruned_dist = pruned_similarities.flatten()
    pruned_sum = np.sum(pruned_dist)
    if pruned_sum < 1e-10:
        return 100.0  # Avoid division by zero
    pruned_dist = pruned_dist / pruned_sum
    
    uniform_dist = np.ones_like(pruned_dist) / len(pruned_dist)
    target_dist = (1 - t) * original_dist + t * uniform_dist
    
    # More stable JSD calculation
    jsd = jensen_shannon_divergence(pruned_dist, target_dist)
    
    # Log values
    print(f"r={r:.3f}, n_keep={n_keep}, JSD={jsd:.6f}")
    
    return jsd


def optimize_cluster_removal(embeddings_cluster, t):
    """Optimize the fraction of samples to remove from a cluster."""
    print(f"Starting optimization for cluster with {len(embeddings_cluster)} samples")
    # Compute pairwise distances
    print("Computing pairwise distances...")
    distances = pairwise_distances(embeddings_cluster, n_jobs=-1)
    
    # Compute original similarity distribution
    print("Computing original similarity distribution...")
    similarities = 1 / (1 + distances + 1e-10)
    original_dist = similarities.flatten()
    original_dist = original_dist / np.sum(original_dist)
    
    # Optimize removal fraction
    print("Optimizing removal fraction...")
    result = minimize(
        lambda r: compute_pruning_objective(r, distances, t, original_dist),
        x0=0.3,
        bounds=[(0.01, 0.99)],
        method='L-BFGS-B',  # Use L-BFGS-B for bound constraints
        options={'maxiter': 300}
    )
    
    optimal_r = result.x[0]
    print(f"Optimal removal fraction: {optimal_r:.4f}")
    
    # Determine indices to keep
    print("Determining indices to keep...")
    n_samples = len(distances)
    n_keep = max(2, int(n_samples * (1 - optimal_r)))
    
    sorted_indices = np.argsort(distances.sum(axis=1))
    keep_indices = sorted_indices[:n_keep]
    
    return keep_indices, optimal_r


def process_single_cluster(cluster_id, cluster_indices, normalized_embeddings, min_cluster_size, t):
    """Process a single cluster - for parallel execution."""
    if cluster_id == -1 or len(cluster_indices) <= min_cluster_size:
        # Keep all samples from noise cluster or small clusters
        print(f"Keeping all {len(cluster_indices)} samples from cluster {cluster_id} (noise or small cluster)")
        return cluster_indices, 0.0
    
    print(f"Processing cluster {cluster_id} with {len(cluster_indices)} samples")
    
    # For very large clusters, apply a simple sampling strategy instead of optimization
    if len(cluster_indices) > 5000:
        print(f"Large cluster detected with {len(cluster_indices)} samples. Using simplified pruning.")
        # Simple strategy: keep 50% of samples randomly
        np.random.seed(42 + cluster_id)  # for reproducibility with different seed per cluster
        keep_ratio = 0.5
        keep_count = max(min_cluster_size, int(len(cluster_indices) * keep_ratio))
        keep_indices_local = np.random.choice(len(cluster_indices), keep_count, replace=False)
        optimal_r = 1.0 - keep_ratio
    else:
        # Get embeddings for this cluster
        cluster_embeddings = normalized_embeddings[cluster_indices].numpy()
        
        # Optimize pruning for this cluster
        try:
            keep_indices_local, optimal_r = optimize_cluster_removal(cluster_embeddings, t)
        except Exception as e:
            print(f"Error in optimization for cluster {cluster_id}: {e}")
            # Fallback to a simple strategy
            keep_ratio = 0.7  # Keep 70% as a fallback
            keep_count = max(min_cluster_size, int(len(cluster_indices) * keep_ratio))
            keep_indices_local = np.arange(keep_count)  # Keep the first keep_count samples
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


def prune_dataset(embeddings, labels, min_cluster_size, min_samples, t):
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
        metric='euclidean',
        cluster_selection_method='eom',
        core_dist_n_jobs=-1  # Use all available cores
    )
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
        delayed(process_single_cluster)(cluster_id, cluster_indices, normalized_embeddings, min_cluster_size, t)
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
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"pruned_{args.dataset}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    train_dataset = get_dataset(args.dataset, args.dataset_dir, subset="train")
    valid_dataset = get_dataset(args.dataset, args.dataset_dir, subset="valid")
    test_dataset = get_dataset(args.dataset, args.dataset_dir, subset="test")
    
    # Setup dataloaders
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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    embeddings, labels = extract_embeddings(encoder, train_loader, device)
    
    # Prune dataset with optimized method
    indices_to_keep, cluster_stats, cluster_labels = prune_dataset(
        embeddings, 
        labels, 
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        t=args.t
    )
    
    # Save optimized indices and statistics
    np.save(os.path.join(output_dir, "indices_to_keep.npy"), indices_to_keep)
    np.save(os.path.join(output_dir, "cluster_labels.npy"), cluster_labels)
    
    # Generate random baseline prunings
    random_indices_list = []
    for run in range(args.n_random_runs):
        random_indices = random_baseline_pruning(cluster_labels, cluster_stats)
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