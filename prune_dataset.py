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
        
    def __getitem__(self, idx):
        original_idx = self.indices_to_keep[idx]
        return self.original_dataset[original_idx]
    
    def __len__(self):
        return len(self.indices_to_keep)
    
    def load(self, idx):
        original_idx = self.indices_to_keep[idx]
        return self.original_dataset.load(original_idx)
    
    def file_path(self, idx):
        original_idx = self.indices_to_keep[idx]
        return self.original_dataset.file_path(original_idx)
    
    def target_file_path(self, idx):
        original_idx = self.indices_to_keep[idx]
        return self.original_dataset.target_file_path(original_idx)
    
    def preprocess(self, idx, sample_rate):
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
    # Use r to determine how many samples to keep
    if r <= 0:
        return 100.0  # Penalize removing no samples
    
    n_samples = len(distances)
    n_keep = max(2, int(n_samples * (1 - r)))  # Keep at least 2 samples
    
    # Sort distances and keep n_keep closest samples
    sorted_indices = np.argsort(distances.sum(axis=1))
    keep_indices = sorted_indices[:n_keep]
    
    # Compute pruned similarity distribution
    pruned_distances = distances[keep_indices][:, keep_indices]
    pruned_similarities = 1 / (1 + pruned_distances + 1e-10)
    pruned_dist = pruned_similarities.flatten()
    pruned_dist = pruned_dist / np.sum(pruned_dist)
    
    # Create uniform distribution
    uniform_dist = np.ones_like(pruned_dist) / len(pruned_dist)
    
    # Target distribution Q = (1-t) * Q_0 + t * Q_1
    target_dist = (1 - t) * original_dist + t * uniform_dist
    
    # Calculate JSD
    jsd = jensen_shannon_divergence(pruned_dist, target_dist)
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
        x0=0.3,  # Initial guess
        bounds=[(0.01, 0.99)],  # Constrain between 1% and 99% removal
        method='L-BFGS-B',
        options={'maxiter': 100, 'ftol': 1e-5}  # Add iteration limit and tolerance
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
    return keep_indices_global, optimal_r


def prune_dataset(embeddings, labels, min_cluster_size, t):
    """Apply HDBSCAN clustering and prune each cluster."""
    print("Starting dataset pruning process...")
    # Normalize embeddings
    print("Normalizing embeddings...")
    normalized_embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
    
    # Apply HDBSCAN clustering
    print(f"Applying HDBSCAN clustering with min_cluster_size={min_cluster_size}...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=5,
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
    return np.array(all_indices_to_keep), cluster_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset pruning using redundancy removal")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    
    # Add pruning-specific arguments
    parser.add_argument("--min_cluster_size", type=int, default=10, 
                        help="Minimum cluster size for HDBSCAN")
    parser.add_argument("--t", type=float, default=0.5, 
                        help="Interpolation parameter for target distribution")
    parser.add_argument("--output_dir", type=str, default="./pruned_data",
                        help="Directory to save pruned dataset info")
    
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
    
    # Prune dataset
    indices_to_keep, cluster_stats = prune_dataset(
        embeddings, 
        labels, 
        min_cluster_size=args.min_cluster_size,
        t=args.t
    )
    
    # Save indices and statistics
    np.save(os.path.join(output_dir, "indices_to_keep.npy"), indices_to_keep)
    
    # Save metadata
    metadata = {
        "original_size": len(train_dataset),
        "pruned_size": len(indices_to_keep),
        "reduction_percentage": (1 - len(indices_to_keep) / len(train_dataset)) * 100,
        "parameters": {
            "min_cluster_size": args.min_cluster_size,
            "t": args.t
        },
        "cluster_stats": cluster_stats
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Pruned dataset information saved to {output_dir}")
    print(f"Original size: {len(train_dataset)}, Pruned size: {len(indices_to_keep)}")
    print(f"Reduction: {metadata['reduction_percentage']:.2f}%")
    
    # Create pruned dataset
    pruned_train_dataset = PrunedMAGNATAGATUNE(train_dataset, indices_to_keep)
    
    # Save pruned dataset indices for evaluation
    with open(os.path.join(output_dir, "pruned_dataset_path.txt"), "w") as f:
        f.write(output_dir) 