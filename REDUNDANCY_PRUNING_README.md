# Audio Dataset Redundancy Pruning

This repository contains scripts to prune redundancy from audio datasets (specifically MagnaTagATune) using a density-based clustering method and evaluate the pruned dataset for classification tasks.

## Method

The redundancy pruning method works as follows:

1. Extracts embeddings from a pre-trained model for all samples in the dataset
2. Applies HDBSCAN clustering to partition the dataset into clusters
3. For each cluster, computes intra-cluster similarity distribution and identifies redundant samples
4. Removes redundant samples based on an optimization procedure that minimizes Jensen-Shannon divergence between the pruned distribution and a target distribution

## Requirements

Install the required dependencies:

```bash
pip install hdbscan scikit-learn scipy numpy torch tqdm
```

These are in addition to the existing CLMR requirements.

## Usage

### 1. Run the Original Baseline

First, run the linear evaluation script to establish a baseline:

```bash
python linear_evaluation.py --checkpoint_path /path/to/clmr/checkpoint.pt --dataset magnatagatune --dataset_dir ./data
```

This will train a linear classifier on the pre-trained embeddings and report metrics.

### 2. Prune the Dataset

Next, run the pruning script:

```bash
python prune_dataset.py --checkpoint_path /path/to/clmr/checkpoint.pt --dataset magnatagatune --dataset_dir ./data --min_cluster_size 20 --t 0.5 --output_dir ./pruned_data
```

Parameters:
- `--min_cluster_size`: Minimum size of clusters for HDBSCAN (default: 10)
- `--t`: Interpolation parameter for target distribution (0 for original distribution, 1 for uniform distribution)
- `--output_dir`: Directory to save pruned dataset information

The script will create a new directory in `./pruned_data` with a timestamp, containing:
- `indices_to_keep.npy`: Indices of samples kept in the pruned dataset
- `metadata.json`: Information about the pruning process, including statistics

### 3. Evaluate the Pruned Dataset

Finally, evaluate the pruned dataset:

```bash
python evaluate_pruned.py --checkpoint_path /path/to/clmr/checkpoint.pt --dataset magnatagatune --dataset_dir ./data --pruned_dataset_path ./pruned_data/pruned_magnatagatune_TIMESTAMP
```

This will train a new linear classifier on the pruned dataset and evaluate it on the test set.

Alternatively, you can evaluate using an existing finetuner checkpoint:

```bash
python evaluate_pruned.py --checkpoint_path /path/to/clmr/checkpoint.pt --finetuner_checkpoint_path /path/to/finetuner.pt --dataset magnatagatune --dataset_dir ./data --pruned_dataset_path ./pruned_data/pruned_magnatagatune_TIMESTAMP
```

## Results

Results will be saved in the `evaluation` directory within your pruned dataset directory. You can compare:

1. The original baseline results (from `linear_evaluation.py`)
2. The pruned dataset results (from `evaluate_pruned.py`)

Metrics include:
- PR-AUC and ROC-AUC for multi-label classification tasks (MagnaTagATune, MSD)
- Accuracy for single-label classification tasks (others)

## Parameters to Tune

For better pruning results, you may want to tune:

1. `--min_cluster_size`: Controls the granularity of clustering
   - Smaller values create more clusters with fewer samples each
   - Larger values create fewer clusters with more samples each

2. `--t`: Controls the aggressiveness of pruning
   - Values closer to 0 favor keeping the original distribution
   - Values closer to 1 favor a more uniform distribution (more aggressive pruning) 