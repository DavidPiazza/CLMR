## Redundancy Pruning for Audio Datasets (Extended CLMR)

This repository builds on the work of [Spijkervet and Burgoyne (CLMR)](https://github.com/Spijkervet/CLMR) to introduce a redundancy pruning workflow for audio datasets. Our approach leverages density-based clustering and intra-cluster diversity selection to remove redundant samples, enabling more efficient and potentially more robust representation learning and evaluation.

### Method Overview
1. **Extract Embeddings:** Use a pre-trained CLMR model to extract embeddings for all samples in the dataset.
2. **Cluster with HDBSCAN:** Apply HDBSCAN clustering to partition the dataset into clusters based on embedding similarity.
3. **Intra-cluster Pruning:** For each cluster, select a diverse subset of samples using facility-location optimization or other diversity-based strategies, removing redundant examples.
4. **Save Pruned Indices:** The indices of retained samples are saved (e.g., as `indices_to_keep.npy`).
5. **Preprocess Pruned Set:** Preprocess only the retained audio files for downstream training.
6. **Train and Evaluate:** Train a CLMR model or a linear head using only the pruned dataset, and evaluate performance.

### Usage Steps

#### 1. (Optional) Establish Baseline
Train and evaluate a linear classifier on the full dataset for comparison:
```bash
python linear_evaluation.py --checkpoint_path /path/to/clmr/checkpoint.pt --dataset magnatagatune --dataset_dir ./data
```

#### 2. Prune the Dataset
Run the pruning script:
```bash
python prune_dataset.py \
  --checkpoint_path /path/to/clmr/checkpoint.pt \
  --dataset magnatagatune \
  --dataset_dir ./data \
  --min_cluster_size 75 \
  --min_samples 1 \
  --eps 0.5 \
  --selection_method dpp \
  --n_random_runs 10 \
  --output_dir ./pruned_data
```
- `--min_cluster_size`: Minimum cluster size for HDBSCAN
- `--eps`: Allowed loss of coverage inside each cluster (smaller = keep more samples)
- `--output_dir`: Where to save pruned dataset info
Key arguments
* `--min_cluster_size 75`: Minimum cluster size for the first-pass HDBSCAN clustering.
* `--min_samples 1`: HDBSCAN `min_samples` hyper-parameter.
* `--eps 0.5`: Allowed loss of facility-location coverage inside each cluster (smaller ⇒ keep more samples).
* `--selection_method dpp`: Use the k-DPP-based intra-cluster diversity selection.
* `--n_random_runs 10`: Generate ten random baseline prunings for statistical significance testing.
* `--output_dir`: Where to save pruned dataset info (indices, metadata, random baselines).

This will create a directory in `./pruned_data` with files like `indices_to_keep.npy` and `metadata.json`.

#### 3. Preprocess the Pruned Dataset
Preprocess only the retained files:
```bash
python preprocess.py --dataset magnatagatune --dataset_dir ./data --indices_file ./pruned_data/your_run/indices_to_keep.npy
```

#### 4. Train on the Pruned Dataset
Train a CLMR model using only the pruned set:
```bash
python main.py --dataset magnatagatune --dataset_dir ./data --indices_file ./pruned_data/your_run/indices_to_keep.npy
```

#### 5. Evaluate the Pruned Set
Evaluate the pruned subset **and its 10 random baselines** over **5 Monte-Carlo splits** using `evaluate_pruned.py`:

```bash
python evaluate_pruned.py \
  --checkpoint_path /path/to/clmr/checkpoint.pt \
  --dataset magnatagatune \
  --dataset_dir ./data \
  --pruned_dataset_path ./pruned_data/your_run \
  --n_splits 5
```

This script trains a linear evaluation head for each split, compares the performance of the optimized pruning against the 10 random baselines, and reports statistical significance.