## Redundancy Pruning for Audio Datasets (Extended CLMR)

This repository extends the work of Spijkervet and Burgoyne (CLMR) by introducing a redundancy pruning workflow for audio datasets, specifically MagnaTagATune. Our approach leverages density-based clustering and intra-cluster diversity selection to remove redundant samples, enabling more efficient and potentially more robust representation learning and evaluation.

### Method Overview
1. **Extract Embeddings:** Use a pre-trained CLMR model to extract embeddings for all samples in the dataset.
2. **Cluster with HDBSCAN:** Apply HDBSCAN clustering to partition the dataset into clusters based on embedding similarity.
3. **Intra-cluster Pruning:** For each cluster, select a diverse subset of samples using facility-location optimization or other diversity-based strategies, removing redundant examples.
4. **Save Pruned Indices:** The indices of retained samples are saved (e.g., as `indices_to_keep.npy`).
5. **Preprocess Pruned Set:** Preprocess only the retained audio files for downstream training.
6. **Train and Evaluate:** Train a CLMR model or a linear head using only the pruned dataset, and evaluate performance as usual.

### Usage Steps

#### 1. (Optional) Establish Baseline
Train and evaluate a linear classifier on the full dataset for comparison:
```bash
python linear_evaluation.py --checkpoint_path /path/to/clmr/checkpoint.pt --dataset magnatagatune --dataset_dir ./data
```

#### 2. Prune the Dataset
Run the pruning script to generate a pruned set:
```bash
python prune_dataset.py --checkpoint_path /path/to/clmr/checkpoint.pt --dataset magnatagatune --dataset_dir ./data --min_cluster_size 20 --eps 0.05 --output_dir ./pruned_data
```
- `--min_cluster_size`: Minimum cluster size for HDBSCAN
- `--eps`: Allowed loss of coverage inside each cluster (smaller = keep more samples)
- `--output_dir`: Where to save pruned dataset info

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
Evaluate a linear head on the pruned dataset:
```bash
python linear_evaluation.py --checkpoint_path /path/to/pruned/clmr_checkpoint.pt --dataset magnatagatune --dataset_dir ./data --indices_file ./pruned_data/your_run/indices_to_keep.npy
```

### Notes
- This workflow enables direct comparison between models trained on the full and pruned datasets.
- All scripts are compatible with the `--indices_file` argument to restrict training/evaluation to the pruned subset.
- Results and checkpoints are saved as usual in the `runs/` directory.

The following command downloads MagnaTagATune, preprocesses it and starts self-supervised pre-training on 1 GPU (with 8 simultaneous CPU workers) and linear evaluation:
```
python3 preprocess.py --dataset magnatagatune

# add --workers 8 to increase the number of parallel CPU threads to speed up online data augmentations + training.
python3 main.py --dataset magnatagatune --gpus 1 --workers 8

python3 linear_evaluation.py --gpus 1 --workers 8 --checkpoint_path [path to checkpoint.pt, usually in ./runs]
```

## Pre-train on your own folder of audio files
Simply run the following command to pre-train the CLMR model on a folder containing .wav files (or .mp3 files when editing `src_ext_audio=".mp3"` in `clmr/datasets/audio.py`). You may need to convert your audio files to the correct sample rate first, before giving it to the encoder (which accepts `22,050Hz` per default).

```
python preprocess.py --dataset audio --dataset_dir ./directory_containing_audio_files

python main.py --dataset audio --dataset_dir ./directory_containing_audio_files
```

## Configuration
The configuration of training can be found in: `config/config.yaml`. I personally prefer to use files instead of long strings of arguments when configuring a run. Every entry in the config file can be overrided with the corresponding flag (e.g. `--max_epochs 500` if you would like to train with 500 epochs).

## Logging and TensorBoard
To view results in TensorBoard, run:
```
tensorboard --logdir ./runs
```
