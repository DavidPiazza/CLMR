import os
import argparse
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import json
import time

from clmr.datasets import get_dataset
from clmr.data import ContrastiveDataset
from clmr.evaluation import evaluate
from clmr.models import SampleCNN
from clmr.modules import ContrastiveLearning, LinearEvaluation
from clmr.utils import (
    yaml_config_hook,
    load_encoder_checkpoint,
    load_finetuner_checkpoint,
)

# Import custom pruned dataset class
from prune_dataset import PrunedMAGNATAGATUNE

# Add a simple wrapper to crop audio
from torch.utils.data import Dataset

class CropAudioDataset(Dataset):
    def __init__(self, base_dataset, length):
        self.base_dataset = base_dataset
        self.length = length

    def __getitem__(self, idx):
        audio, label = self.base_dataset[idx]
        # Ensure audio has channel dimension (assuming mono)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        elif audio.dim() > 2:
             # If more than 2 dims, take the first channel
             audio = audio[0, ...].unsqueeze(0)

        # Pad if shorter
        if audio.shape[-1] < self.length:
            pad_len = self.length - audio.shape[-1]
            # Pad last dimension (length)
            audio = torch.nn.functional.pad(audio, (0, pad_len))
        # Crop the audio (take first N samples)
        audio = audio[..., :self.length]
        return audio, label

    def __len__(self):
        return len(self.base_dataset)

def load_pruned_dataset(args):
    """Load a pruned dataset from the specified path."""
    # First, load the original dataset
    train_dataset = get_dataset(args.dataset, args.dataset_dir, subset="train")
    valid_dataset = get_dataset(args.dataset, args.dataset_dir, subset="valid")
    test_dataset = get_dataset(args.dataset, args.dataset_dir, subset="test")
    
    # Load indices to keep
    indices_path = os.path.join(args.pruned_dataset_path, "indices_to_keep.npy")
    if not os.path.exists(indices_path):
        raise FileNotFoundError(f"Indices file not found at {indices_path}")
    
    indices_to_keep = np.load(indices_path)
    
    # Create pruned dataset
    pruned_train_dataset = PrunedMAGNATAGATUNE(train_dataset, indices_to_keep)
    
    # We don't prune validation and test sets
    return pruned_train_dataset, valid_dataset, test_dataset


def evaluate_multiple_runs(args, indices_list, run_names, train_dataset, valid_dataset, test_dataset, module):
    """Evaluate multiple pruning runs and compute statistical significance."""
    results_list = []
    
    for indices, run_name in zip(indices_list, run_names):
        print(f"\nEvaluating run: {run_name}")
        print(f"Number of indices: {len(indices)}")
        
        # Create pruned dataset for this run
        pruned_train_dataset = PrunedMAGNATAGATUNE(train_dataset, indices)
        print(f"Pruned dataset size: {len(pruned_train_dataset)}")
        
        # Adjust batch size if needed to prevent index out of bounds
        adjusted_batch_size = min(args.finetuner_batch_size, len(pruned_train_dataset))
        if adjusted_batch_size != args.finetuner_batch_size:
            print(f"Warning: Adjusted batch size from {args.finetuner_batch_size} to {adjusted_batch_size} to match pruned dataset size")
        
        # Apply training transforms only to the training set
        # Assuming PrunedMAGNATAGATUNE.__getitem__ needs transforms applied after getting the item
        # Let's modify PrunedMAGNATAGATUNE or wrap it if needed
        # For now, assuming transforms can be handled by DataLoader or are part of the base dataset structure
        # Let's focus on cropping validation/test data first.

        # --- Apply cropping to validation and test datasets --- 
        valid_dataset_cropped = CropAudioDataset(valid_dataset, args.audio_length)
        test_dataset_cropped = CropAudioDataset(test_dataset, args.audio_length)
        # --- End Cropping ---

        # --- Wrap training dataset as well ---
        train_dataset_cropped = CropAudioDataset(pruned_train_dataset, args.audio_length)
        # --- End Wrapping ---

        # Create DataLoaders directly
        # Training transform is currently not applied; revisit if needed.
        train_loader = DataLoader(
            train_dataset_cropped,
            batch_size=adjusted_batch_size,
            num_workers=args.workers,
            shuffle=True,
            persistent_workers=True
        )
        
        valid_loader = DataLoader(
            valid_dataset_cropped, # Use the cropped validation dataset
            batch_size=adjusted_batch_size,
            num_workers=args.workers,
            shuffle=False,
        )
        
        # Create a separate test_loader for the final evaluation
        test_loader = DataLoader(
            test_dataset_cropped,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,  # Use multiple workers for data loading
            pin_memory=True,  # Use pinned memory for faster CPU-GPU transfers
            persistent_workers=True  # Keep workers alive between epochs
        )
        
        # Train and evaluate
        exp_dir = os.path.join(os.getcwd(), "runs", args.eval_name)
        run_dir = os.path.join(exp_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        tb_logger = TensorBoardLogger(save_dir=exp_dir, name=run_name)
        
        # Determine accelerator and devices
        if hasattr(args, 'gpus') and args.gpus is not None and args.gpus > 0 and torch.cuda.is_available():
            accelerator = 'gpu'
            devices = args.gpus
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            accelerator = 'mps'
            devices = 1
        else:
            accelerator = 'cpu'
            devices = 1
            print("Warning: GPUs or MPS not available, using CPU. Training may be slow.")

        trainer = Trainer(
            max_epochs=args.finetuner_max_epochs, 
            logger=tb_logger,
            callbacks=[
                EarlyStopping(monitor="Valid/loss", patience=10, mode="min"),
                ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename="{epoch}-{Valid/loss:.2f}",
                    monitor="Valid/loss",
                    save_last=True,
                    mode="min",
                ),
                LearningRateMonitor(logging_interval="step"),
            ],
            accelerator=accelerator,
            devices=devices,
            log_every_n_steps=10, # Add a reasonable logging frequency
            # Add other relevant trainer args if needed, checking args namespace or using defaults
            # precision=(args.precision if hasattr(args, 'precision') else 32), 
        )

        start = time.time()
        trainer.fit(module, train_loader, valid_loader)
        
        # Evaluate
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = "mps"
        elif args.gpus and torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        
        results = evaluate(
            module.encoder,
            module.model,
            test_loader, # Pass the test loader object now
            args.dataset,
            args.audio_length,
            device=device,
        )
        
        results_list.append(results)
        
        # Save results for this run
        with open(os.path.join(run_dir, "results.txt"), "w") as f:
            for k, v in results.items():
                f.write(f"{k}: {v}\n")
    
    return results_list

def compute_statistical_significance(optimized_results, random_results):
    """Compute statistical significance using t-test."""
    from scipy import stats
    
    metrics = optimized_results[0].keys()
    significance = {}
    
    for metric in metrics:
        optimized_values = [r[metric] for r in optimized_results]
        random_values = [r[metric] for r in random_results]
        
        t_stat, p_value = stats.ttest_ind(optimized_values, random_values)
        significance[metric] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05)
        }
    
    return significance

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pruned dataset")
    # Remove or comment out the deprecated line
    # parser = Trainer.add_argparse_args(parser)

    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    
    # Add pruned dataset path argument
    parser.add_argument("--pruned_dataset_path", type=str, required=True,
                        help="Path to the pruned dataset directory")
    parser.add_argument("--eval_name", type=str, default="pruned-eval",
                        help="Name for the evaluation run")
    
    # Add evaluation-specific arguments
    parser.add_argument("--use_random_baseline", type=int, default=-1,
                        help="If >= 0, use the specified random baseline run index instead of the optimized indices.")

    args = parser.parse_args()
    pl.seed_everything(args.seed)
    args.accelerator = None

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError("Checkpoint does not exist")

    train_transform = [RandomResizedCrop(n_samples=args.audio_length)]

    # ------------
    # dataloaders
    # ------------
    train_dataset, valid_dataset, test_dataset = load_pruned_dataset(args)
    
    print(f"Pruned train dataset size: {len(train_dataset)}")
    print(f"Original validation dataset size: {len(valid_dataset)}")
    print(f"Original test dataset size: {len(test_dataset)}")

    # ------------
    # encoder
    # ------------
    encoder = SampleCNN(
        strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
        supervised=args.supervised,
        out_dim=train_dataset.n_classes,
    )

    n_features = encoder.fc.in_features

    state_dict = load_encoder_checkpoint(args.checkpoint_path, train_dataset.n_classes)
    encoder.load_state_dict(state_dict)

    cl = ContrastiveLearning(args, encoder)
    cl.eval()
    cl.freeze()

    module = LinearEvaluation(
        args,
        cl.encoder,
        hidden_dim=n_features,
        output_dim=train_dataset.n_classes,
    )

    # Save results path
    results_dir = os.path.join(args.pruned_dataset_path, "evaluation")
    os.makedirs(results_dir, exist_ok=True)

    # Load all indices (optimized and random)
    indices_path = os.path.join(args.pruned_dataset_path, "indices_to_keep.npy")
    optimized_indices = np.load(indices_path)
    
    # Load random baseline indices
    random_indices_list = []
    run_idx = 0
    while True:
        random_path = os.path.join(args.pruned_dataset_path, f"random_indices_run_{run_idx}.npy")
        if not os.path.exists(random_path):
            break
        random_indices_list.append(np.load(random_path))
        run_idx += 1
    
    # Combine all indices for evaluation
    all_indices = [optimized_indices] + random_indices_list
    run_names = ["optimized"] + [f"random_run_{i}" for i in range(len(random_indices_list))]
    
    results_list = evaluate_multiple_runs(args, all_indices, run_names, train_dataset, valid_dataset, test_dataset, module)

    if args.finetuner_checkpoint_path:
        # If using pre-trained finetuner, just evaluate once
        state_dict = load_finetuner_checkpoint(args.finetuner_checkpoint_path)
        module.model.load_state_dict(state_dict)
        
        device = "cuda:0" if args.gpus else "cpu"
        results = evaluate(
            module.encoder,
            module.model,
            test_dataset,
            args.dataset,
            args.audio_length,
            device=device,
        )
        print("Results with pre-trained finetuner:")
        print(results)
        
        with open(os.path.join(results_dir, "pretrained_results.txt"), "w") as f:
            for k, v in results.items():
                f.write(f"{k}: {v}\n")
    else:
        # Compute statistical significance
        optimized_results = results_list[:1]
        random_results = results_list[1:]
        significance = compute_statistical_significance(optimized_results, random_results)
        
        # Save statistical significance results
        with open(os.path.join(results_dir, "statistical_significance.json"), "w") as f:
            json.dump(significance, f, indent=2)
        
        print("\nStatistical Significance Results:")
        for metric, stats in significance.items():
            print(f"\n{metric}:")
            print(f"  t-statistic: {stats['t_statistic']:.4f}")
            print(f"  p-value: {stats['p_value']:.4f}")
            print(f"  Significant: {stats['significant']}") 