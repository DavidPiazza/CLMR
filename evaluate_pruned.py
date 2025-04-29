import os
import argparse
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchaudio_augmentations import Compose, RandomResizedCrop
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import json

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
    results_list = []       # list of lists: one sub‑list per run_name
    
    for indices, run_name in zip(indices_list, run_names):
        print(f"\nEvaluating run: {run_name}")
        print(f"Number of indices: {len(indices)}")
        
        split_results = []
        for split_idx in range(args.n_splits):
            pl.seed_everything(args.seed + split_idx)
            # Create pruned dataset for this run
            pruned_train_dataset = PrunedMAGNATAGATUNE(train_dataset, indices)
            print(f"Pruned dataset size: {len(pruned_train_dataset)}")
            
            # Adjust batch size if needed to prevent index out of bounds
            adjusted_batch_size = min(args.finetuner_batch_size, len(pruned_train_dataset))
            if adjusted_batch_size != args.finetuner_batch_size:
                print(f"Warning: Adjusted batch size from {args.finetuner_batch_size} to {adjusted_batch_size} to match pruned dataset size")
            
            # Setup dataloaders
            contrastive_train_dataset = ContrastiveDataset(
                pruned_train_dataset,
                input_shape=(1, args.audio_length),
                transform=Compose(train_transform),
            )
            
            contrastive_valid_dataset = ContrastiveDataset(
                valid_dataset,
                input_shape=(1, args.audio_length),
                transform=Compose(train_transform),
            )
            
            contrastive_test_dataset = ContrastiveDataset(
                test_dataset,
                input_shape=(1, args.audio_length),
                transform=None,
            )
            
            print(f"Contrastive train dataset size: {len(contrastive_train_dataset)}")
            print(f"Contrastive valid dataset size: {len(contrastive_valid_dataset)}")
            print(f"Contrastive test dataset size: {len(contrastive_test_dataset)}")
            
            train_loader = DataLoader(
                contrastive_train_dataset,
                batch_size=adjusted_batch_size,
                num_workers=args.workers,
                shuffle=True,
                persistent_workers=True if args.workers > 0 else False,
            )
            
            valid_loader = DataLoader(
                contrastive_valid_dataset,
                batch_size=adjusted_batch_size,
                num_workers=args.workers,
                shuffle=False,
                persistent_workers=True if args.workers > 0 else False,
            )
            
            test_loader = DataLoader(
                contrastive_test_dataset,
                batch_size=adjusted_batch_size,
                num_workers=args.workers,
                shuffle=False,
                persistent_workers=True if args.workers > 0 else False,
            )
            
            # Train and evaluate
            early_stop_callback = EarlyStopping(
                monitor="Valid/loss", patience=10, verbose=False, mode="min"
            )
            
            # Check for MPS availability
            if args.use_mps and torch.backends.mps.is_available():
                print("MPS (Metal Performance Shaders) is available. Using MPS for acceleration.")
                device = torch.device("mps")
                accelerator = "mps"  # Update this to use MPS accelerator
            elif args.gpus > 0:
                if torch.cuda.is_available():
                    print(f"Using CUDA GPU acceleration with {args.gpus} GPUs.")
                    device = torch.device("cuda:0")
                    accelerator = "gpu"
                else:
                    print("CUDA GPU requested but not available. Falling back to CPU.")
                    device = torch.device("cpu")
                    accelerator = "cpu"
                    args.gpus = 0
            else:
                print("Using CPU for computation.")
                device = torch.device("cpu")
                accelerator = "cpu"
            
            trainer = Trainer(
                logger=TensorBoardLogger(
                    "runs", name=f"CLMRv2-{run_name}-{args.dataset}"
                ),
                max_epochs=args.finetuner_max_epochs,
                callbacks=[early_stop_callback],
                devices=1 if args.gpus or args.use_mps else 0,
                accelerator=accelerator,
            )
            
            trainer.fit(module, train_loader, valid_loader)
            
            # Evaluate
            if args.use_mps and torch.backends.mps.is_available():
                device = "mps"
            elif args.gpus > 0 and torch.cuda.is_available():
                device = "cuda:0"
            else:
                device = "cpu"
            
            results = evaluate(
                module.encoder,
                module.model,
                contrastive_test_dataset,
                args.dataset,
                args.audio_length,
                device=device,
            )
            
            split_results.append(results)
        
        # Compute mean of split results for logging / saving
        mean_results = {}
        for key in split_results[0].keys():
            mean_results[key] = np.mean([r[key] for r in split_results])
        
        print(f"Mean results for run {run_name}:")
        for k, v in mean_results.items():
            print(f"{k}: {v}")
        
        # Save mean results for this run
        with open(os.path.join(results_dir, f"{run_name}_results.txt"), "w") as f:
            for k, v in mean_results.items():
                f.write(f"{k}: {v}\n")
        
        results_list.append(split_results)
    
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
        
        pooled_std = np.sqrt(
            ((len(optimized_values) - 1) * np.var(optimized_values, ddof=1) +
             (len(random_values) - 1) * np.var(random_values, ddof=1))
            / (len(optimized_values) + len(random_values) - 2)
        )
        cohen_d = (np.mean(optimized_values) - np.mean(random_values)) / pooled_std
        delta_pct = 100 * (np.mean(optimized_values) - np.mean(random_values)) / np.mean(random_values)

        # 95 % confidence interval for the difference of means
        n1, n2 = len(optimized_values), len(random_values)
        mean_diff = np.mean(optimized_values) - np.mean(random_values)
        se_diff = np.sqrt(np.var(optimized_values, ddof=1)/n1 +
                          np.var(random_values, ddof=1)/n2)
        ci_low, ci_high = stats.t.interval(
            0.95, df=n1 + n2 - 2, loc=mean_diff, scale=se_diff
        )

        significance[metric] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "cohen_d": float(cohen_d),
            "delta_percent": float(delta_pct),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
        }
    
    return significance

def add_trainer_specific_args(parser):
    """Add PyTorch Lightning Trainer specific arguments to parser."""
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs to use")
    parser.add_argument("--use_mps", action="store_true", help="Use MPS (Metal Performance Shaders) for Apple Silicon")
    parser.add_argument("--precision", type=int, default=32, help="Precision for training (16, 32)")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic training")
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pruned dataset")
    
    # Load config first
    config = yaml_config_hook("./config/config.yaml")
    
    # Add trainer specific arguments
    parser = add_trainer_specific_args(parser)
    
    # Add all config arguments that aren't already defined
    added_args = set([action.dest for action in parser._actions])
    for k, v in config.items():
        if k not in added_args:
            parser.add_argument(f"--{k}", default=v, type=type(v))
    
    # Add pruned dataset path argument
    parser.add_argument("--pruned_dataset_path", type=str, required=True,
                        help="Path to the pruned dataset directory")
    parser.add_argument("--eval_name", type=str, default="pruned-eval",
                        help="Name for the evaluation run")
    parser.add_argument(
        "--n_splits",
        type=int,
        default=20,
        help="Number of Monte‑Carlo train/valid splits per pruning condition",
    )
    
    args = parser.parse_args()
    pl.seed_everything(args.seed)

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
    
    # For pretrained finetuner, create a test dataset for evaluation
    if args.finetuner_checkpoint_path:
        # Setup test dataset
        contrastive_test_dataset = ContrastiveDataset(
            test_dataset,
            input_shape=(1, args.audio_length),
            transform=None,
        )
        
        # Load pretrained finetuner weights
        state_dict = load_finetuner_checkpoint(args.finetuner_checkpoint_path)
        module.model.load_state_dict(state_dict)
        
        # Set device based on availability and user preferences
        if args.use_mps and torch.backends.mps.is_available():
            device = "mps"
        elif args.gpus and torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        
        # Move module components to the right device
        module.encoder = module.encoder.to(device)
        module.model = module.model.to(device)
        
        results = evaluate(
            module.encoder,
            module.model,
            contrastive_test_dataset,
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
        # Run full training and evaluation
        results_list = evaluate_multiple_runs(args, all_indices, run_names, train_dataset, valid_dataset, test_dataset, module)
        
        # Flatten per‑split lists so stats work transparently
        optimized_results = [item for sub in results_list[:1] for item in sub]
        random_results = [item for sub in results_list[1:] for item in sub]
        
        # Compute statistical significance
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