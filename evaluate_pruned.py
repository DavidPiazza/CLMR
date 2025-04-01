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
        )
        
        valid_loader = DataLoader(
            contrastive_valid_dataset,
            batch_size=adjusted_batch_size,
            num_workers=args.workers,
            shuffle=False,
        )
        
        test_loader = DataLoader(
            contrastive_test_dataset,
            batch_size=adjusted_batch_size,
            num_workers=args.workers,
            shuffle=False,
        )
        
        # Train and evaluate
        early_stop_callback = EarlyStopping(
            monitor="Valid/loss", patience=10, verbose=False, mode="min"
        )
        
        trainer = Trainer.from_argparse_args(
            args,
            logger=TensorBoardLogger(
                "runs", name=f"CLMRv2-{run_name}-{args.dataset}"
            ),
            max_epochs=args.finetuner_max_epochs,
            callbacks=[early_stop_callback],
        )
        
        trainer.fit(module, train_loader, valid_loader)
        
        # Evaluate
        device = "cuda:0" if args.gpus else "cpu"
        results = evaluate(
            module.encoder,
            module.model,
            contrastive_test_dataset,
            args.dataset,
            args.audio_length,
            device=device,
        )
        
        results_list.append(results)
        
        # Save results for this run
        with open(os.path.join(results_dir, f"{run_name}_results.txt"), "w") as f:
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
    parser = Trainer.add_argparse_args(parser)

    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    
    # Add pruned dataset path argument
    parser.add_argument("--pruned_dataset_path", type=str, required=True,
                        help="Path to the pruned dataset directory")
    parser.add_argument("--eval_name", type=str, default="pruned-eval",
                        help="Name for the evaluation run")
    
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