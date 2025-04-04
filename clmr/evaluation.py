import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
from sklearn import metrics
import numpy as np
from torch import Tensor


def calculate_overall_metrics(predictions, labels):
    """Calculate overall metrics like AUC_ROC and PR_AUC."""
    roc_auc = metrics.roc_auc_score(labels, predictions, average="macro")
    pr_auc = metrics.average_precision_score(labels, predictions, average="macro")
    return {"AUC_ROC": roc_auc, "PR_AUC": pr_auc}


def evaluate(
    encoder: nn.Module,
    finetuned_head: nn.Module,
    test_loader: DataLoader,
    dataset_name: str,
    audio_length: int,
    device="cpu",
) -> dict:
    encoder.eval()
    finetuned_head.eval()
    encoder.to(device)
    finetuned_head.to(device)

    # Enable mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    all_predictions = []
    all_labels = []

    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch_x, batch_y in tqdm(test_loader, desc="Evaluate", leave=False):
            batch_x = batch_x.to(device, non_blocking=True)  # Use non-blocking transfers
            batch_y = batch_y.to(device, non_blocking=True)

            output = encoder(batch_x)
            if output.dim() > 2:  # Handle potential extra dimensions from SampleCNN reshape
                output = output.reshape(batch_x.shape[0], -1)

            if finetuned_head:
                output = finetuned_head(output)

            # we always return logits, so we need a sigmoid here for multi-label classification
            if dataset_name in ["magnatagatune", "msd"]:
                output = torch.sigmoid(output)
            else:
                output = F.softmax(output, dim=1)

            # Store per-sample predictions, not the batch mean
            # track_prediction = output.mean(dim=0)
            all_predictions.append(output.cpu().numpy()) 
            all_labels.append(batch_y.cpu().numpy())

    # Concatenate results from all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate metrics
    results = calculate_overall_metrics(all_predictions, all_labels)

    # TODO: Add per-tag metrics if needed
    # tags = get_tags_for_dataset(dataset_name) # Need a function to get tag names
    # if tags:
    #     per_tag_results = calculate_per_tag_metrics(all_predictions, all_labels, tags)
    #     results.update(per_tag_results)

    encoder.cpu()
    finetuned_head.cpu()
    return results


# Placeholder for potential per-tag metric calculation
# def calculate_per_tag_metrics(predictions, labels, tags):
#     # ... implementation ...
#     pass


# Placeholder for getting tags
# def get_tags_for_dataset(dataset_name):
#     # ... implementation ...
#     pass
