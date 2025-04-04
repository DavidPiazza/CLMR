import torch
import torch.nn as nn
import torchmetrics
from copy import deepcopy
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from typing import Tuple
from tqdm import tqdm


class LinearEvaluation(LightningModule):
    def __init__(self, args, encoder: nn.Module, hidden_dim: int, output_dim: int):
        super().__init__()
        self.save_hyperparameters(args)

        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        if self.hparams.finetuner_mlp:
            self.model = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        else:
            self.model = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim))
        self.criterion = self.configure_criterion()

        # Determine the task type and number of classes
        # Assuming multiclass based on n_classes usage elsewhere
        num_classes = output_dim
        task_type = "multiclass"
        if output_dim == 1: # Simple heuristic for binary
            task_type = "binary"
        elif output_dim > 1: # Could potentially be multilabel depending on dataset
            # If dataset implies multilabel, change task_type="multilabel"
            # For now, stick with multiclass based on output_dim > 1
            pass 

        self.accuracy = torchmetrics.Accuracy(task=task_type, num_classes=num_classes)
        self.precision_metric = torchmetrics.Precision(task=task_type, num_classes=num_classes)
        self.recall = torchmetrics.Recall(task=task_type, num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(task=task_type, num_classes=num_classes)
        self.auc = torchmetrics.AUROC(task=task_type, num_classes=num_classes)

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        preds = self._forward_representations(x, y)
        loss = self.criterion(preds, y)
        return loss, preds

    def _forward_representations(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Perform a forward pass using either the representations, or the input data (that we still)
        need to extract the represenations from using our encoder.
        """
        print(f"[LinearEvaluation] Input shape to encoder: {x.shape}")
        if x.shape[-1] == self.hidden_dim:
            h0 = x
        else:
            with torch.no_grad():
                h0 = self.encoder(x)
        return self.model(h0)

    def training_step(self, batch, _) -> Tensor:
        x, y = batch
        loss, preds = self.forward(x, y)

        # Convert target to integer tensor before calculating accuracy
        if self.hparams.dataset in ["magnatagatune", "msd"]:
            # For multi-label datasets, use argmax to get integer predictions
            y_int = torch.argmax(y, dim=1) if len(y.shape) > 1 else y.long()
        else:
            # For single-label datasets, ensure y is a long tensor
            y_int = y.long()
        
        self.log("Train/accuracy", self.accuracy(preds, y_int))
        self.log("Train/loss", loss)
        return loss

    def validation_step(self, batch, _) -> Tensor:
        x, y = batch
        loss, preds = self.forward(x, y)
        
        # Convert target to integer tensor before calculating accuracy
        if self.hparams.dataset in ["magnatagatune", "msd"]:
            # For multi-label datasets, use argmax to get integer predictions
            y_int = torch.argmax(y, dim=1) if len(y.shape) > 1 else y.long()
        else:
            # For single-label datasets, ensure y is a long tensor
            y_int = y.long()
        
        self.log("Valid/accuracy", self.accuracy(preds, y_int))
        self.log("Valid/loss", loss)
        return loss

    def configure_criterion(self) -> nn.Module:
        if self.hparams.dataset in ["magnatagatune", "msd"]:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        return criterion

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.finetuner_learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
            verbose=False,
        )
        if scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "Valid/loss",
            }
        else:
            return {"optimizer": optimizer}

    def extract_representations(self, dataloader: DataLoader) -> Dataset:

        representations = []
        ys = []
        for x, y in tqdm(dataloader):
            with torch.no_grad():
                h0 = self.encoder(x)
                representations.append(h0)
                ys.append(y)

        if len(representations) > 1:
            representations = torch.cat(representations, dim=0)
            ys = torch.cat(ys, dim=0)
        else:
            representations = representations[0]
            ys = ys[0]

        tensor_dataset = TensorDataset(representations, ys)
        return tensor_dataset
