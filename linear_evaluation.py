import os
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchaudio_augmentations import Compose, RandomResizedCrop
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SimCLR")

    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    # Add Trainer arguments for MPS support
    # parser.add_argument("--max_epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--accelerator", type=str, default="mps", help="Accelerator to use (cpu, gpu, mps)")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to use")

    args = parser.parse_args()
    pl.seed_everything(args.seed)
    # Set accelerator and device automatically
    import torch
    if torch.backends.mps.is_available():
        args.accelerator = "mps"
        args.devices = 1
        print("Using MPS accelerator.")
    elif torch.cuda.is_available():
        args.accelerator = "cuda"
        args.devices = 1
        print("Using CUDA accelerator.")
    else:
        args.accelerator = "cpu"
        args.devices = 1
        print("Using CPU.")
    # args.accelerator is now set appropriately

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError("That checkpoint does not exist")

    train_transform = [RandomResizedCrop(n_samples=args.audio_length)]

    # ------------
    # dataloaders
    # ------------
    train_dataset = get_dataset(args.dataset, args.dataset_dir, subset="train")
    valid_dataset = get_dataset(args.dataset, args.dataset_dir, subset="valid")
    test_dataset = get_dataset(args.dataset, args.dataset_dir, subset="test")

    contrastive_train_dataset = ContrastiveDataset(
        train_dataset,
        input_shape=[1, args.audio_length],
        transform=Compose(train_transform),
    )

    contrastive_valid_dataset = ContrastiveDataset(
        valid_dataset,
        input_shape=[1, args.audio_length],
        transform=Compose(train_transform),
    )

    contrastive_test_dataset = ContrastiveDataset(
        test_dataset,
        input_shape=[1, args.audio_length],
        transform=None,
    )

    train_loader = DataLoader(
        contrastive_train_dataset,
        batch_size=args.finetuner_batch_size,
        num_workers=args.workers,
        shuffle=True,
    )

    valid_loader = DataLoader(
        contrastive_valid_dataset,
        batch_size=args.finetuner_batch_size,
        num_workers=args.workers,
        shuffle=False,
    )

    test_loader = DataLoader(
        contrastive_test_dataset,
        batch_size=args.finetuner_batch_size,
        num_workers=args.workers,
        shuffle=False,
    )

    # ------------
    # encoder
    # ------------
    encoder = SampleCNN(
        strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
        supervised=args.supervised,
        out_dim=train_dataset.n_classes,
    )

    n_features = encoder.fc.in_features  # get dimensions of last fully-connected layer

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

    train_representations_dataset = module.extract_representations(train_loader)
    train_loader = DataLoader(
        train_representations_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
    )

    valid_representations_dataset = module.extract_representations(valid_loader)
    valid_loader = DataLoader(
        valid_representations_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
    )

    if args.finetuner_checkpoint_path:
        state_dict = load_finetuner_checkpoint(args.finetuner_checkpoint_path)
        module.model.load_state_dict(state_dict)
    else:
        early_stop_callback = EarlyStopping(
            monitor="Valid/loss", patience=10, verbose=False, mode="min"
        )

        trainer = Trainer(
            accelerator=args.accelerator,
            devices=args.devices,
            logger=TensorBoardLogger(
                "runs", name="CLMRv2-eval-{}".format(args.dataset)
            ),
            max_epochs=args.max_epochs,
            callbacks=[early_stop_callback],
        )
        trainer.fit(module, train_loader, valid_loader)

    if args.accelerator == "mps":
        device = "mps"
    elif args.accelerator == "cuda":
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
    print(results)
