import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import torch
import numpy as np

# Audio Augmentations
from torchaudio_augmentations import (
    RandomApply,
    ComposeMany,
    RandomResizedCrop,
    PolarityInversion,
    Noise,
    Gain,
    HighLowPass,
    Delay,
    PitchShift,
    Reverb,
)

from clmr.data import ContrastiveDataset
from clmr.datasets import get_dataset
from clmr.evaluation import evaluate
from clmr.models import SampleCNN
from clmr.modules import ContrastiveLearning, SupervisedLearning
from clmr.utils import yaml_config_hook
from clmr.datasets.pruned import PrunedMAGNATAGATUNE


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CLMR")
    

    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    parser.add_argument("--indices_file", type=str, default=None, help="Path to numpy file of indices to use for pruned training set")

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    # ------------
    # data augmentations
    # ------------
    if args.supervised:
        train_transform = [RandomResizedCrop(n_samples=args.audio_length)]
        num_augmented_samples = 1
    else:
        train_transform = [
            RandomResizedCrop(n_samples=args.audio_length),
            RandomApply([PolarityInversion()], p=args.transforms_polarity),
            RandomApply([Noise()], p=args.transforms_noise),
            RandomApply([Gain()], p=args.transforms_gain),
            RandomApply(
                [HighLowPass(sample_rate=args.sample_rate)], p=args.transforms_filters
            ),
            RandomApply([Delay(sample_rate=args.sample_rate)], p=args.transforms_delay),
            RandomApply(
                [
                    PitchShift(
                        n_samples=args.audio_length,
                        sample_rate=args.sample_rate,
                    )
                ],
                p=args.transforms_pitch,
            ),
            RandomApply(
                [Reverb(sample_rate=args.sample_rate)], p=args.transforms_reverb
            ),
        ]
        num_augmented_samples = 2

    # ------------
    # dataloaders
    # ------------
    train_dataset = get_dataset(args.dataset, args.dataset_dir, subset="train")
    if args.indices_file is not None:
        indices = np.load(args.indices_file)
        train_dataset = PrunedMAGNATAGATUNE(train_dataset, indices)
    valid_dataset = get_dataset(args.dataset, args.dataset_dir, subset="valid")
    contrastive_train_dataset = ContrastiveDataset(
        train_dataset,
        input_shape=(1, args.audio_length),
        transform=ComposeMany(
            train_transform, num_augmented_samples=num_augmented_samples
        ),
    )

    contrastive_valid_dataset = ContrastiveDataset(
        valid_dataset,
        input_shape=(1, args.audio_length),
        transform=ComposeMany(
            train_transform, num_augmented_samples=num_augmented_samples
        ),
    )

    train_loader = DataLoader(
        contrastive_train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
        shuffle=True,
    )

    valid_loader = DataLoader(
        contrastive_valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
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

    # ------------
    # model
    # ------------
    if args.supervised:
        module = SupervisedLearning(args, encoder, output_dim=train_dataset.n_classes)
    else:
        module = ContrastiveLearning(args, encoder)

    logger = TensorBoardLogger("runs", name="CLMRv2-{}".format(args.dataset))
    if args.checkpoint_path:
        module = module.load_from_checkpoint(
            args.checkpoint_path, encoder=encoder, output_dim=train_dataset.n_classes
        )

    else:
        # ------------
        # training
        # ------------

        if args.supervised:
            early_stopping = EarlyStopping(monitor="Valid/loss", patience=20)
        else:
            early_stopping = None

        # Configure accelerator based on available hardware
        if torch.backends.mps.is_available():
            args.accelerator = "mps"
            args.devices = 1
        elif torch.cuda.is_available():
            args.accelerator = "gpu"
        else:
            args.accelerator = "cpu"

        trainer = Trainer(
            logger=logger,
            sync_batchnorm=True,
            max_epochs=args.max_epochs,
            log_every_n_steps=10,
            check_val_every_n_epoch=1,
            accelerator=args.accelerator,
            devices=args.devices,
        )
        trainer.fit(module, train_loader, valid_loader)

    if args.supervised:
        test_dataset = get_dataset(args.dataset, args.dataset_dir, subset="test")

        contrastive_test_dataset = ContrastiveDataset(
            test_dataset,
            input_shape=(1, args.audio_length),
            transform=None,
        )

        # Use appropriate device for evaluation
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
            
        results = evaluate(
            module.encoder,
            None,
            contrastive_test_dataset,
            args.dataset,
            args.audio_length,
            device=device,
        )
        print(results)
