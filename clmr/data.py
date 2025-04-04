"""Wrapper for Torch Dataset class to enable contrastive training
"""
import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing import Tuple, List, Optional, Callable


class ContrastiveDataset(Dataset):
    def __init__(self, dataset: Dataset, input_shape: List[int], transform: Optional[Callable] = None):
        self.dataset = dataset
        self.transform = transform
        self.input_shape = input_shape
        self.ignore_idx = set()  # Using a set for faster lookups
        self.is_pruned = hasattr(dataset, 'indices_to_keep')
        self.max_retries = 10  # Maximum number of retries for finding a valid index

    def _get_valid_index(self, idx: int) -> int:
        """Find a valid index by trying subsequent indices."""
        original_idx = idx
        retries = 0
        
        while retries < self.max_retries:
            try:
                # Try to access the index
                audio, _ = self.dataset[idx]
                if audio.shape[1] >= self.input_shape[1]:
                    return idx
            except (IndexError, RecursionError):
                pass
            
            # Try next index
            idx = (idx + 1) % len(self.dataset)
            retries += 1
            
            # If we've wrapped around to the original index, stop
            if idx == original_idx:
                break
        
        # If we couldn't find a valid index, raise an error
        raise RuntimeError(f"Could not find valid index after {self.max_retries} retries")

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        if idx in self.ignore_idx:
            idx = self._get_valid_index((idx + 1) % len(self.dataset))

        try:
            audio, label = self.dataset[idx]
        except (IndexError, RecursionError):
            idx = self._get_valid_index((idx + 1) % len(self.dataset))
            audio, label = self.dataset[idx]

        if audio.shape[1] < self.input_shape[1]:
            self.ignore_idx.add(idx)
            idx = self._get_valid_index((idx + 1) % len(self.dataset))
            audio, label = self.dataset[idx]

        # Create two copies of the audio for contrastive learning
        if self.transform is not None:
            audio_i = self.transform(audio)
            audio_j = self.transform(audio)
            audio = torch.stack([audio_i, audio_j], dim=0)
        else:
            # If no transform is provided, just duplicate the audio
            audio = torch.stack([audio, audio], dim=0)

        return audio, label

    def __len__(self) -> int:
        return len(self.dataset)

    def concat_clip(self, n: int, audio_length: float) -> Tensor:
        audio, _ = self.dataset[n]
        batch = torch.split(audio, audio_length, dim=1)
        batch = torch.cat(batch[:-1])
        batch = batch.unsqueeze(dim=1)

        if self.transform is not None:
            batch = self.transform(batch)

        return batch
