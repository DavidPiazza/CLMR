import numpy as np
from torch.utils.data import Dataset

class PrunedMAGNATAGATUNE(Dataset):
    """Pruned version of the MagnaTagATune dataset."""
    
    def __init__(self, original_dataset, indices_to_keep):
        self.original_dataset = original_dataset
        self.indices_to_keep = indices_to_keep
        self.n_classes = original_dataset.n_classes
        self._validate_and_filter_indices()
        
    def _validate_and_filter_indices(self):
        """Validate indices and filter out any that are out of bounds."""
        max_idx = len(self.original_dataset)
        invalid_indices = [idx for idx in self.indices_to_keep if idx >= max_idx]
        
        if invalid_indices:
            print(f"Warning: Found {len(invalid_indices)} invalid indices that are out of bounds of the original dataset (size: {max_idx})")
            print("Filtering out invalid indices...")
            # Filter out invalid indices
            self.indices_to_keep = self.indices_to_keep[self.indices_to_keep < max_idx]
            print(f"Remaining valid indices: {len(self.indices_to_keep)}")
    
    def __getitem__(self, idx):
        if idx >= len(self.indices_to_keep):
            raise IndexError(f"Index {idx} is out of bounds for pruned dataset with size {len(self.indices_to_keep)}")
        
        original_idx = self.indices_to_keep[idx]
        try:
            return self.original_dataset[original_idx]
        except IndexError:
            raise IndexError(f"Original dataset index {original_idx} is out of bounds")
    
    def __len__(self):
        return len(self.indices_to_keep)
    
    def load(self, idx):
        if idx >= len(self.indices_to_keep):
            raise IndexError(f"Index {idx} is out of bounds for pruned dataset with size {len(self.indices_to_keep)}")
        
        original_idx = self.indices_to_keep[idx]
        try:
            return self.original_dataset.load(original_idx)
        except IndexError:
            raise IndexError(f"Original dataset index {original_idx} is out of bounds")
    
    def file_path(self, idx):
        if idx >= len(self.indices_to_keep):
            raise IndexError(f"Index {idx} is out of bounds for pruned dataset with size {len(self.indices_to_keep)}")
        
        original_idx = self.indices_to_keep[idx]
        try:
            return self.original_dataset.file_path(original_idx)
        except IndexError:
            raise IndexError(f"Original dataset index {original_idx} is out of bounds")
    
    def target_file_path(self, idx):
        if idx >= len(self.indices_to_keep):
            raise IndexError(f"Index {idx} is out of bounds for dataset with size {len(self.indices_to_keep)}")
        original_idx = self.indices_to_keep[idx]
        return self.original_dataset.target_file_path(original_idx)
    
    def preprocess(self, idx, sample_rate):
        if idx >= len(self.indices_to_keep):
            raise IndexError(f"Index {idx} is out of bounds for dataset with size {len(self.indices_to_keep)}")
        original_idx = self.indices_to_keep[idx]
        return self.original_dataset.preprocess(original_idx, sample_rate) 