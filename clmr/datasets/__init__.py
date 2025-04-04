import os
from .dataset import Dataset
from .audio import AUDIO
from .magnatagatune import MAGNATAGATUNE


def get_dataset(dataset, dataset_dir, subset, download=True):

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    if dataset == "audio":
        d = AUDIO(root=dataset_dir)
    elif dataset == "magnatagatune":
        d = MAGNATAGATUNE(root=dataset_dir, download=download, subset=subset)
    else:
        raise NotImplementedError("Dataset not implemented")
    return d
