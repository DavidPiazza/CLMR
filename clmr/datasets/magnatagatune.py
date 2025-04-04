import os
import warnings
import subprocess
import torch
import numpy as np
import zipfile
from collections import defaultdict
from typing import Any, Tuple, Optional
from tqdm import tqdm
import urllib.request
import hashlib

import soundfile as sf
import torchaudio

from torch import Tensor, FloatTensor
from clmr.datasets import Dataset


def download_url(url: str, root: str, filename: Optional[str] = None, hash_value: Optional[str] = None, hash_type: str = "md5") -> None:
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        hash_value (str, optional): Hash for checking the integrity of the downloaded file
        hash_type (str): Hash type, among "md5" or "sha256"
    """
    if filename is None:
        filename = os.path.basename(url)
    
    os.makedirs(root, exist_ok=True)
    filepath = os.path.join(root, filename)

    if os.path.isfile(filepath):
        if hash_value is None:
            return
        if check_hash(filepath, hash_value, hash_type):
            return
        else:
            warnings.warn(f'{filename} exists, but the hash does not match. Re-downloading.')
    
    print(f'Downloading {url} to {filepath}')
    urllib.request.urlretrieve(url, filepath)

def extract_archive(from_path: str, to_path: Optional[str] = None) -> str:
    """Extract an archive.
    Args:
        from_path (str): Path to the archive.
        to_path (str, optional): Path to extract to. If None, extract to the same directory.
    Returns:
        str: Path to the directory the archive was extracted to.
    """
    if to_path is None:
        to_path = os.path.dirname(from_path)

    with zipfile.ZipFile(from_path, 'r') as zf:
        zf.extractall(to_path)
    
    return to_path

def check_hash(filepath: str, hash_value: str, hash_type: str = "md5") -> bool:
    """Check the hash of a file.
    Args:
        filepath (str): Path to the file
        hash_value (str): Hash to validate against
        hash_type (str): Hash type, among "md5" or "sha256"
    Returns:
        bool: True if the hash matches, else False
    """
    if hash_type == "md5":
        hash_func = hashlib.md5()
    elif hash_type == "sha256":
        hash_func = hashlib.sha256()
    else:
        raise ValueError(f'Unsupported hash type: {hash_type}')

    with open(filepath, "rb") as f:
        chunk = f.read(1024 * 1024)
        while chunk:
            hash_func.update(chunk)
            chunk = f.read(1024 * 1024)
    
    return hash_func.hexdigest() == hash_value

def convert_mp3_to_wav(mp3_path: str, wav_path: str) -> None:
    """Convert an MP3 file to WAV format using torchaudio.
    Args:
        mp3_path (str): Path to the MP3 file
        wav_path (str): Path to save the WAV file
    """
    if os.path.exists(wav_path):
        return
    
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    try:
        waveform, sample_rate = torchaudio.load(mp3_path)
        torchaudio.save(wav_path, waveform, sample_rate)
    except Exception as e:
        warnings.warn(f"Error converting {mp3_path} to WAV: {str(e)}")

FOLDER_IN_ARCHIVE = "magnatagatune"
_CHECKSUMS = {
    "http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.001": "",
    "http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.002": "",
    "http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.003": "",
    "http://mi.soi.city.ac.uk/datasets/magnatagatune/annotations_final.csv": "",
    "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/binary.npy": "",
    "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/tags.npy": "",
    "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/test.npy": "",
    "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/train.npy": "",
    "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/valid.npy": "",
    "https://github.com/jordipons/musicnn-training/raw/master/data/index/mtt/train_gt_mtt.tsv": "",
    "https://github.com/jordipons/musicnn-training/raw/master/data/index/mtt/val_gt_mtt.tsv": "",
    "https://github.com/jordipons/musicnn-training/raw/master/data/index/mtt/test_gt_mtt.tsv": "",
    "https://github.com/jordipons/musicnn-training/raw/master/data/index/mtt/index_mtt.tsv": "",
}


def get_file_list(root, subset, split):
    if subset == "train":
        if split == "pons2017":
            fl = open(os.path.join(root, "train_gt_mtt.tsv")).read().splitlines()
        else:
            fl = np.load(os.path.join(root, "train.npy"))
    elif subset == "valid":
        if split == "pons2017":
            fl = open(os.path.join(root, "val_gt_mtt.tsv")).read().splitlines()
        else:
            fl = np.load(os.path.join(root, "valid.npy"))
    else:
        if split == "pons2017":
            fl = open(os.path.join(root, "test_gt_mtt.tsv")).read().splitlines()
        else:
            fl = np.load(os.path.join(root, "test.npy"))

    if split == "pons2017":
        binary = {}
        index = open(os.path.join(root, "index_mtt.tsv")).read().splitlines()
        fp_dict = {}
        for i in index:
            clip_id, fp = i.split("\t")
            fp_dict[clip_id] = fp

        for idx, f in enumerate(fl):
            clip_id, label = f.split("\t")
            fl[idx] = "{}\t{}".format(clip_id, fp_dict[clip_id])
            clip_id = int(clip_id)
            binary[clip_id] = eval(label)
    else:
        binary = np.load(os.path.join(root, "binary.npy"))

    return fl, binary


class MAGNATAGATUNE(Dataset):
    """Create a Dataset for MagnaTagATune.
    Args:
        root (str): Path to the directory where the dataset is found or downloaded.
        folder_in_archive (str, optional): The top-level directory of the dataset.
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
        subset (str, optional): Which subset of the dataset to use.
            One of ``"training"``, ``"validation"``, ``"testing"`` or ``None``.
            If ``None``, the entire dataset is used. (default: ``None``).
    """

    _ext_audio = ".wav"

    def __init__(
        self,
        root: str,
        folder_in_archive: Optional[str] = FOLDER_IN_ARCHIVE,
        download: Optional[bool] = False,
        subset: Optional[str] = None,
        split: Optional[str] = "pons2017",
    ) -> None:

        super(MAGNATAGATUNE, self).__init__(root)
        self.root = root
        self.folder_in_archive = folder_in_archive
        self.download = download
        self.subset = subset
        self.split = split

        assert subset is None or subset in ["train", "valid", "test"], (
            "When `subset` not None, it must take a value from "
            + "{'train', 'valid', 'test'}."
        )

        self._path = os.path.join(root, folder_in_archive)

        if download:
            if not os.path.isdir(self._path):
                os.makedirs(self._path)

            zip_files = []
            for url, checksum in _CHECKSUMS.items():
                target_fn = os.path.basename(url)
                target_fp = os.path.join(self._path, target_fn)
                if ".zip" in target_fp:
                    zip_files.append(target_fp)

                if not os.path.exists(target_fp):
                    download_url(
                        url,
                        self._path,
                        filename=target_fn,
                        hash_value=checksum,
                        hash_type="md5",
                    )

            if not os.path.exists(
                os.path.join(
                    self._path,
                    "f",
                    "american_bach_soloists-j_s__bach_solo_cantatas-01-bwv54__i_aria-30-59.mp3",
                )
            ):
                merged_zip = os.path.join(self._path, "mp3.zip")
                print("Merging zip files...")
                with open(merged_zip, "wb") as f:
                    for filename in zip_files:
                        with open(filename, "rb") as g:
                            f.write(g.read())

                extract_archive(merged_zip)

            # Convert MP3 files to WAV format
            print("Converting MP3 files to WAV format...")
            for root_dir, _, files in os.walk(self._path):
                for file in tqdm(files):
                    if file.endswith(".mp3"):
                        mp3_path = os.path.join(root_dir, file)
                        wav_path = os.path.join(root_dir, file.replace(".mp3", ".wav"))
                        convert_mp3_to_wav(mp3_path, wav_path)

        if not os.path.isdir(self._path):
            raise RuntimeError(
                "Dataset not found. Please use `download=True` to download it."
            )

        self.fl, self.binary = get_file_list(self._path, self.subset, self.split)
        self.n_classes = 50  # self.binary.shape[1]
        # self.audio = {}
        # for f in tqdm(self.fl):
        #     clip_id, fp = f.split("\t")
        #     if clip_id not in self.audio.keys():
        #         audio, _ = load_magnatagatune_item(fp, self._path, self._ext_audio)
        #         self.audio[clip_id] = audio

    def file_path(self, n: int) -> str:
        _, fp = self.fl[n].split("\t")
        # Convert MP3 path to WAV path
        fp = fp.replace(".mp3", ".wav")
        return os.path.join(self._path, fp)

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, label)``
        """
        clip_id, fp = self.fl[n].split("\t")
        label = self.binary[int(clip_id)]

        audio, _ = self.load(n)
        label = FloatTensor(label)
        return audio, label

    def __len__(self) -> int:
        return len(self.fl)
