import argparse
from tqdm import tqdm
from clmr.datasets import get_dataset
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="magnatagatune")
    parser.add_argument("--dataset_dir", type=str, default="./data")
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--indices_file", type=str, default=None, help="Path to numpy file of indices to preprocess (for pruned dataset)")
    args = parser.parse_args()

    train_dataset = get_dataset(args.dataset, args.dataset_dir, subset="train")
    valid_dataset = get_dataset(args.dataset, args.dataset_dir, subset="valid")
    test_dataset = get_dataset(args.dataset, args.dataset_dir, subset="test")

    if args.indices_file is not None:
        indices = np.load(args.indices_file)
        print(f"Preprocessing only {len(indices)} files from indices in {args.indices_file}")
        for i in tqdm(indices):
            train_dataset.preprocess(int(i), args.sample_rate)
    else:
        for i in tqdm(range(len(train_dataset))):
            train_dataset.preprocess(i, args.sample_rate)

        for i in tqdm(range(len(valid_dataset))):
            valid_dataset.preprocess(i, args.sample_rate)

        for i in tqdm(range(len(test_dataset))):
            test_dataset.preprocess(i, args.sample_rate)
