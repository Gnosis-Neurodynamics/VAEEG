import json
import os
import random

import numpy as np
from joblib import Parallel, delayed
from lighten.utils import get_files
from tqdm import tqdm


def merge_data(input_paths, out_dir):
    band_names = ["whole", "delta", "theta", "alpha",  "low_beta", "high_beta"]

    data = Parallel(n_jobs=8)(delayed(np.load)(f) for f in tqdm(input_paths))
    data = np.concatenate(data, axis=0)
    np.random.shuffle(data)

    for i, name in enumerate(band_names):
        print("save %s to %s" % (name, out_dir))
        out_file = os.path.join(out_dir, name + ".npy")
        sx = data[:, i, :]
        np.save(out_file, sx)


if __name__ == "__main__":
    base_dir = "./new_data/"
    ratio = 0.1
    files = get_files(os.path.join(base_dir, "clips"), [".npy"])
    path_file = os.path.join(base_dir, "dataset_paths.json")

    random.shuffle(files)

    n_train = int((1.0 - ratio) * len(files))
    n_test = len(files) - n_train

    train_paths = files[0:n_train]
    test_paths = files[n_train:]

    # save paths
    with open(path_file, "w") as fo:
        json.dump({"train": train_paths, "test": test_paths}, fp=fo, indent=1)

    # read test
    test_dir = os.path.join(base_dir, "test")
    if not os.path.isdir(test_dir):
        os.makedirs(test_dir)

    merge_data(test_paths, test_dir)

    # read train
    train_dir = os.path.join(base_dir, "train")
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)

    merge_data(train_paths, train_dir)

