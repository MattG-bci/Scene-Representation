import pickle
import pandas as pd
import numpy as np
import mmengine
import random


def split_for_finetuning(pickle_data_path, fraction=1.0):
    data = mmengine.load(pickle_data_path)
    metainfo = data["metainfo"]
    pickle_data = data["data_list"]
    n_scenes = len(pickle_data)
    n_samples = int((n_scenes * fraction))
    pickle_data = np.random.choice(pickle_data, n_samples)
    return dict(data_list=pickle_data, metainfo=metainfo)


if __name__ == "__main__":
    random.seed(1)
    pickle_path = "/home/efs/users/mateusz/data/nuscenes/nuscenes_infos_val.pkl"
    fraction=0.006
    small_data = split_for_finetuning(pickle_path, fraction=fraction)
    file = open(f"/home/efs/users/mateusz/data/nuscenes/nuscenes_infos_val_debugg.pkl", "wb")
    pickle.dump(small_data, file)
    file.close()


