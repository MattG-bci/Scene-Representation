import pickle
import pandas as pd
import numpy as np
import mmengine
import random


random.seed(1)
pickle_path = "/home/efs/users/mateusz/data/nuscenes/nuscenes_infos_train.pkl"


def split_for_finetuning(pickle_data_path, fraction=1.0):
    data = mmengine.load(pickle_data_path)
    metainfo = data["metainfo"]
    pickle_data = data["data_list"]
    n_samples = int(np.ceil(percentage * len(pickle_data)))
    pickle_data = np.random.choice(pickle_data, n_samples)
    return dict(data_list=pickle_data, metainfo=metainfo)



if __name__ == "__main__":
    fraction=0.3
    small_data = split_for_finetuning(pickle_path, fraction=fraction)
    file = open(f"../../../data/nuscenes/small_nuscenes_infos_train_{percentage}_finetune.pkl", "wb")
    pickle.dump(small_data, file)
    file.close()


