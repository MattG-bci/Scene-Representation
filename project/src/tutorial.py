import pickle
import pandas as pd
import numpy as np
import mmengine
import random


path = "/home/efs/datasets/nuscenes/raw/trainval/old_nuscenes_infos_train.pkl"
small_data_path = "/home/efs/users/mateusz/data/nuscenes/nuscenes_infos_train.pkl"
data = mmengine.load(path)
small_data = mmengine.load(small_data_path)
print(small_data["data_list"][0].keys())


metainfo = {'categories': {'car': 0, 'truck': 1, 'trailer': 2, 'bus': 3, 'construction_vehicle': 4, 
            'bicycle': 5, 'motorcycle': 6, 'pedestrian': 7, 'traffic_cone': 8, 'barrier': 9}, 'dataset': 'nuscenes', 'version': 'v1.0-trainval', 'info_version': '1.1'}

data = dict(data_list=data, metainfo=metainfo)
print(data["data_list"][0].keys())

