import pickle
import pandas as pd
import numpy as np
import mmengine
import random
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset
from mmdet3d.registry import TRANSFORMS
from torch.utils.data import DataLoader, Dataset
import mmdet
import open3d
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import torch


class CustomNuScenesDataset(Dataset):
    def __init__(self, data_root, ann_file, transform=None):
        self.data_root = data_root
        self.ann_file = ann_file
        self.transform = transform
        self.data_prefix=dict(
            sweeps="sweeps/LIDAR_TOP",
            pts="samples/LIDAR_TOP",
            CAM_FRONT="samples/CAM_FRONT",
            CAM_FRONT_LEFT="samples/CAM_FRONT_LEFT",
            CAM_FRONT_RIGHT="samples/CAM_FRONT_RIGHT",
            CAM_BACK="samples/CAM_BACK",
            CAM_BACK_LEFT="samples/CAM_BACK_LEFT",
            CAM_BACK_RIGHT="samples/CAM_BACK_RIGHT")
        self.load_type='frame_based' # frame_based also works but mv_image_based which is used in the FCOS3D is not supported for LiDAR.
        self.modality=dict(use_lidar=True, use_camera=True)
        self.metainfo=dict(classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ])
        self.test_mode=False
        self.box_type_3d='Camera'
        self.use_valid_flag=False
        self.backend_args=None

        self.database = NuScenesDataset(data_root=self.data_root, ann_file=self.ann_file, data_prefix=self.data_prefix, pipeline=None, box_type_3d=self.box_type_3d, 
                                        load_type=self.load_type, modality=self.modality, use_valid_flag=True, metainfo=self.metainfo, test_mode=True)

        self.database = self._restructure_database(self.database)
    
    def __len__(self):
        return len(self.database)
    

    def __getitem__(self, idx):
        sample = self.database[idx]
        img = sample["img"]["img_path"]
        lidar = sample["lidar_path"]
        img = self.transform(Image.open(img))
        return sample

    
    def _restructure_database(self, database):
        restructed_database = np.array([])
        for idx, sample in enumerate(database):
            token = sample["token"]
            timestamp = sample["timestamp"]
            ego2global = sample["ego2global"]
            lidar_points = sample["lidar_points"]
            instances = sample["instances"]
            cam_instances = sample["cam_instances"]
            pts_semantic = sample["pts_semantic_mask_path"]
            num_pts_feats = sample["num_pts_feats"]
            eval_ann_info = sample["eval_ann_info"]
            box_type_3d = sample["box_type_3d"]
            box_mode_3d = sample["box_mode_3d"]
            imgs = sample["images"]
            lidar = sample["lidar_path"]
            for view in imgs.keys():
                camera_view = imgs[view]
                restructed_database = np.append(restructed_database, dict(token=token, timestamp=timestamp, ego2global=ego2global, img=camera_view, lidar_points=lidar_points, 
                instances=instances, pts_semantic_mask_path=pts_semantic, cam_instances=cam_instances, num_pts_feats=num_pts_feats, lidar_path=lidar,
                eval_ann_info=eval_ann_info, box_type_3d=box_type_3d, box_mode_3d=box_mode_3d))

        return restructed_database
        

img_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop((224, 224)),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

data_root = "/home/efs/users/mateusz/data/nuscenes"
ann_file = "nuscenes_infos_train_debugg.pkl"


custom_dataset = CustomNuScenesDataset(data_root, ann_file, transform=img_transforms)
train_dataloader = DataLoader(custom_dataset, batch_size=2)

sample = custom_dataset[0]
print(sample["cam_instances"])
#print(sample.keys())
#for batch_idx, stuff in enumerate(train_dataloader):
#    print(stuff)
#    break

