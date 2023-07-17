import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision import transforms
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import os
from nuscenes.utils.splits import create_splits_scenes
from PIL import Image
import numpy as np



NUSC = None

class NuScenesDataset(Dataset):
    def __init__(self, data_path, sensors, split="train", transform=None):
        self.nusc = self._get_nuscenes_db(data_path, "v1.0-trainval") #NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=False)
        self.data_path = data_path
        self.sensors = sensors 
        self.split = split
        self.scene_names = self._get_scenes()
        self.dataset = self._initialise_database()

        if transform is None:
            self.transform = transforms.Compose([transforms.PILToTensor(), 
                                                transforms.ConvertImageDtype(torch.float32)])
        else:
            self.transform = transform

    def _initialise_database(self):
        scene_name_to_token = {}
        scene_token_to_samples = {}

        for scene in self.nusc.scene:
            scene_name = scene["name"]
            if scene_name in self.scene_names:
                scene_token = scene["token"]
                scene_name_to_token[scene_name] = scene_token
                scene_token_to_samples[scene_token] = []

        data = []
        
        for sample in self.nusc.sample:
            scene_token = sample["scene_token"]
            if scene_token in scene_token_to_samples.keys():

            #n_samples = scene["nbr_samples"]
            #sample = None
            #for _ in range(n_samples):
            #    if sample is None:
            #        sample_token = scene["first_sample_token"]
            #        sample = self.nusc.get("sample", sample_token)
            
            #    else:
            #        sample_token = sample["next"]
            #        sample = self.nusc.get("sample", sample_token)
                
                data_point = []
                for sensor in self.sensors:
                    sensor_data = self.nusc.get("sample_data", sample["data"][sensor])
                    file_name = sensor_data["filename"]
                    data_point.append(os.path.join(self.data_path, file_name))
                data.append(data_point)
        return data
    
    def _get_scenes(self):
        scenes = create_splits_scenes()
        assert self.split in scenes.keys(), \
            f"Wrong split key. Please select one of the following: {scenes.keys()}"

        return scenes[self.split]
    
    @staticmethod
    def _init_nuscenes_db(data_root, version):
        nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
        return nusc
    
    def _get_nuscenes_db(self, data_root, version):
        global NUSC
        if NUSC is None:
            NUSC = self._init_nuscenes_db(data_root, version)
        return NUSC
                 
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_point = self.dataset[index]
        img_path = data_point[0]
        img = self.transform(Image.open(img_path))
        return img


if __name__ == "__main__":
    SENSORS = ["CAM_FRONT", "LIDAR_TOP"]

    data_root = "/home/efs/users/mateusz/data/nuscenes/"
    dataset = NuScenesDataset(data_root, sensors=SENSORS, split="train")
    print(len(dataset))
    print(dataset[0])
