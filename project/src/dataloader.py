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
        total_samples = 0
        for scene in self.nusc.scene:
            scene_name = scene["name"]
            total_samples += scene["nbr_samples"]
            if scene_name in self.scene_names:
                scene_token = scene["token"]
                scene_name_to_token[scene_name] = scene_token
                scene_token_to_samples[scene_token] = []

        data = np.empty(shape=(total_samples * len(self.sensors) + 1, 2), dtype=object)
        samples = np.empty(shape=(total_samples), dtype=object)
        for idx, sample in enumerate(self.nusc.sample):
            scene_token = sample["scene_token"]
            if scene_token in scene_token_to_samples.keys():
                samples[idx] = sample["token"]

        samples = samples[samples != np.array(None)]
        for idx, sample_token in enumerate(samples):
            sample = self.nusc.get("sample", sample_token)
            for sensor in self.sensors:
                sensor_data = self.nusc.get("sample_data", sample["data"][sensor])
                camera_file_name = sensor_data["filename"]
                corresponding_lidar_data = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
                lidar_file_name = corresponding_lidar_data["filename"]
                data[idx] = [os.path.join(self.data_path, camera_file_name), os.path.join(self.data_path, lidar_file_name)]
        data = data[data != np.array([None] * 2)]
        data = data.reshape(-1, 2)
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
    SENSORS = ["CAM_FRONT"]

    data_root = "/home/efs/users/mateusz/data/nuscenes_tiny/v1.0-trainval"
    transform = transforms.Compose([transforms.PILToTensor(),
                                    transforms.Resize(size=200), 
                                    transforms.ConvertImageDtype(torch.float32)])
    dataset = NuScenesDataset(data_root, transform=transform, sensors=SENSORS, split="mini_train")
    print(len(dataset))
    print(dataset[0].size())
