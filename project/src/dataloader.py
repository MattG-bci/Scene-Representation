import os

import torch
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.splits import create_splits_scenes
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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
                for sensor in self.sensors:
                    cam_data = self.nusc.get("sample_data", sample["data"][sensor])
                    cam_file_name = cam_data["filename"]
                    lidar_data = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
                    lidar_file_name = lidar_data["filename"]
                    data.append([os.path.join(self.data_path, cam_file_name), os.path.join(self.data_path, lidar_file_name)])
        return data
    
    def _get_scenes(self):
        scenes = create_splits_scenes()
        assert self.split in scenes.keys(), \
            f"Wrong split key. Please select one of the following: {scenes.keys()}"

        return scenes[self.split]
    
    def _open_lidar(self, x):
        pass

    
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
    SENSORS = ["CAM_FRONT", "CAM_BACK"]

    data_root = "/home/efs/users/mateusz/data/nuscenes_tiny/v1.0-trainval"
    transform = transforms.Compose([transforms.PILToTensor(),
                                    transforms.Resize(size=200), 
                                    transforms.ConvertImageDtype(torch.float32)])
    dataset = NuScenesDataset(data_root, transform=transform, sensors=SENSORS, split="mini_train")
    print(len(dataset))
    print(dataset[0].size())
