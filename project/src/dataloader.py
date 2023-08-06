import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.splits import create_splits_scenes
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


NUSC = None
class NuScenesDataset(Dataset):
    def __init__(self, data_path, sensors, split="train", transform=None):
        self.nusc = self._get_nuscenes_db(data_path, "v1.0-mini") #NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=False)
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
            self.pc_transform = True

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


class CrossModalNuScenesDataset(Dataset):
    def __init__(self, data_path, sensors, split="train", transform=None):
        self.nusc = self._get_nuscenes_db(data_path, "v1.0-mini") #NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=False)
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
        lidar_pointcloud = LidarPointCloud.from_file(x)
        return lidar_pointcloud.points
    
    def _pc_translation(self, point_cloud, translation):
        return point_cloud + translation
    
    def _pc_scaling(self, point_cloud, scaling_factor):
        return point_cloud * scaling_factor

    def visualise_point_cloud(self, point_cloud, dim="2d"):
        assert dim in ["2d", "3d"], \
            "Dim for the visualidation not valid. Please use either \"2d\" or \"3d\"."

        x = point_cloud[0]
        y = point_cloud[1]
        z = point_cloud[2]

        fig = plt.figure()
        if dim == "2d":
            ax = fig.add_subplot(111)
            ax.scatter(x, y, s=0.1)
        else:
            ax = fig.add_subplot(projection=dim)
            ax.scatter(x, y, z, s=0.1)
            ax.set_zlabel("Z")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.savefig(f"Transformed lidar plot {dim}.jpg")

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
        lidar = self._open_lidar(data_point[1])
        img = transforms.PILToTensor()(Image.open(img_path))
        
        if self.transform:
            img, pc, img_transformed, pc_transformed = self.transform(img, lidar)
            #save_img = Image.fromarray(((img.cpu().numpy()) * 255).astype(np.uint8))
            #save_img.save("transformed_img.jpg")
            #pc = self._pc_scaling(lidar, np.array([1.2, 1.2, 1.2, 1.0]))
            #pc = self._pc_translation(pc, np.array([0.8, 0.8, 0.8, 0.0]))
            
        #self.visualise_point_cloud(pc.T, dim="3d")
        return [img, pc, img_transformed, pc_transformed]



if __name__ == "__main__":
    SENSORS = ["CAM_FRONT"]
    data_root = "/home/ubuntu/users/mateusz/data/nuscenes"
    transform = transforms.Compose([transforms.PILToTensor(),
                                    transforms.Resize(size=224),
                                    transforms.RandomAffine(degrees=0, translate=[0.1, 0.1], scale=[1.2, 1.2]), 
                                    transforms.ConvertImageDtype(torch.float32)])
    dataset = CrossModalNuScenesDataset(data_root, transform=transform, sensors=SENSORS, split="mini_train")
    print(len(dataset))
    print(dataset[0])
