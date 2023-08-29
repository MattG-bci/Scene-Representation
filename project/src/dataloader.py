import os
import random

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
    def __init__(self, data_path, sensors, split="train", version="v1.0-trainval", transform=None):
        self.nusc = self._get_nuscenes_db(data_path, version=version) #NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=False)
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
    def __init__(self, data_path, sensors, split="train", version="v1.0-trainval", get_max_pts=False, transform=None):
        self.nusc = self._get_nuscenes_db(data_path, version=version)
        self.data_path = data_path
        self.sensors = sensors 
        self.split = split
        self.max_pts = self._get_max_pc_points() if get_max_pts else 3871 if version == "v1.0-trainval" else 3681# already known number, set as a constant to accelerate the dataloader initalisation.
        self.scene_names = self._get_scenes()
        self.dataset = self._initialise_database()
        self.scene_ids = self.convert_scene_name_to_indices()

        self.open_img = lambda x: transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(dtype=torch.float32), transforms.Resize(220)])(Image.open(x))
        
        self.transforms = transform if transform else None
        

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
                    data.append([os.path.join(self.data_path, cam_file_name), sample["token"]])
        return data
    
    def retrieve_relevant_pc(self, sample_token):
        points_3d = self.nusc.render_pointcloud_in_image(sample_token, pointsensor_channel="LIDAR_TOP", verbose=False)
        return points_3d
    
    def _get_scenes(self):
        scenes = create_splits_scenes()
        assert self.split in scenes.keys(), \
            f"Wrong split key. Please select one of the following: {scenes.keys()}"

        return scenes[self.split]
    
    def _get_max_pc_points(self):
        samples = self.nusc.sample
        max_pts = 0
        for sample in samples:
            pc = self.retrieve_relevant_pc(sample["token"])
            if pc.shape[1] > max_pts:
                max_pts = pc.shape[1]
        return max_pts
            
    def _open_lidar(self, x):
        lidar_pointcloud = LidarPointCloud.from_file(x)
        return lidar_pointcloud.points
    
    def _pad_point_cloud(self, x):
        n_points = x.shape[1]
        n_features = x.shape[0]
        if n_points < self.max_pts:
            padding = np.zeros((n_features, self.max_pts - n_points), dtype=np.float32)
            pc_padded = np.concatenate([x.T, padding.T]).T
        else:
            pc_padded = x[:, :self.max_pts]
        return pc_padded
    
    def get_random_sample_token(self, sample_token):
        # getting a random scene
        while True:
            n_random = random.choice(self.scene_ids)
            random_scene = self.nusc.scene[n_random]
            all_samples = self.list_all_samples_of_scene(random_scene["token"])
            random_sample = random.choice(all_samples)
            if random_sample != sample_token:
                break
        return random_sample
    
    def get_neighbour_sample_token(self, sample_token, span=1.0):
        step = span / 0.5
        sample = self.nusc.get("sample", sample_token)
        neighbours = []
        count = 0
        prev = self.nusc.get("sample", sample["prev"]) if sample["prev"] != "" else ""
        next = self.nusc.get("sample", sample["next"]) if sample["next"] != "" else ""

        while count < step:
            if prev != "":
                if prev["prev"] != "":
                    prev = self.nusc.get("sample", prev["prev"])
            
            if next != "":
                if next["next"] != "":
                    next = self.nusc.get("sample", next["next"])
            count += 1
        
        if prev != "":      
            neighbours.append(prev["token"])
        
        if next != "":
            neighbours.append(next["token"])
        
        
        
        #if sample["prev"] != "":
        #    neighbour_sample = sample
        #    for _ in range(int(step)):
        #        neighbour_sample = self.nusc.get("sample", neighbour_sample["prev"])
        #    neighbours.append(neighbour_sample["token"])
        
        #if sample["next"] != "":
        #    neighbour_sample = sample
        #    for _ in range(int(step)):
        #        neighbour_sample = self.nusc.get("sample", neighbour_sample["next"])
        #    neighbours.append(neighbour_sample["token"])
        
        neighbour_sample_token = neighbours[0] #random.choice(neighbours)
        return neighbour_sample_token
         
    def list_all_samples_of_scene(self, scene_token):
        scene = self.nusc.get("scene", scene_token)
        sample = self.nusc.get("sample", scene["first_sample_token"])
        samples = [sample["token"]]
        while sample["next"] != "":
            sample = self.nusc.get("sample", sample["next"])
            samples.append(sample["token"])
        return samples
    
    def convert_scene_name_to_indices(self):
        scenes = self._get_scenes()
        scene_ids = []
        for idx, scene in enumerate(self.nusc.scene):
            scene_name = scene["name"]
            if scene_name in scenes:
                scene_ids.append(idx)
        return scene_ids
    
    def visualise_point_cloud(self, point_cloud, dim="3d", name="LiDAR"):
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
        plt.savefig(f"{name} - {dim}.jpg")
        plt.close()

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
        img_token = data_point[1]
        img = self.open_img(img_path)
        pc = self.retrieve_relevant_pc(img_token)
        pc = self._pad_point_cloud(pc)
        
        #self.visualise_point_cloud(pc, name="t + 2")
        # retrieve a point cloud from a random scene
        
        #random_img_token = self.get_random_sample_token(img_token)
        token = self.get_neighbour_sample_token(img_token, span=1.0)
        pc_random = self.retrieve_relevant_pc(token)
        self.visualise_point_cloud(pc, name="original")
        self.visualise_point_cloud(pc_random, name="neighbour")
        pc_random = self._pad_point_cloud(pc_random)
        
        # transform img
        if self.transforms:
            transformed_img = self.transforms(img)
        else:
            transformed_img = img
        
        return [img, pc.T, 1.0], [transformed_img, pc_random.T, 0.0] # adding pseudo-labels as a third entry of each array



if __name__ == "__main__":
    SENSORS = ["CAM_FRONT"]
    data_root = "/home/ubuntu/users/mateusz/data/nuscenes"
    dataset = CrossModalNuScenesDataset(data_root, sensors=SENSORS, version="v1.0-mini", split="mini_train")
    print(len(dataset))
    pair_1, pair_2 = dataset[-1]
    print(len(pair_1))
    print(pair_1[0].shape)
    print(pair_1[1].shape)
    print(pair_1[2])
