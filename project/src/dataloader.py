import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision import transforms
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import os
from PIL import Image

class NuScenesDataset(Dataset):
    def __init__(self, data_path, sensors, split="train", transform=None):
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=False)
        self.data_path = data_path
        self.sensors = sensors 
        self.dataset = []
        self.split = split
        self.scene_names = self._get_scenes(os.path.join(self.data_path, split + ".txt"))
        self._initialise_database()

        if transform is None:
            self.transform = transforms.Compose([transforms.PILToTensor(), 
                                                transforms.ConvertImageDtype(torch.float32)])
        else:
            self.transform = transform

    def _initialise_database(self):
        n_scenes = self.nusc.scene
        for scene in n_scenes:
            scene_name = scene["name"]
            if scene_name not in self.scene_names:
                continue
            n_samples = scene["nbr_samples"]
            sample = None
            for _ in range(n_samples):
                if sample is None:
                    sample_token = scene["first_sample_token"]
                    sample = self.nusc.get("sample", sample_token)
            
                else:
                    sample_token = sample["next"]
                    sample = self.nusc.get("sample", sample_token)
                
                data_point = []
                for sensor in self.sensors:
                    data = self.nusc.get("sample_data", sample["data"][sensor])
                    file_name = data["filename"]
                    data_point.append(os.path.join(self.data_path, file_name))
                
                self.dataset.append(data_point)
    
    def _get_scenes(self, text_path):
        assert os.path.exists(text_path), \
            f"A path to a text file with scenes for the {self.split} set was not provided."

        f = open(text_path, "r")
        scene_names = f.read()
        scene_names = scene_names.replace("\n", " ").split(" ")
        return scene_names[:-1]
                 
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_point = self.dataset[index]
        img_path = data_point[0]
        img = self.transform(Image.open(img_path))
        return img


if __name__ == "__main__":
    SENSORS = ["CAM_FRONT", "LIDAR_TOP"]

    data_root = "/home/efs/users/mateusz/data/nuscenes_tiny/v1.0-trainval"
    dataset = NuScenesDataset(data_root, sensors=SENSORS, split="train")
    print(len(dataset))
    print(dataset[0])
