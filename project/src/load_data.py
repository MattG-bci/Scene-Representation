from .data_structuring import *
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import os
import numpy as np
from PIL import Image



#X_train, X_test, y_train, y_test = train_test_split(df_train["name"].values, df_train[["weather", "scene", "timeofday"]].values, shuffle=True, random_state=1, stratify=df_train["weather"])
   
class CustomDataset(Dataset):
    def __init__(self, imgs, transforms):
        self.imgs = imgs
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        image = Image.open(self.imgs[index]).convert("RGB")
        image = self.transforms(image)
        return image


#training_data = CustomDataset(X_test, y_test)
#f = open("data_files/tags.json")
#tags = json.load(f)

#for idx in range(10): # Visualising first 10 data points for testing.
#    img, labels = training_data[idx]
#    visualise_data_with_tags(img, labels, tags)

