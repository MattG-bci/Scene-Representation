import warnings
import os
import sys

import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE

sys.path.insert(0, "../")

from src.dataloader import *
from utils.models.SSL.DINO import *
from src.extract_resnet_weights import load_backbone

warnings.filterwarnings("ignore")


def save_features(features):
    df = pd.DataFrame(data=features) 
    path = "./output_embeddings/"
    if not os.path.exists(path):
        os.mkdir(path)
    df.to_csv(path + "features.csv")
    
def plot_imgs_from_points(dataloader, pts):
    path = "./output_images"
    if not os.path.exists(path):
        os.mkdir(path)
    for idx, img in enumerate(dataloader):
        if idx in pts:
            img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            img = img.cpu().detach().numpy()
            plt.imshow(img)
            plt.savefig(path + f"/{idx}.jpg")
            plt.close()

def plot_embeddings(network, dataloader, save_components=True):
    network.to("cuda")
    network.eval()
    all_features = []
    for idx, batch in enumerate(dataloader):
        batch = batch.cuda().detach()
        features = network(batch)
        features = torch.reshape(features, (-1, 2048))
        features = features.cpu().detach().numpy()
        for feature in features:
            all_features.append(feature)

    all_features = np.array(all_features)
    tsne = TSNE(random_state=1, n_components=2, perplexity=50, n_iter=3000).fit_transform(all_features)
    if save_components:
        save_features(tsne)

    plt.figure(figsize=(12, 12))
    plt.grid()
    plt.scatter(tsne[:, 0], tsne[:, 1])
    #for i, point in enumerate(tsne):
    #    if i % 19 == 0:
    #        plt.annotate(str(i), point)
    plt.xlabel("CP1", fontsize=16)
    plt.ylabel("CP2", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig("t-SNE.jpg")
    plt.close()
    

if __name__ == "__main__":
    torch.manual_seed(1)
    np.random.seed(1)
    torch.set_grad_enabled(False)
    backbone = torchvision.models.resnet50()
    backbone.fc = nn.Sequential()
    backbone.load_state_dict(torch.load("/home/ubuntu/users/mateusz/Scene-Representation/project/utils/models/backbones/weights/carnet_depth_rn50_backbone.pth"))
    
    data_root = "/home/ubuntu/users/mateusz/data/nuscenes"
    SENSORS = ["CAM_FRONT"]
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize(size=200),
        transforms.ConvertImageDtype(torch.float32) 
        ])

    batch_size=2
    val_dataset = NuScenesDataset(data_root, sensors=SENSORS, transform=transform, version="v1.0-trainval", split="val")
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4
    )
    
    pts = [i for i in range(len(val_dataset)) if i % 19 == 0]

    plot_embeddings(backbone, val_dataloader, save_components=True)
    #plot_imgs_from_points(val_dataloader, 
    #                      pts=pts)
    
