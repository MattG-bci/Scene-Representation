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
from extract_resnet_weights import load_backbone

warnings.filterwarnings("ignore")


def save_features(features):
    df = pd.DataFrame(data=features)
    df.to_csv("./features.csv")

def plot_imgs_from_points(dataloader, pts):
    for idx, img in enumerate(dataloader):
        if idx in pts:
            img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            img = img.cpu().detach().numpy()
            plt.imshow(img)
            plt.savefig("test.jpg")

def plot_embeddings(network, dataloader, batch_size, save_components=True):
    network.to("cuda")
    network.eval()
    all_features = np.empty((len(dataloader) * batch_size, 2048), dtype=np.float32)
    for idx, batch in enumerate(dataloader):
        batch = batch.cuda().detach()
        features = network(batch)
        features = torch.reshape(features, (-1, 2048))
        features = features.cpu().detach().numpy()
        np.append(all_features, features, axis=0)

    all_features = all_features[np.isfinite(all_features).all(axis=1)]
    tsne = TSNE(n_components=2, perplexity=10, n_iter=3000).fit_transform(all_features)
    if save_components:
        save_features(tsne)

    plt.figure(figsize=(12, 12))
    plt.grid()
    plt.scatter(tsne[:, 0], tsne[:, 1])
    for i, point in enumerate(tsne):
        plt.annotate(str(i), point)
    plt.xlabel("CP1", fontsize=16)
    plt.ylabel("CP2", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig("t-SNE.jpg")

if __name__ == "__main__":
    torch.manual_seed(1)
    torch.set_grad_enabled(False)
    backbone = torchvision.models.resnet50()
    backbone.fc = nn.Sequential()
    checkpoint_path = "/home/ubuntu/users/mateusz/Scene-Representation/project/tb_logs/DINOv1 - NuScenes/multi-view + colorjitter + grayscale/checkpoints/epoch=40-step=9020.ckpt"
    #student_network = load_backbone(checkpoint_path, DINO, backbone, "student_backbone")
    backbone.load_state_dict(torch.load("/home/ubuntu/users/mateusz/Scene-Representation/project/utils/models/backbones/weights/bdd_det_rn50_backbone.pth"))
    #torch.save(student_network.state_dict(), "/home/ubuntu/users/mateusz/Scene-Representation/project/utils/models/backbones/weights/dino_rn50.pth")

    data_root = "/home/ubuntu/users/mateusz/data/nuscenes"#_tiny/v1.0-trainval"
    SENSORS = ["CAM_FRONT"]
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize(size=200),
        transforms.CenterCrop(200),
        transforms.ConvertImageDtype(torch.float32) 
        ])

    batch_size=1
    val_dataset = NuScenesDataset(data_root, sensors=SENSORS, transform=transform, version="v1.0-mini", split="mini_train")
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4
    )

    plot_embeddings(backbone, val_dataloader, batch_size, save_components=False)
    plot_imgs_from_points(val_dataloader, pts=[1])
