import warnings

import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE

from src.dataloader import *
from utils.models.SSL.Dino import *

warnings.filterwarnings("ignore")

def load_backbone(checkpoint_path, model_class, backbone):
    model = model_class.load_from_checkpoint(checkpoint_path)
    backbone = model._modules[backbone]
    return backbone

def save_features(features):
    df = pd.DataFrame(data=features)
    df.to_csv("./features.csv")

def plot_embeddings(network, dataloader, batch_size, save_components=True):
    network.eval()
    all_features = np.empty((len(dataloader) * batch_size, 2048), dtype=np.float32)
    for idx, batch in enumerate(dataloader):
        batch = batch.cuda().detach()
        features = network(batch)
        features = torch.reshape(features, (-1, 2048))
        features = features.cpu().detach().numpy()
        np.append(all_features, features, axis=0)

    all_features = all_features[np.isfinite(all_features).all(axis=1)]
    tsne = TSNE(n_components=2, perplexity=10, n_iter=5000).fit_transform(all_features)
    if save_components:
        save_features(tsne)

    plt.figure(figsize=(12, 12))
    plt.grid()
    plt.scatter(tsne[:, 0], tsne[:, 1])
    plt.xlabel("CP1", fontsize=16)
    plt.ylabel("CP2", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(f"t-SNE.jpg")

if __name__ == "__main__":
    torch.manual_seed(1)
    torch.set_grad_enabled(False)
    checkpoint_path = "/home/efs/users/mateusz/Scene-Representation/project/tb_logs/DINOv1 - NuScenes/version_3/checkpoints/epoch=2-step=660.ckpt"
    student_network = load_backbone(checkpoint_path, DINO, "student_backbone")

    data_root = "/home/efs/users/mateusz/data/nuscenes"#_tiny/v1.0-trainval"
    SENSORS = ["CAM_FRONT", "CAM_BACK"]
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize(size=200),
        transforms.CenterCrop(200),
        transforms.ConvertImageDtype(torch.float32) 
        ])

    batch_size=256
    val_dataset = NuScenesDataset(data_root, sensors=SENSORS, transform=transform, split="val")
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4
    )

    plot_embeddings(student_network, val_dataloader, batch_size, save_components=False)

