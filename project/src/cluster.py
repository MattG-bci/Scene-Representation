import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def  cluster_embeddings(path):
    df = pd.read_csv(path)
    n_clusters = 4
    X = df.iloc[:, 1:]
    model = KMeans(n_clusters=n_clusters, n_init=10)
    labels = model.fit_predict(X)
    name = (path.split("/")[-1]).split(".")[0]
    colors = ["r", "tab:blue", "g", "orange", "tab:purple", "tab:pink", "tab:cyan", "tab:olive"]
    
    plt.plot(figsize=(12, 12))
    for n in range(n_clusters):
        target_labels = np.where(labels == n, True, False)
        plt.scatter(X.iloc[target_labels, 0].values, X.iloc[target_labels, 1].values, 
                    linewidths=0.3, edgecolors="black", label=n, color=colors[n])
    plt.grid()
    plt.legend()
    plt.xlabel("CP1")
    plt.ylabel("CP2")
    plt.title(f"{name.upper()}")
    plt.savefig(f"Clusters - {name.upper()}.jpg")
    plt.close()


if __name__ == "__main__":
    path = "/home/ubuntu/users/mateusz/Scene-Representation/project/src/output_embeddings/imagenet.csv"
    cluster_embeddings(path)