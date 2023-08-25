import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


def  cluster_embeddings(path, n_clusters, return_plot=True):
    df = pd.read_csv(path)
    X = df.iloc[:, 1:]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    model = KMeans(n_clusters=n_clusters, n_init=10)
    labels = model.fit_predict(X)
    sse = model.inertia_
    name = (path.split("/")[-1]).split(".")[0]
    colors = ["r", "tab:blue", "g", "orange", "tab:purple", "tab:pink", "tab:cyan", 
              "tab:olive", "magenta", "yellow", "gray", "slateblue", "lime", "peru"]
    
    if return_plot:
        plt.plot(figsize=(12, 12))
        for n in range(n_clusters):
            target_labels = np.where(labels == n, True, False)
            plt.scatter(X[target_labels, 0], X[target_labels, 1], 
                        linewidths=0.3, edgecolors="black", label=n, color=colors[n])
        plt.grid()
        plt.legend()
        plt.xlabel("CP1")
        plt.ylabel("CP2")
        plt.title(f"{name.upper()}")
        plt.savefig(f"Clusters - {name.upper()}.jpg")
        plt.close()
    return sse

def plot_elbow_figure(path, max_n_clusters):
    errors = []
    for n_clusters in range(2, max_n_clusters + 1):
        sse = cluster_embeddings(path, n_clusters=n_clusters, return_plot=False)
        errors.append((n_clusters, sse))
    
    plt.figure(figsize=(12, 12))
    plt.grid()
    n_clusters = [errors[i][0] for i in range(len(errors))]
    sse = [errors[i][1] for i in range(len(errors))]
    plt.plot(n_clusters, sse)
    plt.xlabel("Number of clusters")
    plt.ylabel("Sum of Squared Errors")
    plt.title("Elbow Plot")
    plt.savefig("Elbow plot.jpg")
    plt.close()


if __name__ == "__main__":
    np.random.seed(1)
    path = "/home/ubuntu/users/mateusz/Scene-Representation/project/src/output_embeddings/imagenet.csv"
    plot_elbow_figure(path, max_n_clusters=12)
    
    