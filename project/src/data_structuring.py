from tkinter import Label
from typing_extensions import Self
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image
import os
import json
import cv2
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#labels_path = "path to .json files with tags and frame names"
#train_data_path = labels_path + "/bdd100k_labels_images_train.json"
#val_data_path = labels_path + "/bdd100k_labels_images_val.json"

def concatenate_data(data_path):
    df_train = pd.read_json(data_path)
    tags = [j[1][1] for j in df_train.iterrows()]
    img_names = [j[1][0] for j in df_train.iterrows()]
    tags = pd.DataFrame(tags)
    img_names = pd.DataFrame(img_names)
    concated = pd.concat([img_names, tags], axis=1)
    concated.rename(columns={0 : "name"}, inplace=True)
    return concated


def create_parquet_file(data, name):
    weather_encoder = LabelEncoder().fit(data["weather"])
    scene_encoder = LabelEncoder().fit(data["scene"])
    timeofday_encoder = LabelEncoder().fit(data["timeofday"])
    tag_dictionary = {"weather": dict(zip(range(len(weather_encoder.classes_)), weather_encoder.classes_)),
                        "scene": dict(zip(range(len(scene_encoder.classes_)), scene_encoder.classes_)),
                        "timeofday": dict(zip(range(len(timeofday_encoder.classes_)), timeofday_encoder.classes_))}
    data['weather'] = weather_encoder.transform(data['weather'])
    data['scene'] = scene_encoder.transform(data['scene'])
    data['timeofday'] = timeofday_encoder.transform(data['timeofday'])
    data['name'] = data['name'].str.replace('\\', '/', regex=True)
    data.to_parquet(name, index=False)

    with open("tags.json", "w") as f:
        json.dump(tag_dictionary, f)

    return tag_dictionary


def visualise_data_with_tags(img, labels, tags):
    label_annotations = {0: "weather", 1: "scene", 2: "timeofday"}
    fig, ax = plt.subplots()
    ax.imshow(img)
    for i in range(len(labels)):
        plt.text(100, 100 + 20*i, f"{label_annotations[i]}: {tags[label_annotations[i]][str(labels[i])]}", color="red")
    rect = patches.Rectangle((90, 80), 250, 80, fill=True, color="gray", alpha=0.7)
    ax.add_patch(rect)
    plt.show()

