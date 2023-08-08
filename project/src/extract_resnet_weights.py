import torch
import torchvision
import pickle

def extract_backbone_weights(path):
    model = torch.load(path)
    new_state_dict = {}
    for name, weight in model["state_dict"].items():
        if "backbone" in name:
            name = ".".join(name.split(".")[1:])
            new_state_dict[name] = weight

    return new_state_dict

def extract_from_pickle(path):
    file = open(path, "rb")
    data = pickle.load(file)
    new_state_dict = {}
    for name, weight in data.items():
        new_name = name.replace("model.depth_net.encoder.encoder.", "")
        new_state_dict[new_name] = weight
    return new_state_dict

def extract_features_from_pickle(path, model):
    file = open(path, "rb")
    data = pickle.load(file)
    new_state_dict = {}
    for (key, val), (key_pickle, val_pickle) in zip(model.state_dict().items(), data.items()):
        new_state_dict[key] = val_pickle
    return new_state_dict

if __name__ == "__main__":
    path =  "/home/ubuntu/users/mateusz/Scene-Representation/project/utils/models/backbones/weights/carnet_depth_resnet50.pickle"
    state_dict = extract_from_pickle(path)
    model = torchvision.models.resnet18()
    model.fc = torch.nn.Sequential()
    state_dict = extract_features_from_pickle(path, model)
    model.load_state_dict(state_dict)


