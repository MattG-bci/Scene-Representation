import torch
import torchvision
import pickle
import sys


def load_backbone(checkpoint_path, model_class, backbone_model, target):
    model = model_class.load_from_checkpoint(checkpoint_path, backbone=backbone_model)
    backbone = model._modules[target]
    state_dict = {}
    for (key, val), (k, v) in zip(backbone_model.state_dict().items(), backbone.state_dict().items()):
        state_dict[key] = v
    backbone_model.load_state_dict(state_dict)
    return backbone_model

def extract_weights(path, model):
    loaded_state_dict = torch.load(path)
    new_state_dict = {}
    for (key, val), (key_pickle, val_pickle) in zip(model.state_dict().items(), loaded_state_dict["state_dict"].items()):
        new_state_dict[key] = val_pickle
    return new_state_dict

def extract_weights_from_pickle(path, model):
    file = open(path, "rb")
    data = pickle.load(file)
    new_state_dict = {}
    for (key, val), (key_pickle, val_pickle) in zip(model.state_dict().items(), data.items()):
        new_state_dict[key] = val_pickle
    return new_state_dict

if __name__ == "__main__":
    path =  "/home/ubuntu/users/mateusz/Scene-Representation/project/tb_logs/Network - NuScenes/version_5/checkpoints/epoch=21-step=25806.ckpt"
    model = torchvision.models.resnet50()
    model.fc = torch.nn.Sequential()
    #sys.path.insert(0, "../")
    #from utils.models.SSL.DINO import *
    state_dict = extract_weights(path, model)
    #backbone = load_backbone(path, DINO, model, "student_backbone")
    model.load_state_dict(state_dict)
    torch.save(model.state_dict(), "/home/ubuntu/users/mateusz/Scene-Representation/project/utils/models/backbones/weights/multimodal_network_rn50.pth")


