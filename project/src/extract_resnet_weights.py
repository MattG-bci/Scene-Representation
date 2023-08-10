import torch
import torchvision
import pickle

def extract_weights(path, model):
    loaded_state_dict = torch.load(path)
    new_state_dict = {}
    for (key, val), (key_pickle, val_pickle) in zip(model.state_dict().items(), loaded_state_dict.items()):
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
    path =  "/home/ubuntu/users/mateusz/Scene-Representation/project/utils/models/backbones/weights/model_40000_pkl"
    model = torchvision.models.resnet50()
    model.fc = torch.nn.Sequential()
    state_dict = extract_weights(path, model)
    model.load_state_dict(state_dict)
    torch.save(model.state_dict(), "/home/ubuntu/users/mateusz/Scene-Representation/project/utils/models/backbones/weights/kitti_d4lcn_rn50.pth")


