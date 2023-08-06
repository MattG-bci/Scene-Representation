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


if __name__ == "__main__":
    #path = "/home/ubuntu/users/mateusz/cascade_rcnn_r50_fpn_1x_det_bdd100k.pth"
    #model = torchvision.models.resnet50()
    #model.fc = torch.nn.Sequential()
    #state_dict = extract_backbone_weights(path)
    
    #model.load_state_dict(state_dict)
    #torch.save(model.state_dict(), "bdd_det_backbone.pth")
    #print(model)
    path =  "/home/ubuntu/users/mateusz/carnet_depth_resnet50.pickle"
    state_dict = extract_from_pickle(path)
    model = torchvision.models.resnet50()
    #model.fc = torch.nn.Sequential()
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print('Ignoring "' + str(e) + '"')
    torch.save(model.state_dict(), "depth_backbone.pth")
    print(model)

