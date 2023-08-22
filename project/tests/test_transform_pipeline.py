from multiprocessing.dummy import freeze_support
from torchvision import transforms
import warnings
import sys

sys.path.insert(0, "../")

from utils.transforms.protonet_transform import *
from utils.models.SSL.prototype import *
from utils.models.backbones.scripts.PointNet import *
from src.dataloader import *
from utils.models.SSL.DINO import *


warnings.filterwarnings("ignore")

SENSORS = ["CAM_FRONT"]
data_root = "/home/ubuntu/users/mateusz/data/nuscenes"
dataset = CrossModalNuScenesDataset(data_root, sensors=SENSORS, version="v1.0-mini", split="mini_train")


train_dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    drop_last=False,
    num_workers=4
)
device = "cuda" if torch.cuda.is_available() else "cpu"
img_backbone = torchvision.models.resnet50()
pc_backbone = PointNet(point_dim=4, return_local_features=False, device=device)
model = Network(img_backbone, pc_backbone)
model = model.to(device)
for idx, batch in enumerate(train_dataloader):
    print(batch[0].shape)
    print(batch[1].shape)
    print(batch[2].shape)
    print(batch[3].shape)
    vision_features, pc_features = model.forward(batch[:2])
    print(vision_features.shape)
    print(pc_features.shape)
    break
