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
transform = ProtoNetTransform(img_resize=224, degrees=0, translate=[0.1, 0.1, 0.1, 0.0], scale=[1.2, 1.2, 1.2, 1.0])
dataset = CrossModalNuScenesDataset(data_root, transform=transform, sensors=SENSORS, version="v1.0-mini", split="mini_train")


train_dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    drop_last=False,
    num_workers=4
)

img_backbone = torchvision.models.resnet50()
pc_backbone = PointNet(point_dim=4, return_local_features=False)
model = ProtoNet(img_backbone, pc_backbone)
for idx, batch in enumerate(train_dataloader):
    print(model.training_step(batch, idx))
    break