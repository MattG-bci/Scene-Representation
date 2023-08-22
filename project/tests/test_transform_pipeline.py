from multiprocessing.dummy import freeze_support
from torchvision import transforms
import warnings
import sys
import pytorch_lightning as pl

sys.path.insert(0, "../")

from utils.transforms.protonet_transform import *
from utils.models.SSL.prototype import *
from utils.models.backbones.scripts.PointNet import *
from src.dataloader import *
from utils.models.SSL.DINO import *


warnings.filterwarnings("ignore")

SENSORS = ["CAM_FRONT"]
data_root = "/home/ubuntu/users/mateusz/data/nuscenes"
img_transforms = torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=3), torchvision.transforms.RandomCrop(200)])
dataset = CrossModalNuScenesDataset(data_root, sensors=SENSORS, version="v1.0-mini", split="mini_train", transform=img_transforms)

train_dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    drop_last=False,
    num_workers=4
)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("medium")

img_backbone = torchvision.models.resnet50()
pc_backbone = PointNet(point_dim=4, return_local_features=False, device=device)
model = Network(img_backbone, pc_backbone)
model = model.to(device)

trainer = pl.Trainer(max_epochs=100, accelerator=device, devices=1, fast_dev_run=False)
trainer.fit(model, train_dataloader)
