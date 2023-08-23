from multiprocessing.dummy import freeze_support
from torchvision import transforms
import warnings
import sys
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.tuner.tuning import Tuner

from utils.transforms.protonet_transform import *
from utils.models.SSL.prototype import *
from utils.models.backbones.scripts.PointNet import *
from src.dataloader import *
from utils.models.SSL.DINO import *


warnings.filterwarnings("ignore")

SENSORS = ["CAM_FRONT"]
data_root = "/home/ubuntu/users/mateusz/data/nuscenes"
img_transforms = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(200)])
data_root = "/home/ubuntu/users/mateusz/data/nuscenes"
train_dataset = CrossModalNuScenesDataset(data_root, sensors=SENSORS, transform=None, version="v1.0-trainval", split="train")
val_dataset = CrossModalNuScenesDataset(data_root, sensors=SENSORS, transform=None, version="v1.0-trainval", split="val")

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=30,
    shuffle=True,
    drop_last=False,
    num_workers=4
)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=30,
    shuffle=False,
    drop_last=False,
    num_workers=4
)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("medium")
    logger = TensorBoardLogger("/home/ubuntu/users/mateusz/Scene-Representation/project/tb_logs", name="Network - NuScenes")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
    #early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min")

    img_backbone = torchvision.models.resnet50()
    pc_backbone = PointNet(point_dim=4, return_local_features=False, device=device)
    model = Network(img_backbone, pc_backbone)
    model = model.to(device)

    trainer = pl.Trainer(max_epochs=100, accelerator=device, logger=logger, devices=1, callbacks=[checkpoint_callback], fast_dev_run=False)
    trainer.fit(model, train_dataloader, val_dataloader)
