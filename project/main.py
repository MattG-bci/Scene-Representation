from multiprocessing.dummy import freeze_support
from lightly.data.multi_view_collate import MultiViewCollate
from lightly.transforms.simclr_transform import SimCLRTransform
from utils.models.SSL.Dino import *
from lightly.data import LightlyDataset
from torchvision import transforms
from src.dataloader import NuScenesDataset
import pandas as pd
import warnings
from pytorch_lightning.loggers import TensorBoardLogger


warnings.filterwarnings("ignore")
SENSORS = ["CAM_FRONT", "LIDAR_TOP"]

backbone = torchvision.models.resnet18() # for ResNet-50 there was an issue in memory allocation. Probably something to be optimised. 
model = DINO(backbone)
transforms = DINOTransform(global_crop_size=(480, 270), normalize=None)
data_root = "/home/efs/users/mateusz/data/nuscenes_tiny/v1.0-trainval"
train_dataset = NuScenesDataset(data_root, sensors=SENSORS, transform=transforms, split="train")
val_dataset = NuScenesDataset(data_root, sensors=SENSORS, transform=transforms, split="val")

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
    drop_last=False,
    num_workers=4
)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=256,
    shuffle=False,
    drop_last=False,
    num_workers=4
)

### Debugging the DINO architecture.
if __name__ == "__main__":
    freeze_support()
    torch.set_float32_matmul_precision("medium")
    logger = TensorBoardLogger("tb_logs", name="my_model_run_name")
    accelerator = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(max_epochs=2, accelerator=accelerator, devices=1, logger=logger)
    trainer.fit(model, train_dataloader, val_dataloader)