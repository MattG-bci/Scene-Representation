from multiprocessing.dummy import freeze_support
from torchvision import transforms
import warnings
import sys
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from utils.transforms.protonet_transform import *
from src.dataloader import *
from utils.models.SSL.Dino import *


warnings.filterwarnings("ignore")

SENSORS = ["CAM_FRONT"]
data_root = "/home/ubuntu/users/mateusz/data/nuscenes"
transform = ProtoNetTransform(img_resize=224, degrees=0, translate=[0.1, 0.1, 0.1, 0.0], scale=[1.2, 1.2, 1.2, 1.0])
dataset = CrossModalNuScenesDataset(data_root, transform=transform, sensors=SENSORS, split="mini_train")


train_dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    drop_last=False,
    num_workers=4
)

for idx, batch in enumerate(train_dataloader): # padding point clouds
    print(batch)
    break
