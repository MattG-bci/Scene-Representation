from multiprocessing.dummy import freeze_support
from utils.models.SSL.DINO import *
from torchvision import transforms
from src.dataloader import *
import warnings
import wandb
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from lightly.transforms.dino_transform import DINOTransform

warnings.filterwarnings("ignore")
SENSORS = ["CAM_FRONT"]

backbone = torchvision.models.resnet50() 
model = DINO(backbone)
transform = transforms.Compose([
    transforms.Resize(size=224),
    DINOTransform(global_crop_size=170, local_crop_size=64, n_local_views=6, cj_prob=0.8, hf_prob=0.5, vf_prob=0.5, solarization_prob=0.0, random_gray_scale=0.2, gaussian_blur=(1, 0.1, 0.5), normalize=None) # they use RandomResizeCrop so int => (size, size)
])
data_root = "/home/ubuntu/users/mateusz/data/nuscenes"
train_dataset = NuScenesDataset(data_root, sensors=SENSORS, transform=transform, version="v1.0-trainval", split="train")
val_dataset = NuScenesDataset(data_root, sensors=SENSORS, transform=transform, version="v1.0-trainval", split="val")

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    drop_last=False,
    num_workers=4
)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
    drop_last=False,
    num_workers=4
)

### Debugging the DINO architecture.
if __name__ == "__main__":
    freeze_support()
    torch.set_float32_matmul_precision("medium")
    logger = TensorBoardLogger("/home/ubuntu/users/mateusz/Scene-Representation/project/tb_logs", name="DINOv1 - NuScenes")
    #wandb.init("DINOv1 - NuScenes")
    #logger = WandbLogger(project="DINOv1 - NuScenes")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
    early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min")
    accelerator = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(max_epochs=100, accelerator=accelerator, logger=logger, devices=1, callbacks=[checkpoint_callback, early_stopping_callback])
    trainer.fit(model, train_dataloader, val_dataloader)

