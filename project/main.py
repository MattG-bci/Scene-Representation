from multiprocessing.dummy import freeze_support
from utils.models.SSL.Dino import *
from torchvision import transforms
from src.dataloader import *
import warnings
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


warnings.filterwarnings("ignore")
SENSORS = ["CAM_FRONT"]

backbone = torchvision.models.resnet50() 
model = DINO(backbone)
transform = transforms.Compose([
    transforms.Resize(size=224),
    DINOTransform(global_crop_size=200, global_crop_ratio=(16/9, 16/9), local_crop_size=96, cj_prob=0.0, hf_prob=0.0, solarization_prob=0.0, random_gray_scale=0.0, gaussian_blur=(0.0, 0.0, 0.0), normalize=None) # they use RandomResizeCrop so int => (size, size)
])
data_root = "/home/efs/users/mateusz/data/nuscenes"
train_dataset = NuScenesDataset(data_root, sensors=SENSORS, transform=transform, split="train")
val_dataset = NuScenesDataset(data_root, sensors=SENSORS, transform=transform, split="val")

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
    logger = TensorBoardLogger("tb_logs", name="DINOv1 - NuScenes")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min")
    accelerator = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(max_epochs=100, accelerator=accelerator, devices=1, logger=logger, callbacks=[checkpoint_callback, early_stopping_callback])
    trainer.fit(model, train_dataloader, val_dataloader)

