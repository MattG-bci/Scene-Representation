from multiprocessing.dummy import freeze_support
from lightly.data.multi_view_collate import MultiViewCollate
from lightly.transforms.simclr_transform import SimCLRTransform
from utils.models.Dino import *
from lightly.data import LightlyDataset
from torchvision import transforms
from src.load_data import CustomDataset
import pandas as pd
import warnings


warnings.filterwarnings("ignore")


backbone = torchvision.models.resnet18() # for ResNet-50 there was an issue in memory allocation. Probably something to be optimised. 
model = DINO(backbone)

transforms = DINOTransform()


df_train = pd.read_parquet("/home/efs/users/mateusz/Data-Mining/src/pq_labels/det_train_new.parquet", engine="fastparquet")
df_val = pd.read_parquet("/home/efs/users/mateusz/Data-Mining/src/pq_labels/det_val_new.parquet", engine="fastparquet")
df = pd.concat([df_train, df_val])
images = df.name.values

dataset = CustomDataset(images, transforms=transforms)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    shuffle=False,
    drop_last=True,
    num_workers=4
)

### Debugging the DINO architecture.
if __name__ == "__main__":
    freeze_support()
    accelerator = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(max_epochs=1, accelerator=accelerator, devices=1, fast_dev_run=True)
    trainer.fit(model=model, train_dataloaders=dataloader)

