from multiprocessing.dummy import freeze_support
from lightly.data.multi_view_collate import MultiViewCollate
from lightly.transforms.simclr_transform import SimCLRTransform
from utils.models.BYOL import *
from lightly.data import LightlyDataset
from torchvision import transforms

model = BYOL()

transforms = SimCLRTransform(input_size=32)


cifar_path = "./datasets/"
dataset = LightlyDataset(cifar_path, transform=transforms)
collate_fn = MultiViewCollate()

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    collate_fn=None,
    shuffle=True,
    drop_last=True,
    num_workers=8
)

if __name__ == "__main__":
    freeze_support()
    accelerator = "mps" if torch.backends.mps.is_available() else "cpu"

    trainer = pl.Trainer(max_epochs=1, devices=1, accelerator=accelerator)
    trainer.fit(model=model, train_dataloaders=dataloader)

