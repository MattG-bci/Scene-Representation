from multiprocessing.dummy import freeze_support
from ..src.MoCo import MoCo
from ..src.tutorial import *

gpu = 1 if torch.backends.mps.is_available() else 0

if __name__ == "__main__":
    freeze_support()
    model = MoCo()
    trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpu, accelerator="mps")
    trainer.fit(model, dataloader_train_moco)
