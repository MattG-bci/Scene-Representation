from turtle import update
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import copy
import lightly

from lightly.models.modules.heads import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum
from lightly.models.utils import batch_shuffle
from lightly.models.utils import batch_unshuffle


class MoCo(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()

        # Creating a ResNet-18 backbone without the classification head.
        resnet = lightly.models.ResNetGenerator("resnet-18", 1, num_splits=8)
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )

        self.projection_head = MoCoProjectionHead(512, 512, 128)
        self.backbone_momentum = copy.deepcopy(self.projection_head)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        self.criterion = lightly.loss.NTXentLoss(
            temperature=0.1,
            memory_bank_size=4096
        )

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def training_step(self, batch):
        (x_q, x_k), _, _ = batch

        update_momentum(self.backbone, self.backbone_momentum, 0.99)
        update_momentum(self.projection_head, self.projection_head_momentum, 0.99)

        # compute queries
        q = self.backbone(x_q).flatten(start_dim=1)
        q = self.projection_head(q)

        # compute keys
        k, shuffle = batch_shuffle(x_k)
        k = self.backbone_momentum(k).flatten(start_dim=1)
        k = self.projection_head_momentum(k)
        k = batch_unshuffle(k, shuffle)

        loss = self.criterion(q, k)
        self.log(f"Training Loss: {loss}")
        return loss

    def training_epoch_end(self, outputs):
        self.custom_histogram_weights()

    def custom_histogram_weights(self):
        for name, params in self.named_paramteres():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch
            )
    
    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(),
            lr=6e-2,
            momentum=0.9,
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 1)
        return [optim], scheduler


