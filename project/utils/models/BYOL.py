import warnings
import copy
import numpy as np

import pytorch_lightning as pl
import torch
import torchvision
import torch.nn as nn

from lightly.models._momentum import _MomentumEncoderMixin
from lightly.models.modules import BYOLProjectionHead, BYOLPredictionHead
from lightly.loss import NegativeCosineSimilarity


class BYOL(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = BYOLProjectionHead(512, 1024, 256)
        self.prediction_head = BYOLPredictionHead(256, 1024, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_momentum = copy.deepcopy(self.projection_head)

        self._deactivate_requires_grad(self.backbone_momentum)
        self._deactivate_requires_grad(self.projection_momentum)

        self.criterion = NegativeCosineSimilarity()

    
    def _deactivate_requires_grad(self, module):
        for param in module.parameters():
            param.requires_grad = False
    

    def _update_momentum(self, module, module_ema, tau):
        for param_ema, param_theta in zip(module_ema.parameters(), module.parameters()):
            param_ema.data = param_ema * tau + param_theta.data * (1 - tau)


    def _cosine_scheduler(self, step, max_steps, start_value, end_value):
        if max_steps == 1:
            decay = end_value
        
        elif step == max_steps:
            decay = end_value

        else:
            decay = end_value - (end_value - start_value) * (np.cos(np.pi * step / (max_steps - 1)) + 1) / 2
        
        return decay


    def forward(self, x):
        representation = self.backbone(x).flatten(start_dim=1)
        projection = self.projection_head(representation)
        prediction = self.prediction_head(projection)
        return prediction


    def forward_momentum(self, x):
        representation_momentum = self.backbone(x).flatten(start_dim=1)
        projection_momentum = self.projection_head(representation_momentum)
        projection_momentum = projection_momentum.detach()
        return projection_momentum

    
    def training_step(self, batch, batch_idx):
        momentum = self._cosine_scheduler(self.current_epoch, 10, 0.996, 1)
        self._update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        self._update_momentum(self.projection_head, self.projection_momentum, m=momentum)
        (x0, x1), _, _ = batch
        p0 = self.forward(x0)
        z0 = self.forward_momentum(x0)
        p1 = self.forward(x1)
        z1 = self.forward_momentum(x1)
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        return loss


    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.06)



model = BYOL()

