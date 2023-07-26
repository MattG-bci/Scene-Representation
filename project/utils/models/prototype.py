import copy
from typing import Any
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
import numpy as np
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule
from torchsummary import summary
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

from feature_extractors.PointNet import PointNet


class CrossAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        pass


class ProtoNet(pl.LightningModule):
    def __init__(self, img_backbone, point_backbone) -> None:
        super().__init__()
        self.img_backbone = nn.Sequential(*list(img_backbone.children())[:-1])
        self.point_backbone = point_backbone

    def forward(self, x):
        img = x[0]
        point_cloud = x[1]



if __name__ == "__main__":
    img_backbone = torchvision.models.resnet50()
    point_backbone = PointNet(point_dim=3, return_local_features=False)
    model = ProtoNet(img_backbone, point_backbone)
