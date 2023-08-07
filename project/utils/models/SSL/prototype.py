import copy
import sys
from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
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


class CrossAttentionModule(nn.Module):
    def __init__(self, in_channels_query, in_channels_kv, out_channels, num_heads=16):
        super(CrossAttentionModule, self).__init__()
        # Define the dimension of the queries, keys, and values for both modalities
        self.in_channels_query = in_channels_query
        self.in_channels_kv = in_channels_kv
        self.out_channels = out_channels
        # Number of attention heads
        self.num_heads = num_heads
        # Linear projections for queries, keys, and values
        self.query_projection = nn.Linear(in_channels_query, out_channels)
        self.key_projection = nn.Linear(in_channels_kv, out_channels)
        self.value_projection = nn.Linear(in_channels_kv, out_channels)
        # Scaled Dot-Product Attention layer
        self.softmax = nn.Softmax(dim=-1)
        self.scale_factor = 1.0 / (out_channels // num_heads) ** 0.5
        # Output projection
        self.output_projection = nn.Linear(out_channels, out_channels)
        
    def forward(self, query, key, value):
        # Project the queries, keys, and values
        q = self.query_projection(query)
        k = self.key_projection(key)
        v = self.value_projection(value)
        # Split the queries, keys, and values into multiple heads
        q = q.view(*q.size()[:-1], self.num_heads, -1)
        k = k.view(*k.size()[:-1], self.num_heads, -1)
        v = v.view(*v.size()[:-1], self.num_heads, -1)
        # Compute the attention scores and attention weights
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale_factor
        attention_weights = self.softmax(scores)
        # Apply attention to the values
        attention_output = torch.matmul(attention_weights, v)
        # Concatenate the attention outputs from all heads and reshape
        attention_output = attention_output.view(*attention_output.size()[:-2], -1)
        # Output projection
        output = self.output_projection(attention_output)
        return output


class ProtoNet(pl.LightningModule):
    def __init__(self, img_backbone, point_backbone) -> None:
        super().__init__()
        self.img_backbone = nn.Sequential(*list(img_backbone.children())[:-1])
        self.point_backbone = point_backbone
        self.cross_attention_module = CrossAttentionModule(in_channels_query=1, in_channels_kv=1, out_channels=64, num_heads=1) # probably not very useful
        self.conv = nn.Conv2d(kernel_size=1, in_channels=64, out_channels=1)

    def forward(self, x):
        imgs = x[0]
        point_cloud = x[1].float()
        vision_features = self.img_backbone(imgs)
        vision_features = vision_features.squeeze(-2)
        point_cloud_features, _ = self.point_backbone(point_cloud)
        point_cloud_features = point_cloud_features.unsqueeze(-1)
        attention_out = self.cross_attention_module(vision_features, point_cloud_features, point_cloud_features)
        attention_out = attention_out.transpose(0, -1)
        out = self.conv(attention_out)
        out = out.transpose(0, -1)
        fused_features = torch.cat([vision_features, out], 1)
        return fused_features
    
    def training_step(self, batch, batch_idx):
        fused_features = self.forward(batch)


if __name__ == "__main__":
    import torchvision
    sys.path.insert(0, "../")
    from backbones.scripts.PointNet import PointNet
    
    img_backbone = torchvision.models.resnet50()
    point_backbone = PointNet(point_dim=3, return_local_features=False)
    model = ProtoNet(img_backbone, point_backbone)
    point_cloud = torch.rand(2, 30000, 3)
    imgs = torch.rand(2, 3, 224, 224)
    x = (imgs, point_cloud)
    transforms = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop(size=200)])
    out = transforms(x[0])

    