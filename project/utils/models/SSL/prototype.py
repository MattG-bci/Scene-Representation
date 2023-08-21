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
import lightly
from lightly.models.modules.heads import MoCoProjectionHead


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
        
        assert num_heads <= out_channels, \
            "The number of heads must be lower or equal than the embedding dimension."
            
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

class BlenderModule(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, n_layers=2):
        super(BlenderModule, self).__init__()
        self.in_features = in_features
        layers = []
        
        layers.append([
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_features)])
        for _ in range(n_layers):
            layers.append([nn.Linear(in_features=hidden_features, out_features=hidden_features),
                           nn.ReLU(),
                           nn.BatchNorm1d(hidden_features)])
        layers = [layers[i][j] for i in range(len(layers)) for j in range(len(layers[i]))]
        layers.append(nn.Linear(in_features=hidden_features, out_features=out_features))
        self.blender_network = nn.Sequential(*layers)
    
    def forward(self, features_1, features_2):
        fused_features = torch.cat([features_1, features_2], dim=1).squeeze(-1)
        
        assert fused_features.shape[1] == self.in_features, \
            f"Mismatch of dimensions. Got input of dimension {fused_features.shape} but defined {self.in_features} as input features."
        
        out = self.blender_network(fused_features)
        return out

class ProtoNet(pl.LightningModule):
    def __init__(self, img_backbone, point_backbone) -> None:
        super().__init__()
        self.img_backbone = nn.Sequential(*list(img_backbone.children())[:-1])
        self.point_backbone = point_backbone
        #self.cross_attention_module = CrossAttentionModule(in_channels_query=1, in_channels_kv=1, out_channels=64, num_heads=1) # probably not very useful
        #self.conv = nn.Conv2d(kernel_size=1, in_channels=64, out_channels=1)
        #self.projection_head = MoCoProjectionHead(4096, 2048, 128)
        
        #self.projection_head_aug = copy.deepcopy(self.projection_head)
        
        
        self.blender_module = BlenderModule(in_features=4096, hidden_features=2048, out_features=2048)
        self.criterion = lightly.loss.DINOLoss(output_dim=2048) # loss subject to change

    def forward(self, x):
        imgs = x[0].float().to(self.device)
        pc = x[1].float().to(self.device)
        vision_features = self.img_backbone(imgs)
        vision_features = vision_features.squeeze(-2)
        pc_features, _ = self.point_backbone(pc)
        pc_features = pc_features.unsqueeze(-1)
        #attention_out = self.cross_attention_module(vision_features, point_cloud_features, point_cloud_features)
        #attention_out = attention_out.transpose(0, -1)
        #out = self.conv(attention_out)
        #out = out.transpose(0, -1)
        #fused_features = torch.cat([vision_features, out], 1)
        return vision_features, pc_features
    
    def training_step(self, batch, batch_idx):
        vision_features, pc_features = self.forward(batch)
        fused_features = self.blender_module(vision_features, pc_features)
        #aug_fused_features = self.forward(batch[2:]).flatten(start_dim=1)
        #q = self.projection_head(fused_features)
        #v = self.projection_head_aug(aug_fused_features)
        loss = self.criterion(vision_features, fused_features, epoch=self.current_epoch)
        return loss
    
    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=1e-7)
        return optim


if __name__ == "__main__":
    import torchvision
    sys.path.insert(0, "../../")
    from models.backbones.scripts.PointNet import PointNet
    #from transforms.protonet_transform import ProtoNetTransform
        
    img_backbone = torchvision.models.resnet50()
    point_backbone = PointNet(point_dim=4, return_local_features=False)
    model = ProtoNet(img_backbone, point_backbone)
    point_cloud = torch.rand(2, 30000, 4)
    imgs = torch.rand(2, 3, 224, 224)
    accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, img, pc):
            self.img = img
            self.pc = pc
            
        def __len__(self):
            return len(self.img)
        
        def __getitem__(self, idx):
            return self.img[idx], self.pc[idx]

    dataset = CustomDataset(imgs, point_cloud)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        drop_last=False,
        num_workers=4
    )
    
    #print(model.training_step([imgs, point_cloud], 0))
    trainer = pl.Trainer(max_epochs=100, accelerator=accelerator, devices=1, fast_dev_run=True)
    trainer.fit(model, dataloader)

    