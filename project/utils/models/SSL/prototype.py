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
    def __init__(self, in_channels, out_channels):
        super(BlenderModule, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
    
    def forward(self, imgs, pcs):
        fused_features = torch.cat([imgs, pcs], dim=1)
        out = self.conv(fused_features.unsqueeze(-1))
        return out.squeeze(-1)


class NetworkProjectionHead(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, n_layers=2):
        super(NetworkProjectionHead, self).__init__()
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
        self.projection_network = nn.Sequential(*layers)
    
    def forward(self, x):
        assert x.shape[1] == self.in_features, \
            f"Mismatch of dimensions. Got input of dimension {x.shape} but defined {self.in_features} as input features."
        
        out = self.projection_network(x)
        return out
    
class ClippedBCELoss(nn.Module):
    def __init__(self, clip_val) -> None:
        super(ClippedBCELoss, self).__init__()
        self.clip_val = clip_val
        self.loss = nn.BCELoss()
        
    def forward(self, x, y, step="train"):
        loss = self.loss(x, y)
        if step == "train":
            gradients = torch.autograd.grad(loss, x, create_graph=True)[0]
            clipped_gradients = torch.clamp(gradients, -self.clip_val, self.clip_val)
            loss = torch.sum(clipped_gradients * gradients)
        return loss
    
class Network(pl.LightningModule):
    def __init__(self, img_backbone, point_backbone) -> None:
        super().__init__()
        self.img_backbone = nn.Sequential(*list(img_backbone.children())[:-1])
        self.point_backbone = point_backbone
        #self.cross_attention_module = torch.nn.MultiheadAttention(embed_dim=2048, num_heads=16, batch_first=True)
        self.blender_module = BlenderModule(4096, 2048)
        self.projection_head = NetworkProjectionHead(4096, 2048, 1, n_layers=2)
        self.sigmoid = torch.nn.Sigmoid()
        self.criterion = ClippedBCELoss(clip_val=100.0)

    def forward(self, x):
        imgs = x[0].float().to(self.device)
        pc = x[1].float().to(self.device)
        vision_features = self.img_backbone(imgs)
        vision_features = vision_features.flatten(start_dim=1)
        pc_features, _ = self.point_backbone(pc)
        pc_features = pc_features.flatten(start_dim=1)
        #attention_out, _ = self.cross_attention_module(vision_features, pc_features, pc_features)
        out = self.blender_module(vision_features, pc_features)

        fused_features = torch.cat([vision_features, out], dim=1)
        out = self.projection_head(fused_features)
        prediction = self.sigmoid(out)
        return prediction

    def training_step(self, batch, batch_idx):
        original_pair, random_pair = batch
        original_pair_data = original_pair[:2]
        original_pair_label = original_pair[-1].float()
        original_pair_prediction = self.forward(original_pair_data).squeeze(1)
        original_pair_loss = self.criterion(original_pair_prediction, original_pair_label)
        original_pair_prediction = torch.where(original_pair_prediction > 0.5, 1, 0)
    
        random_pair_data = random_pair[:2]
        random_pair_label = random_pair[-1].float()
        random_pair_prediction = self.forward(random_pair_data).squeeze(1)
        random_pair_loss = self.criterion(random_pair_prediction, random_pair_label)
        random_pair_prediction = torch.where(random_pair_prediction > 0.5, 1, 0)
        
        total_loss = original_pair_loss + random_pair_loss
        train_acc = self.compute_accuracy((original_pair_prediction, original_pair_label), 
                                          (random_pair_prediction, random_pair_label))
                
        self.log_dict({
            "train_loss": total_loss,
            "train_accuracy": train_acc},
            on_step=True,
            on_epoch=True,
            prog_bar=True
        )
        return {"loss": total_loss, "train_accuracy": train_acc}
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        original_pair, random_pair = batch
        original_pair_data = original_pair[:2]
        original_pair_label = original_pair[-1].float()
        random_pair_data = random_pair[:2]
        random_pair_label = random_pair[-1].float()
        original_pair_prediction = self.forward(original_pair_data).squeeze(1)
        random_pair_prediction = self.forward(random_pair_data).squeeze(1)
        
        original_pair_loss = self.criterion(original_pair_prediction, original_pair_label, step="val")
        random_pair_loss = self.criterion(random_pair_prediction, random_pair_label, step="val")
        total_loss = original_pair_loss + random_pair_loss
        original_pair_prediction = torch.where(original_pair_prediction > 0.5, 1, 0)
        random_pair_prediction = torch.where(random_pair_prediction > 0.5, 1, 0)
        val_acc = self.compute_accuracy((original_pair_prediction, original_pair_label), 
                                          (random_pair_prediction, random_pair_label))
        self.log_dict({
            "val_loss": total_loss,
            "val_accuracy": val_acc},
            on_step=True,
            on_epoch=True,
            prog_bar=True
        )
        return {"val_loss": total_loss, "val_accuracy": val_acc}
    
    def compute_accuracy(self, pair1, pair2):
        tp = torch.sum(pair1[0] == pair1[1])
        fn = len(pair1[0]) - tp
        
        tn = torch.sum(pair2[0] == pair2[1])
        fp = len(pair2[0]) - tn
        return (tp + tn) / (tp + tn + fp + fn)
    
    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=1e-7)
        return optim


if __name__ == "__main__":
    import torchvision
    sys.path.insert(0, "../../")
    from models.backbones.scripts.PointNet import PointNet
    #from transforms.protonet_transform import ProtoNetTransform
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_backbone = torchvision.models.resnet50()
    point_backbone = PointNet(point_dim=4, return_local_features=False, device=device)
    model = Network(img_backbone, point_backbone)
    point_cloud = torch.rand(2, 3000, 4)
    imgs = torch.rand(2, 3, 224, 224)
    
    
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, img, pc):
            self.img = img
            self.pc = pc
            
        def __len__(self):
            return len(self.img)
        
        def __getitem__(self, idx):
            return [self.img[idx], self.pc[idx], 1], [self.img[idx], self.pc[idx], 0]

    dataset = CustomDataset(imgs, point_cloud)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        drop_last=False,
        num_workers=4
    )
    
    #img_features = torch.rand(2, 2048)
    #pc_features = torch.rand(2, 2048)
    #blender_module = BlenderModule(in_channels=4096, out_channels=2048) # B x 2 * Fdim
    #print(blender_module(img_features, pc_features).shape)
    #print(model.training_step([imgs, point_cloud], 0))
    trainer = pl.Trainer(max_epochs=100, accelerator=device, devices=1, fast_dev_run=False)
    trainer.fit(model, dataloader)

    