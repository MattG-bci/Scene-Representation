import copy
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
import numpy as np
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule


class DINO(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        self.save_hyperparameters()
        backbone = nn.Sequential(*list(backbone.children())[:-1])
        input_dim = 2048
        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, 512, 64, 2048, freeze_last_layer=1
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def _common_step(self, batch, batch_idx):
        views = batch
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        return loss

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)
        loss = self._common_step(batch, batch_idx)
        self.log_dict({
            "train_loss": loss
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True)
        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
      loss = self._common_step(batch, batch_idx)
      self.log_dict({
            "val_loss": loss
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True)
      return {"val_loss": loss}

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=1e-7)
        return optim


if __name__ == "__main__":
    backbone = torchvision.models.resnet50() 
    model = DINO(backbone)
    num_params = sum(p.numel() for p in model.parameters())
    model_size_mb = num_params * 4 / (1024**2)
    print(f"The model size is: {model_size_mb:.2f} MB")
    
