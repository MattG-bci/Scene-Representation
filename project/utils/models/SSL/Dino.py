import copy
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
import numpy as np
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.transforms.dino_transform import DINOTransform
from torchsummary import summary


class DINO(pl.LightningModule):
    def __init__(self, backbone_network):
        super().__init__()
        backbone = nn.Sequential(*list(backbone_network.children())[:-1])
        input_dim = 2048
        backbone.apply(self._init_weights)
        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(input_dim, 512, 64, 2048, freeze_last_layer=1) # no. neurons in the projection head not exactly matching in the original DINO paper
                                                                                                # which would be (in_dim, 2048, 256, 4096) here.
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        self._deactivate_requires_grad(self.teacher_backbone)
        self._deactivate_requires_grad(self.teacher_head)
        self.criterion = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)
        self.save_hyperparameters()

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
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            module.bias.data.fill(0.01)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.xavier_normal_(module.weight)
    
    def _common_step(self, batch, batch_idx):
        views = batch[0]
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view.unsqueeze(0)) for view in global_views]
        student_out = [self.forward(view.unsqueeze(0)) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        return loss
    
    def training_step(self, batch, batch_idx):
        momentum = self._cosine_scheduler(self.current_epoch, 10, 0.996, 1)
        self._update_momentum(self.student_backbone, self.teacher_backbone, tau=momentum)
        self._update_momentum(self.student_head, self.teacher_head, tau=momentum)
        loss = self._common_step(batch, batch_idx)
        self.log_dict({
            "train_loss": loss
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True)
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log_dict({
            "val_loss": loss
        })
        return {"loss": loss}

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)
    
    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=0.0005) # 0.0005 according to the original DINO paper
        return optim

if __name__ == "__main__":
    backbone = torchvision.models.resnet50() 
    model = DINO(backbone)
    num_params = sum(p.numel() for p in model.parameters())
    model_size_mb = num_params * 4 / (1024**2)
    print(f"The model size is: {model_size_mb:.2f} MB")
    
