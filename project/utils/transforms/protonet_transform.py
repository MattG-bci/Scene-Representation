import PIL
import torchvision.transforms as T
from PIL.Image import Image
from torch import Tensor
import torch
import numpy as np

from lightly.transforms.multi_view_transform import MultiViewTransform

class PCTranslation:
    def __init__(self, translate):
        self.translate = np.array(translate) # it should be the probability of random pixel

    def __call__(self, pc):
        return (pc.T + self.translate).T # Not sure how to make random_t in here

    def __repr__(self):
        return "PCTranslation"
    
class PCScaling:
    def __init__(self, scale_factor):
        self.scale_factor = np.array(scale_factor) # it should be the probability of random pixel

    def __call__(self, pc):
        return (pc.T + self.scale_factor).T # Not sure how to make random_t in here

    def __repr__(self):
        return "PCScaling"

class ProtoNetTransform:
    def __init__(self, img_resize=224, degrees=0, translate=[0.1, 0.1, 0.1, 0.0], scale=[1.2, 1.2, 1.2, 1.0]):     
        self.img_transforms = T.Compose([
            T.Resize(size=img_resize),
            T.RandomAffine(degrees=degrees, translate=translate[:2], scale=scale[:2]), 
            T.ConvertImageDtype(torch.float32)
        ])
        
        self.pc_transforms = T.Compose([
            PCTranslation(translate=translate),
            PCScaling(scale_factor=scale)  
        ])
    
    def __call__(self, img, pc):
        img_transformed = self.img_transforms(img)
        pc_transformed = self.pc_transforms(pc)
        return [img, pc, img_transformed, pc_transformed]