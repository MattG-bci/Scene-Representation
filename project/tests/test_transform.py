import torch
import torchvision.transforms as T
import numpy as np
#import torch_geometric.transforms as T


transform = T.Compose([T.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2))])

def apply_translation(point_cloud, translation):
    # Your code to apply translation to the point cloud
    return point_cloud + translation

def apply_rotation(point_cloud, rotation_angle):
    # Your code to apply rotation to the point cloud
    # Assuming the point cloud has shape (N, 3), where N is the number of points
    # and the rotation is around the z-axis (third dimension)
    x_rotated = point_cloud[:, 0] * torch.cos(rotation_angle) - point_cloud[:, 1] * torch.sin(rotation_angle)
    y_rotated = point_cloud[:, 0] * torch.sin(rotation_angle) + point_cloud[:, 1] * torch.cos(rotation_angle)
    rotated_point_cloud = torch.stack((x_rotated, y_rotated, point_cloud[:, 2]), dim=1)
    return rotated_point_cloud

def apply_scaling(point_cloud, scaling_factor):
    # Your code to apply scaling to the point cloud
    return point_cloud * scaling_factor


img = torch.randn(3, 224, 224)
point_cloud = torch.rand(3000, 3)

img_transformed = transform(img)

# Define your augmentation parameters for the point cloud
translation = torch.tensor([0.1, 0.1, 0.1])
rotation_angle = torch.tensor(np.radians(30))
scaling_factor = torch.tensor(1.2)

# Apply translation to the point cloud
translated_point_cloud = apply_translation(point_cloud, translation)

# Apply rotation to the point cloud
rotated_point_cloud = apply_rotation(translated_point_cloud, rotation_angle)

# Apply scaling to the point cloud
scaled_point_cloud = apply_scaling(rotated_point_cloud, scaling_factor)

