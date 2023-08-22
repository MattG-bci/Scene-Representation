import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class TransformationNet(nn.Module):
  def __init__(self, input_dim, output_dim, device):
    super(TransformationNet, self).__init__()
    self.output_dim = output_dim
    self.device = device

    self.conv1 = nn.Conv1d(input_dim, 64, 1)
    self.conv2 = nn.Conv1d(64, 128, 1)
    self.conv3 = nn.Conv1d(128, 1024, 1)

    self.bn1 = nn.BatchNorm1d(64)
    self.bn2 = nn.BatchNorm1d(128)
    self.bn3 = nn.BatchNorm1d(1024)
    self.bn4 = nn.BatchNorm1d(512)
    self.bn5 = nn.BatchNorm1d(256)

    self.fc1 = nn.Linear(1024, 512)
    self.fc2 = nn.Linear(512, 256)
    self.fc3 = nn.Linear(256, self.output_dim * self.output_dim)

  def forward(self, x):
    n_points = x.shape[1]
    x = x.transpose(2, 1)
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))

    x = nn.MaxPool1d(n_points)(x)
    x = x.view(-1, 1024)

    x = F.relu(self.bn4(self.fc1(x)))
    x = F.relu(self.bn5(self.fc2(x)))
    x = self.fc3(x)

    eye = torch.eye(self.output_dim).to(self.device)

    x = x.view(-1, self.output_dim, self.output_dim) + eye
    return x
  

class PointNet(nn.Module):
  def __init__(self, point_dim, device, return_local_features=False):
    super(PointNet, self).__init__()
    self.return_local_features = return_local_features
    self.input_transform = TransformationNet(input_dim=point_dim, output_dim=point_dim, device=device)
    self.feature_transform = TransformationNet(input_dim=64, output_dim=64, device=device)
    self.device = device

    self.conv1 = nn.Conv1d(point_dim, 64, 1)
    self.conv2 = nn.Conv1d(64, 64, 1)
    self.conv3 = nn.Conv1d(64, 64, 1)
    self.conv4 = nn.Conv1d(64, 256, 1)
    self.conv5 = nn.Conv1d(256, 2048, 1)

    self.bn1 = nn.BatchNorm1d(64)
    self.bn2 = nn.BatchNorm1d(64)
    self.bn3 = nn.BatchNorm1d(64)
    self.bn4 = nn.BatchNorm1d(256)
    self.bn5 = nn.BatchNorm1d(2048)

  def forward(self, x):
    n_points = x.shape[1]

    input_transformed = self.input_transform(x)
    x = torch.bmm(x, input_transformed)
    x = x.transpose(2, 1)
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = x.transpose(2, 1)

    features_transformed = self.feature_transform(x)
    x = torch.bmm(x, features_transformed)
    local_point_features = x

    x = x.transpose(2, 1)
    x = F.relu(self.bn3(self.conv3(x)))
    x = F.relu(self.bn4(self.conv4(x)))
    x = F.relu(self.bn5(self.conv5(x)))
    x = nn.MaxPool1d(n_points)(x)

    x = x.view(-1, 2048)
    if self.return_local_features:
      x = x.view(-1, 2048, 1).repeat(1, 1, n_points)
      return torch.cat([x.transpose(2, 1), local_point_features], 2), features_transformed
    else:
      return x, features_transformed


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    point_cloud = torch.rand(2, 3000, 4).to(device)
    model = PointNet(point_dim=4, return_local_features=False, device=device)
    model = model.to(device)
    x, features_transformed = model(point_cloud)
    print(x.shape)
    print(features_transformed.shape)
