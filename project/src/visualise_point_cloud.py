import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


path = "/home/ubuntu/users/mateusz/data/nuscenes/samples/LIDAR_TOP/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin"

lidar_pts = np.fromfile(path, dtype=np.float32, count=-1).reshape([-1, 5])[:,:3] #(x, y, z, intensity, ring index)
lidar_pointcloud = o3d.geometry.PointCloud()
lidar_pointcloud.points = o3d.utility.Vector3dVector(lidar_pts)
o3d.io.write_point_cloud("pointcloud.pcd", lidar_pointcloud)
