from nuscenes.nuscenes import NuScenes 


DATA_ROOT = "/home/efs/datasets/nuscenes_tiny"
VERSION = "v1.0-trainval"

nusc = NuScenes(version=VERSION, dataroot=DATA_ROOT, verbose=True)

# Scenes in the dataset
#nusc.list_scenes()


#scene_1 = nusc.scene[0]
#print(scene_1)
#print(scene_1["first_sample_token"])

#my_sample = nusc.get("sample", scene_1["first_sample_token"])
#print(my_sample)


my_sample = nusc.sample[10]
print(my_sample)
#print(my_sample["token"])
#print(my_sample["data"]["LIDAR_TOP"])
#nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='LIDAR_TOP')


