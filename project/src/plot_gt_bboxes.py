from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt

path = "/home/ubuntu/users/mateusz/data/nuscenes"
nusc = NuScenes(dataroot=path, version="v1.0-trainval", verbose=False)

my_scene = nusc.scene[420]
my_sample = nusc.get('sample', my_scene["first_sample_token"])
cam_front_data = nusc.get("sample_data", my_sample["data"]["CAM_FRONT"])

plt.figure(figsize=(8, 8))
nusc.render_sample_data(cam_front_data["token"])
plt.savefig("./bboxes.jpg")
plt.close()