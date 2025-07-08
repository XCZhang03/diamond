import h5py
import json

INSERT_DATASET_PATH = "/datapool/data2/home/linhw/zhangxiangcheng/DiffRL/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeeServeMug/2024-05-01/demo_gentex_im128_randcams.hdf5"
f = h5py.File(INSERT_DATASET_PATH)
demo = f["data"]["demo_5"]                        # access demo 5
obs = demo["obs"]    
print(obs.keys())                             # obervations across all timesteps
left_img = obs["robot0_agentview_left_image"][:]  # get left camera images in numpy format
print(left_img.shape)
actions = demo["actions"][:]                     # get actions in numpy format
print(actions.shape)
ep_meta = json.loads(demo.attrs["ep_meta"])       # get meta data for episode
lang = ep_meta["lang"]                            # get language instruction for episode
f.close()
