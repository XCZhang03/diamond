import h5py
import json
import argparse
from functools import partial
from pathlib import Path
from multiprocessing import Pool
import shutil
import subprocess

import torch
import torchvision.transforms.functional as T
from tqdm import tqdm

from data.dataset import Dataset, CSGOHdf5Dataset
from data.episode import Episode
from data.segment import SegmentId

from data.robocasa_registry import get_ds_path, SINGLE_STAGE_TASK_DATASETS


def get_ep(demo, info) -> Episode:
    obs = demo["obs"]
    left_img = torch.from_numpy(obs["robot0_agentview_left_image"][:]).permute(0, 3, 1, 2).div(255).mul(2).sub(1)  # Convert to CxHxW format
    right_img = torch.from_numpy(obs["robot0_agentview_right_image"][:]).permute(0, 3, 1, 2).div(255).mul(2).sub(1)
    eye_img = torch.from_numpy(obs["robot0_eye_in_hand_image"][:]).permute(0, 3, 1, 2).div(255).mul(2).sub(1)
    empty_img = torch.zeros_like(left_img).div(255).mul(2).sub(1)
    # Concatenate images into a 2x2 grid: [left, right; eye, empty]
    top_row = torch.cat([left_img, right_img], dim=-1)
    bottom_row = torch.cat([eye_img, empty_img], dim=-1)
    full_res_obs = torch.cat([top_row, bottom_row], dim=-2)
    obs = T.resize(full_res_obs, (32, 32))
    action = demo['actions'][:]
    act = torch.from_numpy(action)
    rew = torch.zeros(obs.size(0))
    end = torch.zeros(obs.size(0), dtype=torch.uint8)
    trunc = torch.zeros(obs.size(0), dtype=torch.uint8)
    
    return Episode(obs=obs, act=act, rew=rew, end=end, trunc=trunc, info=info, full_res_obs=full_res_obs)

def main():
    out_dir = Path("dataset/robocasa/mg_im")
    
    train_dataset = Dataset(
        directory=out_dir / "train",
        dataset_full_res=None,)
    test_dataset = Dataset(
        directory=out_dir / "test",
        dataset_full_res=None,)

    for task in tqdm(list(SINGLE_STAGE_TASK_DATASETS.keys())[:-1]):
        ds_path = get_ds_path(task, ds_type="mg_im", return_info=False)
        f = h5py.File(ds_path, "r")
        demo_keys = list(f['data'].keys())
        num_demos = len(demo_keys)
        temp_len = int(0.1 * num_demos)
        split_idx = int(0.9 * temp_len)
        train_keys = demo_keys[:split_idx]
        test_keys = demo_keys[split_idx:temp_len]

        for key in train_keys:
            # Add to train_dataset
            demo = f['data'][key]
            ## fill info
            ep_meta = json.loads(demo.attrs['ep_meta'])
            info = {}
            info['task'] = task
            info['lang'] = ep_meta['lang']
            info['demo_id'] = key
            info['_filename'] = ds_path
            ## fill data
            ep = get_ep(demo, info)
            train_dataset.add_episode(ep)
            assert train_dataset.lengths[-1] == len(ep)

        for key in test_keys:
            # Add to test_dataset
            demo = f['data'][key]
            ## fill info
            ep_meta = json.loads(demo.attrs['ep_meta'])
            info = {}
            info['task'] = task
            info['lang'] = ep_meta['lang']
            info['demo_id'] = key
            info['_filename'] = ds_path
            ## fill data
            ep = get_ep(demo, info)
            test_dataset.add_episode(ep)
        print(f"Split train/test data in task {task} with ({train_dataset.num_episodes}/{test_dataset.num_episodes} episodes)\n")
    train_dataset.save_to_default_path()
    test_dataset.save_to_default_path()
    # train_dataset.load_from_default_path()
    # test_dataset.load_from_default_path()
    # num_train_episodes = train_dataset.num_episodes
    # num_test_episodes = test_dataset.num_episodes
    # print(f"Loaded train dataset with {num_train_episodes} episodes and test dataset with {num_test_episodes} episodes.")
    # train_dataset._reset()
    # test_dataset._reset()
    # for episode_id in tqdm(range(num_train_episodes)):
    #     episode = train_dataset.load_episode(episode_id)
    #     train_dataset.add_episode(episode)
    # for episode_id in tqdm(range(num_test_episodes)):
    #     episode = test_dataset.load_episode(episode_id)
    #     test_dataset.add_episode(episode)
    # train_dataset.save_to_default_path()
    # test_dataset.save_to_default_path()
    # print(f"Saved train dataset with {train_dataset.num_episodes} episodes and test dataset with {test_dataset.num_episodes} episodes.")
        
if __name__ == "__main__":
    main()
        
