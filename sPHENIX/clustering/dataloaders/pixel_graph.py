from collections import namedtuple
# System imports
import os
import random

# External imports
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, Sampler
import torch_geometric
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import default_collate
import tqdm
import functools
from typing import Union

from numpy.linalg import inv
from icecream import ic
from collections import namedtuple
from disjoint_set import DisjointSet
import dataclasses
from scipy.stats import mode
from . import utils
from torch_geometric.data import Data

@dataclasses.dataclass
class EventInfo:
    pixel_cartesian: Union[np.ndarray, torch.Tensor]
    true_pixel: Union[np.ndarray, torch.Tensor]
    edge_index: Union[np.ndarray, torch.Tensor]
    edge_attr: Union[np.ndarray, torch.Tensor]


def load_graph(filename, max_radius):
    with np.load(filename) as f:
        pixel_cartesian = f['pixel_cartesian']
        hit_cartesian = f['hit_cartesian']
        distances = np.linalg.norm(pixel_cartesian[:, None, :] - pixel_cartesian[None, :, :], axis=-1)
        keep = distances <= max_radius

        row = np.arange(distances.shape[0])[:, None].repeat(distances.shape[1], 1)
        column = np.arange(distances.shape[0])[None, :].repeat(distances.shape[1], 0)
        start = row[keep]
        end = column[keep]
        distances = distances[keep]
        edge_index = np.stack([start, end], axis=0)

        hit_distances = np.linalg.norm(hit_cartesian[:, None, :] - pixel_cartesian[None, :, :], axis=-1)
        true_pixels = np.argmin(hit_distances, axis=-1)
        pixel_classification = np.zeros(pixel_cartesian.shape[0], dtype=int)
        pixel_classification[true_pixels] = 1


        return EventInfo(
            pixel_cartesian=pixel_cartesian,
            edge_index=edge_index,
            edge_attr=distances[:, None],
            true_pixel=pixel_classification
        )

class ClusterDataset(object):
    """PyTorch dataset specification for hit graphs"""

    def __init__(
            self, 
            trigger_input_dir, 
            nontrigger_input_dir, 
            n_trigger_samples,
            n_nontrigger_samples,
            min_edge_probability=0.5,
            max_radius=5e-3
            ):
        self.filenames = []
        if trigger_input_dir is not None:
            input_dir = os.path.expandvars(trigger_input_dir)
            filenames = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)
                                if f.startswith('event') and not f.endswith('_ID.npz')])
            random.shuffle(filenames)
            self.filenames = filenames[:n_trigger_samples]

        if nontrigger_input_dir is not None:
            input_dir = os.path.expandvars(nontrigger_input_dir)
            filenames = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)
                            if f.startswith('event') and not f.endswith('_ID.npz')])
            self.filenames += filenames[:n_nontrigger_samples]
            random.shuffle(self.filenames)

        self.max_radius = max_radius


    def __getitem__(self, file_index):
        event_info = load_graph(self.filenames[file_index], self.max_radius)
        return Data(
                x=torch.from_numpy(event_info.pixel_cartesian),
                y=torch.from_numpy(event_info.true_pixel),
                edge_index=torch.from_numpy(event_info.edge_index),
                edge_attr=torch.from_numpy(event_info.edge_attr)
            )

    def __len__(self):
        return len(self.filenames)


def get_datasets(n_train, n_valid, n_test, 
        trigger_input_dir=None, 
        nontrigger_input_dir=None,
        max_radius=5e-3):
    data = ClusterDataset(trigger_input_dir=trigger_input_dir,
                        nontrigger_input_dir=nontrigger_input_dir,
                        n_trigger_samples=n_train+n_valid+n_test,
                        n_nontrigger_samples=n_train+n_valid+n_test,
                        max_radius=max_radius)

    total = (trigger_input_dir is not None) + (nontrigger_input_dir is not None)
    train_data, valid_data, test_data = random_split(data, [total*n_train, total*n_valid, total*n_test])

    return train_data, valid_data, test_data
