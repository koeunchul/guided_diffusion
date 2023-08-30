import io
import os
import socket

import blobfile as bf
import torch as th

def setup_dist(gpu_num):
    """
    Setup a distributed process group.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num

def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")

def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    return th.load(path, **kwargs)
