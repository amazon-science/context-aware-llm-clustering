"""This file contain common utility functions."""

import json
import os
import random
import string
from datetime import datetime

from pytz import timezone
from tqdm import tqdm

tqdm.pandas()
from typing import Any, Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from transformers import set_seed


def get_curr_time() -> str:
    """Get current date and time in PST as str."""
    return datetime.now().astimezone(timezone("US/Pacific")).strftime("%d/%m/%Y %H:%M:%S")


class Logger:
    """Class to write message to both output_dir/filename.txt and terminal."""

    def __init__(self, output_dir: str = None, filename: str = None) -> None:
        if filename is not None:
            self.log = os.path.join(output_dir, filename)

    def write(self, message: Any, show_time: bool = True) -> None:
        "write the message"
        message = str(message)
        if show_time:
            # if message starts with \n, print the \n first before printing time
            if message.startswith("\n"):
                message = "\n" + get_curr_time() + " >> " + message[1:]
            else:
                message = get_curr_time() + " >> " + message
        print(message)
        if hasattr(self, "log"):
            with open(self.log, "a") as f:
                f.write(message + "\n")


def set_all_seeds(seed: int) -> None:
    """Function to set seeds for all RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    set_seed(seed)


def simplify_string(s: str, is_cluster_name: bool = True) -> str:
    """Function to simplify a string s by removing punctuation,
    extra spaces, using lower case, stc.."""
    s = s.replace("-", " ").replace("&amp", "&").replace("&quot", '"')
    translator = str.maketrans("", "", string.punctuation)
    s = s.lower().translate(translator).strip()
    # if s is a cluster name and starts like 'cluster 1',
    # remove that prefix part.
    if is_cluster_name:
        if s.startswith("cluster "):
            s = s[8:].lstrip("0123456789").lstrip()
    return " ".join(s.split())


class CycleIndex:
    """Class to generate batches of training ids,
    shuffled after each epoch."""

    def __init__(self, indices: Union[int, list], batch_size: int, shuffle: bool = True) -> None:
        if type(indices) == int:
            indices = np.arange(indices)
        self.indices = indices
        self.num_samples = len(indices)
        self.batch_size = batch_size
        self.pointer = 0
        if shuffle:
            np.random.shuffle(self.indices)
        self.shuffle = shuffle

    def get_batch_ind(self):
        """Get indices for next batch."""
        start, end = self.pointer, self.pointer + self.batch_size
        # If we have a full batch within this epoch, then get it.
        if end <= self.num_samples:
            if end == self.num_samples:
                self.pointer = 0
                if self.shuffle:
                    np.random.shuffle(self.indices)
            else:
                self.pointer = end
            return self.indices[start:end]
        # Otherwise, fill the batch with samples from next epoch.
        last_batch_indices_incomplete = self.indices[start:]
        remaining = self.batch_size - (self.num_samples - start)
        self.pointer = remaining
        if self.shuffle:
            np.random.shuffle(self.indices)
        return np.concatenate((last_batch_indices_incomplete, self.indices[:remaining]))


def save_ckpt(save_path, model, optimizer, num_batches_trained, best_val_metric):
    """Save model weights, optimizer states, and other
    variables needed to resume training."""
    changing_params = [k for k, p in model.named_parameters() if p.requires_grad]
    model_weights = model.state_dict()
    model_weights = {k: model_weights[k].cpu() for k in changing_params}
    save_dict = {
        "model_weights": model_weights,
        "optimizer_state": optimizer.state_dict(),
        "num_batches_trained": num_batches_trained,
        "best_val_metric": best_val_metric,
        "random_rng_state": random.getstate(),
        "np_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": [
            torch.cuda.get_rng_state(device="cuda:" + str(i)) for i in range(torch.cuda.device_count())
        ],
    }
    torch.save(save_dict, save_path)
