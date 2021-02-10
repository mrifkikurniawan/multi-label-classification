# built-in
from typing import Union, Dict, Callable
from collections import namedtuple
import os.path as osp

# library
import numpy as np
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset

# local library
from mlc.utils import encode_multihot

__all__ = ['HPASingleCellClassification']
CHANNELS = ["blue", "green", "red", "yellow"]
IMG_FORMAT = 'png'

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

class HPASingleCellClassification(Dataset):
    def __init__(self,
                 root: str,
                 metadata: str,
                 num_class: int,
                 transform: Callable = None,
                 target_transform: Callable = None):
    
        self.root = root
        self.metadata_path = metadata
        self.transform = transform
        self.target_transform = target_transform
        self.num_class = num_class

        # read the metadata()
        metadata = pd.read_csv(self.metadata_path)

        # convert label from str to int
        metadata["Label"] = metadata.Label.str.split('|').apply(lambda x: [int(i) for i in x])

        self.metadata = metadata.to_dict(orient="records")
    

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index: int):
        # vars
        img_id = self.metadata[index]["ID"]

        # read image
        # concate all channels into one
        for i, channel in enumerate(CHANNELS):
            img_name = f"{img_id}_{channel}.{IMG_FORMAT}"
            img_path = osp.join(self.root, img_name)
            img = pil_loader(img_path)
            img_width, img_height = img.size

            if i == 0:
                img_np = np.zeros((img_height, img_width, 4), dtype=np.int16)
            
            # convert to numpy
            img_np[:,:,i] = np.asarray(img.getdata(), dtype=np.int8).reshape(img_height, img_width)

        # transform image
        if self.transform:
            img_np = self.transform(img_np)

        # read label and convert to multi-hot
        label = self.metadata[index]["Label"]
        label = encode_multihot(label, num_class=self.num_class)

        return img_np, label