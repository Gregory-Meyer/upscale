#!/usr/bin/env python3

import pathlib

import imageio
import skimage
import skimage.transform as transform
import torch.utils.data as data
import torchvision.transforms.functional as F


class ImagenetDataset(data.Dataset):
    def __init__(self, folder):
        super(ImagenetDataset, self).__init__()

        self.image_paths = list(pathlib.Path(folder).iterdir())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = skimage.img_as_float32(imageio.imread(path))
        downscaled = transform.rescale(img, 0.5, multichannel=True,
                                       anti_aliasing=True)

        return (F.to_tensor(downscaled), F.to_tensor(img))
