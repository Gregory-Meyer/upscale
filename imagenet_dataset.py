#!/usr/bin/env python3

# BSD 3-Clause License
#
# Copyright (c) 2018, Gregory Meyer
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import pathlib

import PIL
import skimage.util as util
import torch
import torch.utils.data as data
import torchvision.transforms.functional as F


class ImagenetDataset(data.Dataset):
    def __init__(self, folder):
        super(ImagenetDataset, self).__init__()

        self.image_paths = list(pathlib.Path(folder).iterdir())
        self._calculate_normalization_constants()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self._get_image(idx)
        downsampled = ImagenetDataset._downsample(img)

        return (self._normalize(downsampled), ImagenetDataset._to_tensor(img))

    def _get_image(self, idx):
        path = self.image_paths[idx]

        return PIL.Image.open(str(path.resolve()))

    def _calculate_normalization_constants(self):
        self.means, num_pixels = self._calculate_image_mean_and_pixel_count()
        self.stds = self._calculate_image_std(num_pixels)

    def _calculate_image_mean_and_pixel_count(self):
        num_pixels = 0
        means = torch.zeros(3, dtype=torch.float64)

        for img in self._get_tensors_iter():
            means += torch.sum(img, (1, 2))
            num_pixels += torch.numel(img[0, :, :])

        means /= float(num_pixels)

        return means, num_pixels

    def _calculate_image_std(self, num_pixels):
        variances = torch.zeros(3, dtype=torch.float64)

        for img in self._get_tensors_iter():
            variances += torch.sum(torch.pow(img - self.means, 2))

        variances /= float(num_pixels - 1)
        stds = torch.sqrt(variances)

        return stds

    def _get_images_iter(self):
        return (PIL.Image.open(str(p.resolve())) for p in self.image_paths)

    def _get_tensors_iter(self):
        return (ImagenetDataset._to_f64_tensor(img)
                for img in self.get_images_iter())

    def _downsample(img):
        return img.resize((img.width // 2, img.height // 2), PIL.Image.LANCZOS)

    def _normalize(self, img):
        as_tensor = ImagenetDataset._to_tensor(img)

        as_tensor -= self.means
        as_tensor /= self.stds

        return as_tensor

    def _to_tensor(img):
        return F.to_tensor(util.img_as_float32(img))

    def _to_f64_tensor(img):
        return F.to_tensor(util.img_as_float64(img))
