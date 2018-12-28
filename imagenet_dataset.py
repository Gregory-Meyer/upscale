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
import tqdm


class ImagenetDataset(data.Dataset):
    def __init__(self, folder):
        super(ImagenetDataset, self).__init__()

        print('loading image paths')
        self.paths = list(tqdm.tqdm(filter(lambda p: p.is_file(),
                                           pathlib.Path(folder).iterdir())))
        self.paths.sort(key=lambda p: p.stem)

        try:
            self.means, self.stds, num_images = \
                torch.load('dataset.tar', map_location=torch.device('cpu'))

            if num_images != len(self.paths):
                raise Exception('number of images doesn\'t match')
        except Exception as e:
            print('couldn\'t load normalization constants: {}'.format(e))

            self.means, self.stds = self._get_normalization_constants()
            torch.save((self.means, self.stds, len(self.paths)), 'dataset.tar')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = PIL.Image.open(self.paths[idx])

        tensor, decimated = ImagenetDataset._decimate_and_tensorify(img)
        decimated -= self.means
        decimated /= self.stds

        return decimated, tensor

    def _get_normalization_constants(self):
        means, num_pixels = self._get_channel_means_and_pixel_count()
        stds = self._get_channel_stds(means, num_pixels)

        return means, stds

    def _get_channel_means_and_pixel_count(self):
        num_pixels = 0
        means = torch.zeros((3, 1, 1), dtype=torch.float64)

        print('calculating channel means')
        for img in map(ImagenetDataset._open_f32_tensor,
                       tqdm.tqdm(self.paths)):
            means += torch.sum(img, (1, 2), keepdim=True).to(torch.float64)
            num_pixels += torch.numel(img[0, :, :])

        means /= float(num_pixels)

        return means.to(torch.float32), num_pixels

    def _get_channel_stds(self, means, num_pixels):
        variances = torch.zeros((3, 1, 1), dtype=torch.float64)

        print('calculating channel stds')
        for img in map(ImagenetDataset._open_f32_tensor,
                       tqdm.tqdm(self.paths)):
            variances += torch.sum(torch.pow(img - means, 2), (1, 2),
                                   keepdim=True).to(torch.float64)

        variances /= float(num_pixels - 1)
        stds = torch.sqrt(variances)

        return stds.to(torch.float32)

    def _decimate_and_tensorify(img):
        new_size = (img.height // 2, img.width // 2)
        decimated = F.resize(img, new_size, PIL.Image.LANCZOS)

        tensor = F.to_tensor(img)
        decimated_tensor = F.to_tensor(decimated)

        return tensor, decimated_tensor

    def _open_f32_tensor(path):
        return F.to_tensor(PIL.Image.open(path))
