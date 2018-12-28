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

import functools
import itertools
import math
import pathlib

import PIL
import torch
import torch.utils.data as data
import torchvision.transforms.functional as F
import tqdm


def _grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n

    return itertools.zip_longest(*args, fillvalue=fillvalue)


class ImagenetDataset(data.Dataset):
    def __init__(self, folder, examples_per_file=32, files_to_cache=16):
        super(ImagenetDataset, self).__init__()

        self.group_size = examples_per_file

        print('loading image paths')
        paths = list(tqdm.tqdm(filter(lambda p: p.is_file(),
                                      pathlib.Path(folder).iterdir())))
        paths.sort(key=lambda p: p.stem)
        self.num_examples = len(paths)

        try:
            self.means, self.stds, num_images = \
                torch.load('dataset.tar', map_location=torch.device('cpu'))

            if num_images != self.num_examples:
                raise Exception('number of images doesn\'t match')
        except Exception as e:
            print('couldn\'t load normalization constants: {}'.format(e))

            self.means, self.stds = self._get_normalization_constants(paths)
            torch.save((self.means, self.stds, self.num_examples),
                       'dataset.tar')

        print('caching and serializing')
        pathlib.Path('pickled').mkdir(parents=True, exist_ok=True)
        self.paths = self._serialize_all(paths)

        @functools.lru_cache(max_size=files_to_cache)
        def _get_line(idx):
            return torch.load(self.paths[idx])

        self._get_line = _get_line

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        if (idx < 0):
            idx += self.num_examples

        line = self._get_line(idx // self.examples_per_file)

        return line[idx % self.examples_per_file]

    def _serialize_all(self, paths):
        groups_iter = enumerate(_grouper(tqdm.tqdm(paths), self.group_size))
        serialized_paths = list(map(lambda ig: self._serialize_group(*ig),
                                    groups_iter))

        return serialized_paths

    def _serialize_group(self, idx, paths_group):
        to_serialize = list(map(lambda p: self._load_and_decimate(p),
                                paths_group))

        filename = 'pickled/{}.tar'.format(idx)
        torch.save(to_serialize, filename)

        return filename

    def _load_and_decimate(self, img_path):
        img = PIL.Image.open(img_path)

        tensor, decimated = \
            ImagenetDataset._decimate_and_tensorify(img)
        decimated -= self.means
        decimated /= self.stds

        return tensor, decimated

    def _get_normalization_constants(self, paths):
        means, num_pixels = self._get_channel_means_and_pixel_count(paths)
        stds = self._get_channel_stds(paths, means, num_pixels)

        return means, stds

    def _get_channel_means_and_pixel_count(self, paths):
        num_pixels = 0
        means = torch.zeros((3, 1, 1), dtype=torch.float64)

        print('calculating channel means')
        for img in map(ImagenetDataset._open_f32_tensor, tqdm.tqdm(paths)):
            means += torch.sum(img, (1, 2), keepdim=True).to(torch.float64)
            num_pixels += torch.numel(img[0, :, :])

        means /= float(num_pixels)

        return means.to(torch.float32), num_pixels

    def _get_channel_stds(self, paths, means, num_pixels):
        variances = torch.zeros((3, 1, 1), dtype=torch.float64)

        print('calculating channel stds')
        for img in map(ImagenetDataset._open_f32_tensor, tqdm.tqdm(paths)):
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
