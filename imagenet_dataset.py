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
import pathlib

import PIL
import torch
import torch.utils.data as data
import torchvision.transforms.functional as F
import tqdm


class ImagenetDataset(data.Dataset):
    def __init__(self, folder):
        super(ImagenetDataset, self).__init__()

        self._paths = _load_paths(folder)
        self._means, self._stds = \
            _load_or_get_normalization_constants(self._paths)

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, idx):
        return _load_and_decimate(self._means, self._stds, self._paths[idx])


def _load_paths(folder):
    print('loading image paths')

    paths = list(tqdm.tqdm(filter(lambda p: p.is_file(),
                                  pathlib.Path(folder).iterdir())))
    paths.sort(key=lambda p: p.stem)

    return paths


def _load_or_get_normalization_constants(paths, constants_path='dataset.tar'):
    num_examples = len(paths)

    try:
        print('loading normalization constants')
        means, stds, num_images = torch.load(constants_path)

        if num_images != num_examples:
            raise Exception('number of images doesn\'t match')
    except Exception as e:
        print('couldn\'t load normalization constants: {}'.format(e))
        print('calculating normalization constants')

        means, stds = _channelwise_mean_and_std(paths)
        torch.save((means, stds, num_examples), constants_path)

    return means, stds


def _load_and_decimate(means, stds, img_path):
    img = PIL.Image.open(img_path)

    tensor, decimated = _decimate_and_tensorify(img)
    decimated -= means
    decimated /= stds

    return decimated, tensor


def _decimate_and_tensorify(img):
    new_size = (img.height // 2, img.width // 2)
    decimated = F.resize(img, new_size, PIL.Image.LANCZOS)

    tensor = F.to_tensor(img)
    decimated_tensor = F.to_tensor(decimated)

    return tensor, decimated_tensor


def _channelwise_mean_and_std(img_paths):
    channel_means, num_pixels = _channelwise_mean_and_pixel_count(img_paths)
    channel_stds = _channelwise_std(img_paths, channel_means, num_pixels)

    return channel_means, channel_stds


def _channelwise_mean_and_pixel_count(img_paths):
    imgs = map(_open_as_f64, tqdm.tqdm(img_paths))
    sum_and_pixel_counts = map(_sum_and_pixel_count, imgs)

    print('calculating channelwise means')
    channelwise_totals, pixel_count = \
        functools.reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]),
                         sum_and_pixel_counts)
    channelwise_means = channelwise_totals / float(pixel_count)

    return torch.reshape(channelwise_means, (-1, 1, 1)), pixel_count


def _channelwise_std(img_paths, channelwise_means, num_pixels):
    imgs = map(_open_as_f64, tqdm.tqdm(img_paths))
    differences = map(lambda i: i - channelwise_means, imgs)
    squared_differences_sum = map(_sum_of_squares, differences)

    print('calculating channelwise standard deviations')
    channelwise_vars = sum(squared_differences_sum) / float(num_pixels - 1)
    channelwise_stds = torch.sqrt(channelwise_vars)

    return torch.reshape(channelwise_stds, (-1, 1, 1))


def _sum_and_pixel_count(img):
    C, H, W = img.shape

    return torch.sum(img, (1, 2)), H * W


def _sum_of_squares(img):
    return torch.sum(torch.pow(img, 2), (1, 2))


def _open_as_f64(path):
    return _open_as_f32(path).double()


def _open_as_f32(path):
    img = PIL.Image.open(path)

    return F.to_tensor(img)
