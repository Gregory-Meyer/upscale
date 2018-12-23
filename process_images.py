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
import sys

import magic
import numpy as np
from PIL import Image


def round_to_nearest(x):
    sizes = np.array([64, 96, 128, 192, 256, 384, 512, 768, 1024],
                     dtype=np.int_)
    is_less = x <= sizes

    if np.count_nonzero(is_less) == 0:
        return sizes[-1]

    return sizes[is_less][0]


def main():
    dirpath = pathlib.Path('./images/')
    to_copy = pathlib.Path('./processed/')

    if not to_copy.is_dir():
        to_copy.mkdir(parents=True)

    valid_types = {'image/png', 'image/jpeg', 'image/gif'}
    forbidden = ['error', 'unknown', 'notfound', 'not-found',
                 'not_found', 'unavailable']

    def is_valid(path):
        if not path.is_file():
            return False

        filename = str(path.name).lower()

        for substr in forbidden:
            if substr in filename:
                return False

        mime_type = magic.from_file(str(path.resolve()), mime=True)

        return mime_type in valid_types

    i = 0
    for file in filter(is_valid, dirpath.iterdir()):
        try:
            img = Image.open(str(file.resolve()))
            img = img.convert(mode='RGB')

            if img.height < 256 or img.width < 256:
                raise Exception('shape {} too small'.format(img.size))

        except Exception as e:
            print('\033[0;31mskipping {}: {}\033[0m'.format(file.name, e),
                  file=sys.stderr)

            continue

        transformed_path = to_copy / '{:0>6}.png'.format(i)

        while img.height > 1024 or img.width > 1024:
            img = img.resize(np.array(img.size) // 2, resample=Image.LANCZOS)

        if img.height % 2 != 0:
            new_height = img.height - 1
        else:
            new_height = img.height

        if img.width % 2 != 0:
            new_width = img.width - 1
        else:
            new_width = img.width

        img = img.crop((0, 0, new_height, new_width))

        try:
            img.save(str(transformed_path), optimize=True)
        except Exception as e:
            print('\033[0;31merror writing to {}: {}\033[0m'
                  .format(transformed_path, e), file=sys.stderr)

            continue

        print('{} -> {}: {}'.format(file, transformed_path, img.size))
        i += 1


if __name__ == '__main__':
    main()
