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

import imageio
import magic
import skimage.color as color


def main():
    dirpath = pathlib.Path('./data/')
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
            img = imageio.imread(str(file.resolve()))
        except Exception as e:
            print('skipping {}: {}'.format(file.name, e), file=sys.stderr)

        if img.shape[0] < 32 or img.shape[1] < 32:
            continue

        transformed_path = to_copy / '{:0>6}.png'.format(i)

        if len(img.shape) == 2:
            img = color.gray2rgb(img)
        elif len(img.shape) == 3 and img.shape[2] == 1:
            img = color.gray2rgb(img[:, :, 0])
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img = img[:, :, :3]

        if len(img.shape) != 3 or (len(img.shape) == 3 and img.shape[2] != 3):
            import pdb
            pdb.set_trace()

            pass

        if img.shape[0] % 2 != 0:
            img = img[:-1, :, :]

        if img.shape[1] % 2 != 0:
            img = img[:, :-1, :]

        try:
            print('{} -> {}: {}'.format(file, transformed_path, img.shape[:2]))
            imageio.imwrite(transformed_path, img, optimize=True)
            i += 1
        except Exception as e:
            print('\033[0;31merror writing: {}\033[0m'.format(e))


if __name__ == '__main__':
    main()
