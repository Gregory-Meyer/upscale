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

import sys

import imageio
import numpy as np
import skimage
import torch
import torchvision.transforms.functional as F

import upscale_module


def main():
    if len(sys.argv) < 2:
        sys.exit("missing image path!")

    device = torch.device('cpu')
    dtype = torch.float

    model = upscale_module.UpscaleModule()
    checkpoint = torch.load('model.tar')
    model.load_state_dict(checkpoint['model'])
    model = model.to(device, dtype)

    with torch.no_grad():
        X = F.to_tensor(skimage.img_as_float32(imageio.imread(sys.argv[1]))) \
            .to(device, dtype)
        C, H, W = X.shape
        X = X.reshape(1, C, H, W)
        y = np.array(F.to_pil_image(model(X)[0, :, :, :]))

        imageio.imwrite('output.png', y, optimize=True)


if __name__ == '__main__':
    main()
