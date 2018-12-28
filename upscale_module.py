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

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, d_in, d_out, C=32):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(d_in, d_out // 2, 1)
        self.conv2 = nn.Conv2d(d_out // 2, d_out // 2, 3, padding=1, groups=C)
        self.conv3 = nn.Conv2d(d_out // 2, d_out, 1)

        if d_in != d_out:
            self.channel_matcher = nn.Conv2d(d_in, d_out, 1)
        else:
            self.channel_matcher = lambda x: x

    def forward(self, x):
        z = F.leaky_relu(self.conv1(x), inplace=True)
        z = F.leaky_relu(self.conv2(z), inplace=True)
        z = self.conv3(z)

        return F.leaky_relu(z + self.channel_matcher(x), inplace=True)


class UpscaleModule(nn.Module):
    def __init__(self, depths=[64, 256, 256, 256], C=32):
        super(UpscaleModule, self).__init__()

        assert(len(depths) >= 2)

        self.input1 = nn.Conv2d(3, depths[0], 7, padding=3)
        self.input2 = nn.Conv2d(depths[0], depths[0], 3, padding=1)

        self.conv = nn.ModuleList(Bottleneck(depths[i - 1], depths[i], C=C)
                                  for i in range(1, len(depths)))

        # directly upscale to rgb output
        self.output = nn.ConvTranspose2d(depths[-1], 3, 3, 2, 1, 1)

    def forward(self, x):
        x = F.leaky_relu(self.input1(x), inplace=True)
        x = F.leaky_relu(self.input2(x), inplace=True)

        for z in self.conv:
            x = z(x)

        x = torch.tanh(self.output(x))  # in range [-1, 1]

        return x / 2.0 + 0.5  # transform to range [0, 1]
