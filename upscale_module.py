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


class SqueezeExcitation(nn.Module):
    def __init__(self, d, r=16):
        super(SqueezeExcitation, self).__init__()

        self.squeezer = nn.Linear(d, d // r)
        self.expander = nn.Linear(d // r, d)

    def forward(self, x):
        N, C, H, W = x.shape

        z = torch.mean(x, (2, 3))
        z = F.leaky_relu(self.squeezer(z))
        z = torch.sigmoid(self.expander(z))

        return x * torch.reshape(z, (N, C, 1, 1))


class Bottleneck(nn.Module):
    def __init__(self, d_in, d_out, C=32, r=16):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(d_in, d_out // 2, 1)
        self.conv2 = nn.Conv2d(d_out // 2, d_out // 2, 3, padding=1, groups=C)
        self.conv3 = nn.Conv2d(d_out // 2, d_out, 1)
        self.cse = SqueezeExcitation(d_out, r)

        if d_in != d_out:
            self.channel_matcher = nn.Conv2d(d_in, d_out, 1)
        else:
            self.channel_matcher = lambda x: x

    def forward(self, x):
        z = F.leaky_relu(self.conv1(x))
        z = F.leaky_relu(self.conv2(z))
        z = self.conv3(z)

        z = self.cse(z)

        return F.leaky_relu(z + self.channel_matcher(x))


class UpscaleModule(nn.Module):
    def __init__(self, depths=[64, 256, 256, 256, 256],
                 C=32):
        super(UpscaleModule, self).__init__()

        assert(len(depths) >= 2)

        self.input1 = nn.Conv2d(3, depths[0], 7, padding=3)
        self.input2 = nn.Conv2d(depths[0], depths[0], 3, padding=1)

        self.conv = nn.ModuleList(Bottleneck(depths[i - 1], depths[i], C=C)
                                  for i in range(1, len(depths)))

        self.output = nn.Conv2d(depths[-1], 12, 7, padding=3)

        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = F.leaky_relu(self.input1(x))
        x = F.leaky_relu(self.input2(x))

        for z in self.conv:
            x = z(x)

        x = self.output(x)
        x = self.shuffle(x)

        return torch.sigmoid(x)
