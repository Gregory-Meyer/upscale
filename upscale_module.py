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

import itertools

import torch
import torch.jit as jit
import torch.nn as nn
import torch.nn.functional as F


class UpscaleModule(jit.ScriptModule):
    __constants__ = ['conv']

    def __init__(self, depths=[8, 16, 32]):
        super(UpscaleModule, self).__init__()

        assert(len(depths) >= 1)

        self.conv = nn.ModuleList(itertools.chain(
            [nn.Conv2d(3, depths[0], 3, padding=1)],
            (nn.Conv2d(depths[i - 1], depths[i], 3, padding=1)
             for i in range(1, len(depths)))
        ))

        self.output = nn.Conv2d(depths[-1], 3, 3, padding=1)

    @jit.script_method
    def forward(self, x):
        x = self.upscale(x)

        for z in self.conv:
            x = F.relu(z(x))

        return self.output(x)

    @jit.script_method
    def upscale(self, x):
        N, C, H, W = x.shape
        upscaled = torch.zeros((N, C, H * 2, W * 2), dtype=torch.float32)

        for i in range(H):
            for j in range(W):
                upscaled[:, :, 2 * i, 2 * j] = x[:, :, i, j]

        return upscaled
