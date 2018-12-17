#!/usr/bin/env python3

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
