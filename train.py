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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data
import tqdm

import imagenet_dataset
import upscale_module


def load_urls(path, num_urls):
    with path.open() as f:
        urls = []

        for line in itertools.islice(f, num_urls):
            url = line.strip().split()[1]
            print(url)
            urls.append(url)

    return urls


def init_weights(m):
    if type(m) == nn.Conv2d:
        if m.weight.shape[0] == 3:
            init.xavier_normal_(m.weight, gain=init.calculate_gain('tanh'))
        else:
            init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')

        m.bias.data.fill_(0.0)


def train_epoch(model, train_loader, optimizer, criterion):
    for X, y in tqdm.tqdm(train_loader):
        optimizer.zero_grad()

        y_hat = model(X)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()


def evaluate_epoch(model, val_loader, criterion):
    with torch.no_grad():
        running_loss = []

        for X, y in tqdm.tqdm(val_loader):
            y_hat = model(X)
            running_loss.append(criterion(y_hat, y).item())

    return np.mean(running_loss)


def main():
    print('loading images')

    dataset = imagenet_dataset.ImagenetDataset('./processed/')
    num_samples = min(len(dataset), 8192 * 5 // 4)
    num_train = num_samples * 4 // 5

    train_dataset = data.Subset(dataset, range(num_train))
    val_dataset = data.Subset(dataset, range(num_train, num_samples))

    train_loader = data.DataLoader(train_dataset, shuffle=True)
    val_loader = data.DataLoader(val_dataset, shuffle=True)

    model = upscale_module.UpscaleModule().float()
    model.apply(init_weights)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.001,
                          momentum=0.9)
    start_epoch = 0

    try:
        checkpoint = torch.load('model.tar')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

        print('restored model from epoch {}'.format(start_epoch))
    except Exception as e:
        print('couldn\'t restore model: {}'.format(e))
        pass

    for epoch in range(start_epoch, 30):
        print('training epoch {}'.format(epoch))

        train_epoch(model, train_loader, optimizer, criterion)

        print('evaluating epoch {}'.format(epoch))
        loss = evaluate_epoch(model, val_loader, criterion)
        print('epoch {} loss: {}'.format(epoch, loss))

        torch.save({'epoch': epoch, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                   'model.tar')


if __name__ == '__main__':
    main()
