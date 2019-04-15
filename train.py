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
import multiprocessing

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import tqdm


import imagenet_dataset
import upscale_module


device = torch.device('cuda')
dtype = torch.float32
torch.backends.cudnn.enabled = True


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
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        y_hat = model(X)

        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()


def evaluate_epoch(model, val_loader, criterion):
    with torch.no_grad():
        losses = torch.empty(len(val_loader), device=device, dtype=dtype)

        for i, (X, y) in enumerate(tqdm.tqdm(val_loader)):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            y_hat = model(X)

            losses[i] = criterion(y_hat, y)

    return torch.mean(losses).item()


def main():
    torch.cuda.is_available()
    dataset = imagenet_dataset.ImagenetDataset('./processed/')
    num_train = len(dataset) * 4 // 5
    num_val = len(dataset) - num_train

    train_dataset, val_dataset = \
        data.random_split(dataset, (num_train, num_val))

    num_workers = multiprocessing.cpu_count() * 5
    train_loader = data.DataLoader(train_dataset, shuffle=True,
                                   num_workers=num_workers,
                                   pin_memory=True)
    val_loader = data.DataLoader(val_dataset, num_workers=num_workers,
                                 pin_memory=True)

    model = upscale_module.UpscaleModule().to(device, dtype)
    model.apply(init_weights)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0001,
                          momentum=0.9)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    start_epoch = 0

    try:
        checkpoint = torch.load('model.tar', map_location=device)
        model.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']

        print('restored model from epoch {}'.format(start_epoch))
    except Exception as e:
        print('couldn\'t restore model: {}'.format(e))

    for epoch in range(start_epoch, 2048):
        print('evaluating epoch {}'.format(epoch))
        loss = evaluate_epoch(model, val_loader, criterion)
        scheduler.step(loss)
        print('epoch {} loss: {}'.format(epoch, loss))

        print('training epoch {}'.format(epoch + 1))
        train_epoch(model, train_loader, optimizer, criterion)

        torch.save({'epoch': epoch + 1, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()},
                   'model.tar')


if __name__ == '__main__':
    main()
