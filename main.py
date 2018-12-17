#!/usr/bin/env python3

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
        init.kaiming_normal_(m.weight, nonlinearity='relu')
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

        for X, y in val_loader:
            y_hat = model(X)
            running_loss.append(criterion(y_hat, y).item())

    return np.mean(running_loss)


def main():
    print('loading images')

    dataset = imagenet_dataset.ImagenetDataset('./processed/')
    num_samples = min(len(dataset), 8192)
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
        loss = evaluate_epoch(model, val_loader, criterion)

        print('epoch {} loss: {}'.format(epoch, loss))

        torch.save({'epoch': epoch, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                   'model.tar')


if __name__ == '__main__':
    main()
