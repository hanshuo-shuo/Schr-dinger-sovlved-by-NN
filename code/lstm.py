#!/usr/bin/env python
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv', usecols=[1])
# plt.plot(data)
# plt.show()

dataset = data.dropna().values.astype('float32')

max_value = np.max(dataset)
min_value = np.min(dataset)
dataset = (dataset - min_value) / (max_value-min_value)
print(dataset.shape)


def create_dataset(dataset, look_back=10):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i+look_back])
    return np.array(dataX), np.array(dataY)


X, Y = create_dataset(dataset)
print(X.shape, Y.shape)

train_size = int(len(X) * 0.7)
valid_size = len(X) - train_size
print(train_size, valid_size)

X_train = X[:train_size]
Y_train = Y[:train_size]

X_valid = X[train_size:]
Y_valid = Y[train_size:]

X_train = X_train.transpose(1, 0, 2)
X_valid = X_valid.transpose(1, 0, 2)

X_train = torch.from_numpy(X_train)
Y_train = torch.from_numpy(Y_train)
X_valid = torch.from_numpy(X_valid)


class LSTMRegression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, cn) = self.lstm(x)
        hn = hn.squeeze()
        out = self.linear(hn)
        return out


model = LSTMRegression(input_size=1, hidden_size=4, output_size=1)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

epochs = 5000
for epoch in range(epochs):
    out = model(X_train)
    loss = criterion(out, Y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch: {epoch:5d}, Loss: {loss.item():.4e}')


# test
X = X.transpose(1, 0, 2)
X = torch.from_numpy(X)
Y_pred = model(X)
Y_pred = Y_pred.view(-1).data.numpy()

plt.plot(Y_pred, 'r', label='prediction')
plt.plot(Y, 'b-', label='groundtruth')
plt.legend(loc='best')
plt.show()
