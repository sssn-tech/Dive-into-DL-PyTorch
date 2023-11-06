import torch
import matplotlib.pyplot as plt
import numpy as np
import random

num_inputs = 5
N = 3000
true_w = torch.tensor([1.23, 2.34, -3.45, -4.56, 5.67]).reshape(num_inputs, -1)
true_b = -6.78
features = torch.rand(size=(N, num_inputs)) * 10; # torch.rand()默认0-1
labels = torch.matmul(features, true_w) + true_b
noise = torch.normal(size=labels.size(), mean=0, std=0.05)

def net(X, w, b):
    return torch.matmul(X, w) + b

def loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def data_iter(batch_size, features, labels):
    N = len(features)
    indices = list(range(N))
    random.shuffle(indices)
    for i in range(0, N, batch_size):
        # min的作用是防止切片越界, 因为最后一次迭代不一定够一个batch
        select = torch.LongTensor(indices[i: min(i + batch_size, N)])
        # yield是迭代器关键字
        yield features.index_select(0, select), labels.index_select(0, select)

net_w = torch.normal(mean=0, std=0.5, size=true_w.shape).requires_grad_(requires_grad=True)
net_b = torch.normal(mean=0, std=0.5, size=(1, 1)).requires_grad_(requires_grad=True)

def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size

lr = 0.03
num_epochs = 20
batch_size = 10

for epoch in range (num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        y_hat = net(X, net_w, net_b)
        l = loss(y_hat, y).sum()
        l.backward()
        sgd([net_w, net_b], lr, batch_size)

        # 梯度清零
        net_w.grad.data.zero_()
        net_b.grad.data.zero_()
    train_l = loss(net(features, net_w, net_b), labels)
    print('epoch %d: loss %f' %(epoch + 1, train_l.mean().item()))