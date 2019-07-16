import mxnet as mx
from chapter3.utils import gdata
import numpy as np
from mxnet.gluon import loss as gloss
from chapter3.C_3_3_linreg import sgd
from chapter3.C_3_6_softmax import evaluate_accuracy

from mxnet import autograd
from mxnet import nd

# 1 数据
input_shape = 784
train_epochs = 10
num_outputs = 10
hidden_layers = 100
batch_size = 256
num_workers = 4
lr = 0.5
trainsformer = gdata.vision.transforms.ToTensor()
mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)
train_iter = gdata.DataLoader(mnist_train.transform_first(trainsformer),
                              batch_size, shuffle=True, num_workers=num_workers)
test_iter = gdata.DataLoader(mnist_test.transform_first(trainsformer),
                             batch_size, shuffle=True, num_workers=num_workers)

# 2 网络

w1 = nd.random.normal(scale=0.01, shape=(input_shape, hidden_layers))
b1 = nd.zeros((1, hidden_layers))

w2 = nd.random.normal(scale=0.01, shape=(hidden_layers, num_outputs))
b2 = nd.zeros((1, num_outputs))

params = [w1, b1, w2, b2]

w1.attach_grad()
b1.attach_grad()

w2.attach_grad()
b2.attach_grad()


def mlp(X):
    X = X.reshape((-1, input_shape))
    H = relu(nd.dot(X, w1) + b1)
    return nd.dot(H, w2) + b2


def relu(X):
    return nd.maximum(X, 0)

loss = gloss.SoftmaxCrossEntropyLoss()

for epoch in range(train_epochs):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        # 3 损失函数
        with autograd.record():
            y_logit = mlp(X)
            l = loss(y_logit, y).sum()

        # 4 方向传播
        l.backward()

        # 5 训练
        sgd(params, lr, batch_size)

        y = y.astype('float32')
        train_l_sum += l.asscalar()
        train_acc_sum += (y_logit.argmax(axis=1) == y).sum().asscalar()
        n += y.size

    test_acc = evaluate_accuracy(test_iter, mlp)
    print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' %
          (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
