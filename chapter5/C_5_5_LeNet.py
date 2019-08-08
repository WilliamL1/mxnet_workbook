# LeNet
from mxnet.gluon import nn, loss as gloss
from mxnet import nd, autograd, init, gluon

import d2lzh.d2l as d2l
import mxnet as mx
import time

# 5.5.1
net = nn.Sequential()
net.add(
    # layer 1
    nn.Conv2D(channels=6, kernel_size=5, strides=1, padding=0, activation='sigmoid'),
    nn.MaxPool2D(pool_size=2, strides=2),

    # layer 2
    nn.Conv2D(16, (5, 5), strides=1, padding=0, activation='sigmoid'),
    nn.MaxPool2D(pool_size=2, strides=2),

    # layer 3、4、5
    nn.Dense(120, activation='sigmoid'),
    nn.Dense(84, activation='sigmoid'),
    nn.Dense(10)
)


def test_551():
    X = nd.random.uniform(shape=(1, 1, 28, 28))
    net.initialize()
    for layer in net:
        X = layer(X)
        print(layer.name, 'output shape: \t', X.shape)


'''
    conv0 output shape: 	 (1, 6, 24, 24)
    pool0 output shape: 	 (1, 6, 12, 12)
    conv1 output shape: 	 (1, 16, 8, 8)
    pool1 output shape: 	 (1, 16, 4, 4)
    dense0 output shape: 	 (1, 120)
    dense1 output shape: 	 (1, 84)
    dense2 output shape: 	 (1, 10)
    Process finished with exit code 0
'''

# 5.5.2

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)


def try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx


ctx = try_gpu()


def evaluate_accuracy(data_iter, net, ctx):
    acc_sum, n = nd.array([0], ctx=ctx), 0
    for X, y in data_iter:
        # 如何ctx 代表GPU及相应的显存，将数据复制到显存上
        X, y = X.as_in_context(ctx), y.as_in_context(ctx).astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum()
        n += y.size
    return acc_sum.asscalar() / n


def train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs):
    """

    :param net: 网络
    :param train_iter: 训练集
    :param test_iter: 测试集
    :param batch_size: 块大小
    :param trainer: 训练器
    :param ctx: 内容
    :param num_epochs: 迭代次数
    :return:
    """

    print('training on', ctx)
    # 定义损失函数
    loss = gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        # 当前 epoch
        for X, y in train_iter:
            # 将 X 分配至 ctx 所在设备
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            # 前项传播并记录梯度
            with autograd.record():
                y_logit = net(X)
                # 计算损失
                l = loss(y_logit, y).sum()
            # 反向传播
            l.backward()
            # trainer.step:
            # Makes one step of parameter update. Should be called after
            # `autograd.backward()` and outside of `record()` scope.
            trainer.step(batch_size)
            y = y.astype('float32')
            # 获得 loss 总数
            train_l_sum += l.asscalar()
            # 获得准确预测总数
            train_acc_sum += (y_logit.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc, time.time() - start))


lr, num_epochs = 0.9, 5
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)

"""
training on cpu(0)
epoch 1, loss 2.3135, train acc 0.112, test acc 0.186, time 16.3 sec
epoch 2, loss 1.3181, train acc 0.481, test acc 0.635, time 15.6 sec
epoch 3, loss 0.8433, train acc 0.675, test acc 0.718, time 15.3 sec
epoch 4, loss 0.7113, train acc 0.721, test acc 0.735, time 16.6 sec
epoch 5, loss 0.6379, train acc 0.747, test acc 0.769, time 16.4 sec
"""
