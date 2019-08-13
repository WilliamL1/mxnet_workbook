# Batch Normalization

from d2lzh import d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import nn
import mxnet as mx


def batch_norm(X, gamma, bata, moving_mean, moving_var, eps, momentum):
    """
    moving_mean 在训练阶段并不使用，而是作为推理阶段的均值和方差，进行BN。
    在训练阶段，会对 X 通过求解具体的 mean 和 var 获得 BN。
    在训练阶段，卷积会对每个通道独立求均值和方差，并且该均值和方差作为当前通道进行处理。
    """
    if not autograd.is_training():
        # 训练时，使用移动均值和方差处理样本
        X_hat = (X - moving_mean) / nd.sqrt(moving_var + eps)
    else:
        # 预测时，使用小批量样本的平均均值和方差处理样本
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层时，计算特征维上的均值和方差
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # 使用二维卷积层的情况， 计算通道上（axis=1）的均值和方差。这里我们需要保持
            # X的形状以便后面做广播运算。
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)

        # 训练模式下用当前的均值和方差做标准化
        X_hat = (X - mean) / nd.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * mean
    Y = gamma * X_hat + bata  # 拉伸和偏移
    return Y, moving_mean, moving_var


class BatchNorm(nn.Block):
    def __init__(self, num_features, num_dims, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成0和1
        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        self.bate = self.params.get('beta', shape=shape, init=init.One())

        # 不参与求梯度和迭代的标量
        self.moving_mean = nd.zeros(shape)
        self.moving_var = nd.zeros(shape)

    def forward(self, X):
        # 如果X不在内存上，将 moving_mean 和 moving_var 复制到X所在的显存上
        if self.moving_mean.context != X.context:
            self.moving_mean = self.moving_mean.copyto(X.context)
            self.moving_var = self.moving_var.copyto(X.context)
        # 保存更新过的 moving_mean 和 moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma.data(), self.bate.data(), self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y


net = nn.Sequential()
net.add(
    nn.Conv2D(6, kernel_size=5),
    BatchNorm(6, num_dims=4),
    nn.Activation('sigmoid'),
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Conv2D(16, kernel_size=5),
    BatchNorm(16, num_dims=4),
    nn.Activation('sigmoid'),
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Dense(120),
    BatchNorm(120, num_dims=2),
    nn.Activation('sigmoid'),
    nn.Dense(84),
    BatchNorm(84, num_dims=2),
    nn.Activation('sigmoid'),
    nn.Dense(10)
)

lr, num_epochs, batch_size, ctx = 2.0, 5, 256, mx.cpu()
net.initialize(ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)

net2 = nn.Sequential()
net2.add(
    nn.Conv2D(6, kernel_size=5),
    nn.BatchNorm(),
    nn.Activation('sigmoid'),
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Conv2D(16, kernel_size=5),
    nn.BatchNorm(),
    nn.Activation('sigmoid'),
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Dense(120),
    nn.BatchNorm(),
    nn.Activation('sigmoid'),
    nn.Dense(86),
    nn.BatchNorm(),
    nn.Activation('sigmoid'),
    nn.Dense(10)
)

net2.initialize(ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net2.collect_params(), 'sgd', {'learning_rate': lr})
d2l.train_ch5(net2, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
