# GoogleNet

from mxnet import gluon, init, nd
from mxnet.gluon import nn
import mxnet as mx
from d2lzh import d2l



class Inception(nn.Block):
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1, 1x1卷积
        self.p1_1 = nn.Conv2D(c1, kernel_size=1, strides=1, padding=0, activation='relu')

        # 线路2, 1x1卷积, 3x3卷积层
        self.p2_1 = nn.Conv2D(c2[0], kernel_size=1, strides=1, padding=0, activation='relu')
        self.p2_2 = nn.Conv2D(c2[1], kernel_size=3, strides=1, padding=1, activation='relu')

        # 线路3, 1x1卷积，5x5卷积
        self.p3_1 = nn.Conv2D(c3[0], kernel_size=1, strides=1, padding=0, activation='relu')
        self.p3_2 = nn.Conv2D(c3[1], kernel_size=5, strides=1, padding=2, activation='relu')

        # 线路4, 3x3最大池化层，1x1卷积层
        self.p4_1 = nn.MaxPool2D(pool_size=3, strides=1, padding=1)
        self.p4_2 = nn.Conv2D(c4, kernel_size=1, strides=1, padding=0, activation='relu')

    def forward(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))

        return nd.concat(p1, p2, p3, p4, dim=1)


b1 = nn.Sequential()
b1.add(
    nn.Conv2D(64, kernel_size=7, strides=2, padding=3, activation='relu'),
    nn.MaxPool2D(pool_size=3, strides=2, padding=1)
)

b2 = nn.Sequential()
b2.add(
    nn.Conv2D(64, kernel_size=1, strides=1, padding=0, activation='relu'),
    nn.Conv2D(192, kernel_size=3, strides=1, padding=1, activation='relu'),
    nn.MaxPool2D(pool_size=3, strides=2, padding=1)
)

# 第三模块串联2个Inception块，
#   第一个Inception块的输出通道数为 64 + 128 + 32 + 32 = 256, 其中4条线路的输出通道比例为 64:128:32:32 = 2:4:1:1
#   第二个Inception块的输出通道数为 128 + 192 + 96 + 64 = 480，每条线路的输出通道之比为 128：192：96：64 = 4：6：3：2
b3 = nn.Sequential()
b3.add(
    Inception(64, (96, 128), (16, 32), 32),
    Inception(128, (128, 192), (32, 96), 64),
    nn.MaxPool2D(pool_size=3, strides=2, padding=1)
)

# 第四模块串联5个Inception块，
b4 = nn.Sequential()
b4.add(
    Inception(192, (96, 208), (16, 48), 64),
    Inception(160, (112, 224), (24,64), 64),
    Inception(128, (128, 256), (24, 64), 64),
    Inception(112, (144, 288), (32, 64), 64),
    Inception(256, (160, 320), (32, 128), 128),
    nn.MaxPool2D(pool_size=3, strides=2, padding=1)
)

# 第五模块串联2个Inception块
b5 = nn.Sequential()
b5.add(
    Inception(256, (160, 320), (32, 128), 128),
    Inception(384, (192, 384), (48, 128), 128),
    nn.GlobalAvgPool2D())

net = nn.Sequential()
net.add(b1, b2, b3, b4, b5, nn.Dense(10))
#
# X = nd.random.uniform(shape=(1,1,96,96))
# net.initialize()
# for layer in net:
#     X = layer(X)
#     print(X.shape)

lr, num_epochs, batch_size, ctx = 0.1, 5, 128, mx.cpu()
net.initialize()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=(96))
d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)