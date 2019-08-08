# VGG block
from mxnet.gluon import nn
from mxnet import nd
from d2lzh import d2l
import mxnet as mx
from chapter5.C_5_5_LeNet import train_ch5


def vgg_block(num_convs, num_channels):
    """
    vgg block:
    连续使用数个相同的填充为1，窗口形状为3x3的卷积层，后接上一个步幅为2，窗口为2x2的最大池化层。
    卷积层保持宽和高不变，池化层保证对其减半。
    :param num_convs: 卷积层数
    :param num_channels: 输出通道数
    :return:
    """
    blk = nn.Sequential()
    for _ in range(num_convs):
        blk.add(
            nn.Conv2D(channels=num_channels, kernel_size=3, padding=1, activation='relu'),
        )
    blk.add(nn.MaxPool2D(pool_size=2, strides=2))
    return blk


vgg_11 = (
    (1, 64),
    (1, 128),
    (2, 256),
    (2, 512),
    (2, 512)
)


def vgg(conv_arch):
    net = nn.Sequential()

    def vgg_block(num_convs, num_channels):
        blk = nn.Sequential()
        for _ in range(num_convs):
            blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1, activation='relu'))
        blk.add(nn.MaxPool2D(pool_size=2, strides=2))
        return blk

    for num_conv, num_channel in conv_arch:
        net.add(vgg_block(num_conv, num_channel))

    net.add(
        nn.Dense(4096, activation='relu'),
        nn.Dropout(0.5),
        nn.Dense(4096, activation='relu'),
        nn.Dropout(0.5),
        nn.Dense(10)
    )
    return net


def display_vgg(net):
    X = nd.random.uniform(shape=(1, 1, 224, 224))
    for layer in net:
        X = layer(X)
        print(layer.name, 'output shape:\t', X.shape)
    """
    sequential1 output shape:	 (1, 64, 112, 112)
    sequential2 output shape:	 (1, 128, 56, 56)
    sequential3 output shape:	 (1, 256, 28, 28)
    sequential4 output shape:	 (1, 512, 14, 14)
    sequential5 output shape:	 (1, 512, 7, 7)
    dense0 output shape:	 (1, 4096)
    dropout0 output shape:	 (1, 4096)
    dense1 output shape:	 (1, 4096)
    dropout1 output shape:	 (1, 4096)
    dense2 output shape:	 (1, 10)
    """


def train_small_vgg():
    ratio = 4
    small_vgg_11 = [(pair[0], pair[1] // ratio) for pair in vgg_11]
    net = vgg(small_vgg_11)
    lr, num_epochs, batch_size, ctx = 0.01, 20, 256, mx.cpu()
    net.initialize()
    # mx.gluon 训练器中，定义需要更新的参数 net.collect_params(), 更新算法 sgd, 以及学习率 lr
    trainer = mx.gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)


if __name__ == '__main__':
    # 5.7.2
    # net = vgg(vgg_11)
    # net.initialize(force_reinit=True)
    # display_vgg(net)

    # 5.7.3
    train_small_vgg()
