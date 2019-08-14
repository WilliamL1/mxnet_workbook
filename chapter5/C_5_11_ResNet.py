# ResNet Block
import mxnet
from mxnet.gluon import nn
from mxnet import gluon, init, nd
from d2lzh import d2l


class Residual(nn.Block):
    # 残差块
    def __init__(self, channels, use_1x1conv=False, strides=1, **kwargs):
        """

        :param channels: int
        :param use_1x1conv: boolean
        :param strides: int
        """
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(channels=channels, kernel_size=3, padding=1, strides=strides)
        self.conv2 = nn.Conv2D(channels=channels, kernel_size=3, padding=1)

        if use_1x1conv:
            # 这个 1x1conv的作用是reshape。
            self.conv3 = nn.Conv2D(channels=channels, kernel_size=1, padding=0, strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, X):
        Y = nd.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return nd.relu(Y + X)


def test_residual():
    blk = Residual(3)
    blk.initialize()
    X = nd.random.uniform(shape=(4, 3, 6, 6))
    Y = blk(X)
    print(Y.shape)

    blk2 = Residual(6, strides=2, use_1x1conv=True)
    blk2.initialize()
    Y2 = blk2(X)
    print(Y2.shape)


def residual_block(num_channels, num_residuals, first_block=False):
    # 残差模块
    blk = nn.Sequential()
    for i in range(num_residuals):
        if not first_block and i == 0:
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk


def residual_network():
    # 共18层，称为 ResNet-18，  共包括 1个卷积 + 4x2x2 个卷积( 4个残差模块，每个模块2个残差块，每个残差块2个卷积，共 4x2x2)
    net = nn.Sequential()

    net.add(
        nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1)
    )

    # 每个模块包括2个残差快，第一个模块的通道数同输入通道数一致。
    # 之后每个模块的第1个残差块，将上个模块的通道数翻倍，同时将高和宽减半。
    net.add(
        residual_block(64, 2, first_block=True),
        residual_block(128, 2),
        residual_block(256, 2),
        residual_block(512, 2)
    )

    # 之后接全局平均池化层后接上全连接层输出。
    net.add(
        nn.GlobalAvgPool2D(),
        nn.Dense(10)
    )

    return net


def test_resnet_18():
    X = nd.random.uniform(shape=(1, 1, 224, 224))
    net = residual_network()
    net.initialize()
    for layer in net:
        X = layer(X)
        print(layer.name, ': output shape : ', X.shape)


def train_resnet_18():
    net = residual_network()
    net.initialize(force_reinit=True, init=init.Xavier())

    lr, batch_size, ctx, epoch_num = 1.0, 256, mxnet.cpu(), 5
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
    d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, epoch_num)


if __name__ == '__main__':
    # test_residual()
    # test_resnet_18()
    train_resnet_18()
