# AlexNet

from mxnet import gluon, autograd, init, nd
from mxnet.gluon import nn, data as gdata
from d2lzh import d2l
import sys, os

# 5.6.1
net = nn.Sequential()
net.add(
    # layer1: 第一层卷积核为11，strides为4，无需填充，因为输入图片一般比较大，用较大的卷积尺寸。
    nn.Conv2D(96, kernel_size=11, strides=4, padding=0, activation='relu'),
    nn.MaxPool2D(pool_size=3, strides=2),

    # layer2: 第二层卷积核为5， strides为1，且用2像素进行填充，为了保持通道的尺寸，同时均匀卷积。
    # padding 一般取 kernel_size / 2 的下整，用于保持特征图尺寸不变。
    nn.Conv2D(256, kernel_size=5, strides=1, padding=2, activation='relu'),
    nn.MaxPool2D(pool_size=3, strides=2),

    # layer3、4、5， 进行更为细致的卷积，同时将padding 设置为 kernel_size/2 的下整，进而保持特征图尺寸不变。
    nn.Conv2D(384, kernel_size=3, strides=1, padding=1, activation='relu'),
    nn.Conv2D(384, kernel_size=3, strides=1, padding=1, activation='relu'),
    nn.Conv2D(256, kernel_size=3, strides=1, padding=1, activation='relu'),
    nn.MaxPool2D(pool_size=3, strides=2),

    # layer 6,7， 采用较大的神经元个数 4096 来学习特征表达，同时为防止过拟合，引入 dropout.
    nn.Dense(4096, activation='relu'),
    nn.Dropout(0.5),
    nn.Dense(4096, activation='relu'),
    nn.Dropout(0.5),

    # layer 8: 输出结果。
    nn.Dense(10)
)

X = nd.random.uniform(shape=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape: \t', X.shape)

"""
    
    1： conv0 output shape: 	     (1, 96, 54, 54)
        pool0 output shape: 	 (1, 96, 26, 26)
    2:  conv1 output shape: 	 (1, 256, 26, 26)
        pool1 output shape: 	 (1, 256, 12, 12)
    3： conv2 output shape: 	     (1, 384, 12, 12)
    4： conv3 output shape: 	     (1, 384, 12, 12)
    5： conv4 output shape: 	     (1, 256, 12, 12)
        pool2 output shape: 	 (1, 256, 5, 5)
    6： dense0 output shape: 	 (1, 4096)
        dropout0 output shape: 	 (1, 4096)
    7： dense1 output shape: 	 (1, 4096)
        dropout1 output shape: 	 (1, 4096)
    8： dense2 output shape: 	 (1, 10)
    pool 和 drop 层没有参数，因此不特定作为一层。
"""


# 5.6.2

def load_data_fashion_mnist(batch_size, resize=None, root=os.path.join(
    '~', '.mxnet', 'datasets', 'fashion-mnist')):
    root = os.path.expanduser(root)

    # 做了一个 transformer 的管道
    transformer = []
    if resize:
        transformer += [gdata.vision.transforms.Resize(resize)]
    transformer += [gdata.vision.transforms.ToTensor]
    transformer = gdata.vision.transforms.Compose(transformer)
    mnist_train = gdata.vision.FashionMNIST(root=root, train=True)
    mnist_test = gdata.vision.FashionMNIST(root=root, train=False)
    num_workers = 4
    train_iter = gdata.DataLoader(
        mnist_train.transform_first(transformer), batch_size, shuffle=True, num_workers=num_workers
    )
    test_iter = gdata.DataLoader(
        mnist_test.transform_first(transformer), batch_size, shuffle=False, num_workers=num_workers
    )

    return train_iter, test_iter


batch_size = 128
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)


# 5.6.4

lr, num_epochs, ctx  = 0.01, 5, d2l.try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})

from chapter5.C_5_5_LeNet import train_ch5
train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)


"""
epoch 1, loss 2.3163, train acc 0.105, test acc 0.100, time 14.2 sec
epoch 2, loss 1.3860, train acc 0.452, test acc 0.631, time 14.1 sec
epoch 3, loss 0.8735, train acc 0.659, test acc 0.713, time 14.2 sec
epoch 4, loss 0.7200, train acc 0.718, test acc 0.744, time 13.7 sec
epoch 5, loss 0.6461, train acc 0.746, test acc 0.766, time 15.3 sec
"""