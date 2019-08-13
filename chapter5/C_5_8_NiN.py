from mxnet.gluon import nn, Trainer
from mxnet import nd
from d2lzh import d2l


def evaluate_accuracy(data_iter, net, ctx):
    acc_sum, n = nd.array([0], ctx=ctx), 0
    for X, y in data_iter:
        # 如何ctx 代表GPU及相应的显存，将数据复制到显存上
        X, y = X.as_in_context(ctx), y.as_in_context(ctx).astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum()
        n += y.size
    return acc_sum.asscalar() / n


def nin_block(num_channels, kernel_size, strides, padding):
    blk = nn.Sequential()
    blk.add(
        nn.Conv2D(num_channels, kernel_size=kernel_size,
                  strides=strides, padding=padding, activation='relu'),
        nn.Conv2D(num_channels, kernel_size=1, strides=1, padding=0, activation='relu'),
        nn.Conv2D(num_channels, kernel_size=1, strides=1, padding=0, activation='relu')
    )
    return blk


net = nn.Sequential()

net.add(
    nin_block(96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2D(pool_size=3, strides=2),
    nin_block(256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2D(pool_size=3, strides=2),
    nin_block(384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2D(pool_size=3, strides=2),
    nn.Dropout(0.5),
    nin_block(10, kernel_size=3, strides=1, padding=1),
    nn.GlobalAvgPool2D(),
    nn.Flatten()
)

X = nd.random.uniform(shape=(1, 1, 224, 224))

net.initialize()
for blk in net:
    X = blk(X)
    print(blk.name, 'blk shape :\t', X.shape)

"""
    sequential1 blk shape :	 (1, 96, 54, 54)
    pool0 blk shape :	 (1, 96, 26, 26)
    sequential2 blk shape :	 (1, 256, 26, 26)
    pool1 blk shape :	 (1, 256, 12, 12)
    sequential3 blk shape :	 (1, 384, 12, 12)
    pool2 blk shape :	 (1, 384, 5, 5)
    dropout0 blk shape :	 (1, 384, 5, 5)
    sequential4 blk shape :	 (1, 10, 5, 5)
    pool3 blk shape :	 (1, 10, 1, 1)
    flatten0 blk shape :	 (1, 10)
"""

lr, num_epochs, batch_size = 0.01, 5, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
trainer = Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
from mxnet.gluon import loss as gloss
from mxnet import autograd

loss = gloss.SoftmaxCrossEntropyLoss()

for _ in range(num_epochs):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backword()
        trainer.step(batch_size)
        y = y.astype('float32')
        train_l_sum += l.asscalar()
        train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
        n += y.size
    test_acc = evaluate_accuracy(test_iter, net, ctx)
    print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
          % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc, time.time() - start))


def nin_block(num_channel, strides, kernel_size, padding):
    blk = nn.Sequential()
    blk.add(
        nn.Conv2D(channels=num_channel, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu'),
        nn.Conv2D(channels=num_channel, kernel_size=1, strides=1, padding=0, activation='relu'),
        nn.Conv2D(channels=num_channel, kernel_size=1, strides=1, padding=0, activation='relu')
    )
    return blk
