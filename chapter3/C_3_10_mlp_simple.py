from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn
from chapter3 import utils


batch_size = 256
num_epochs = 5

train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dense(10))

net.initialize(init.Normal(sigma=0.01))

loss = gloss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
num_epoch = 5
utils.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, trainer)