import mxnet as mx
from mxnet.gluon import loss as gloss
from mxnet.gluon import data as gdata

from mxnet import nd
from mxnet import autograd

# 3.3.1 生成数据
num_inputs = 2
num_examples = 1000
true_w = [2, -3.5]
true_b = 4.9
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = nd.dot(features, nd.array(true_w)) + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

# 3.3.2 读取数据
batch_size = 10
dataset = gdata.ArrayDataset(features, labels)
data_iter = gdata.DataLoader(dataset, batch_size=batch_size, shuffle=True)

if not False:
    for x,y in data_iter:
        print(x,y)
        break

# 3.3.3 定义<模型>

from mxnet.gluon import nn
net = nn.Sequential()
net.add(nn.Dense(1))

# 3.3.4 初始化<模型参数>

from mxnet import init
net.initialize(init.Normal(sigma=0.01))

# 3.3.5 定义<损失函数>

from mxnet.gluon import loss as gloss

loss = gloss.L2Loss()

# 3.3.6 定义<优化函数>

from mxnet import gluon
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.03})

# 3.3.7 训练<模型>

num_epoch = 6
for epoch in range(num_epoch):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('epoch {}, loss: {}'.format(epoch, l.mean().asnumpy()))

dense = net[0]
print("\ntrue_w {}, \nw {}".format(true_w, dense.weight.data()))
print("\ntrue_b {}, \nb {}".format(true_b, dense.bias.data()))