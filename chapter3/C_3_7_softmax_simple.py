# import d2lzh as d2l
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn
from chapter3 import utils

# 1. 定义超参
batch_size = 256
num_epochs = 5

# 2. 定义数据
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)

# 3. 定义网络
net = nn.Sequential()
net.add(nn.Dense(10))

# 4. 初始化网络
net.initialize(init.Normal(sigma=0.01))

# 5. 定义损失函数
loss = gloss.SoftmaxCrossEntropyLoss()

# 6. 定义优化器及学习率
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

# 7. 训练
utils.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, trainer)