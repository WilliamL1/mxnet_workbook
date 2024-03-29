"""
page: 169~ 175, 5 pages
time: 1 hour from 8:20am to 9:40am
"""

from d2lzh import d2l
from mxnet import nd
from mxnet.gluon import rnn

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size

ctx = d2l.try_gpu()


def get_params():
    def _one(shape):
        return nd.random.normal(scale=0.01, shape=shape, ctx=ctx)

    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                nd.zeros(num_hiddens, ctx=ctx))

    W_xz, W_hz, bz = _three()  # 更新门
    W_xr, W_hr, br = _three()  # 重置门
    W_xh, W_hh, bh = _three()  # 候选隐藏状态
    W_hq = _one((num_hiddens, num_hiddens))
    bq = nd.zeros(num_hiddens)

    params = [W_xz, W_hz, bz, W_xr, W_hr, br, W_xh, W_hh, bh, W_hq, bq]

    for param in params:
        param.attach_grad()
    return params


# 定义模型
## 定义隐藏层状态初始化

def init_gru_state(batch_size, num_hiddens, ctx):
    # 初始的隐藏层状态全部为0
    return (nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx),)


def gru(inputs, state, params):
    W_xz, W_hz, bz, W_xr, W_hr, br, W_xh, W_hh, bh, W_hq, bq = params
    H, = state
    outputs = []

    for X in inputs:
        # 重置门
        R = nd.sigmoid(nd.dot(X, W_xr) + nd.dot(H, W_hr) + br)
        # 更新门
        Z = nd.sigmoid(nd.dot(X, W_xz) + nd.dot(H, W_hz) + bz)
        # 候选隐藏状态
        H_tilda = nd.tanh(nd.dot(X, W_xh) + nd.dot(R * H, W_hh) + bh)
        # 隐藏状态
        H = Z * H + (1 - Z) * H_tilda
        # 输出
        Y = nd.dot(H, W_hq) + bq
        outputs.append(Y)
    return outputs, (H,)
