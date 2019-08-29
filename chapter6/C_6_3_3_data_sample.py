from d2lzh.d2l import data_iter_random

my_seq = list(range(30))

for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
    print("X: ", X, "Y: ", Y)
