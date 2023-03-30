from dnnv.properties import *

f_nn = Network("N")    # learned index
key_0 = 10; pos_0 = 1  # db[key_0] => pos_0
key_1 = 12; pos_1 = 2  # db[key_1] => pos_1
epsilon = Parameter("epsilon", int, default=100) # error bound

Forall(
    x_,
    Implies(
        (key_0 <= x_ <= key_1),  # [10,12]
        (pos_0 - epsilon <= f_nn(x_) <= pos_1 + epsilon), # [-99, 102]
    ),
)
