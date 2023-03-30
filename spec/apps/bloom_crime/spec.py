from dnnv.properties import *

f_nn = Network("N") # symbolic values

lat_bound = [0, 1]
long_bound = [0, 1]

Forall(
    x_,
    Implies(
        And(lat_bound[0] <= x_[0] <= lat_bound[1],
            long_bound[0] <= x_[1] <= long_bound[1]),
        f_nn(x_)[0] == 0),
    prob=80,
)
