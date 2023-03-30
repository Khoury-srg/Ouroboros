from dnnv.properties import *

# symbolic values
N = Network("N")
epsilon = Parameter("errorbound", int, default=5)

# FIXME: fake db
key_space = 1000
block_size = 10
pos_space = int(key_space / block_size)
db = {
    i : int(i/block_size)
    for i in range(key_space)
}

def h_true_pos(key) -> int:
    if key < 0:
        return 0
    elif key > key_space:
        return int(key_space/block_size)
    else:
        return db[int(key)]

def f_partition():
    input_ranges = []
    prev_key = 0
    for key in range(1, key_space):
        input_ranges.append((prev_key, key))
        prev_key = key
    return input_ranges

def f_expectation(x):
    return h_true_pos(x)

@symbolic
def f_constraint(x_, x_range, f_exp):
    lower_x = x_range[0]
    upper_x = x_range[1]
    true_pos_lower = f_exp(lower_x)
    true_pos_upper = f_exp(upper_x)
    return Implies(
        # input constraint
        (lower_x <= x_[0] <= upper_x),
        # output constraint
        (true_pos_lower - epsilon <= \
         N(x_) <= true_pos_upper + epsilon),
    )


# the whole spec
Forall(
    x_,
    And(
        [f_constraint(x_, x_range, f_expectation)
        for x_range in f_partition()]
    )
)
