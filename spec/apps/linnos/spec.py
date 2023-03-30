from dnnv.properties import *

N = Network("N") # symbolic values

Forall(
    x_,
    And(
        MonoDec(x_[4], N(x_)[0]),
        MonoDec(x_[8], N(x_)[0]),
    )
)
