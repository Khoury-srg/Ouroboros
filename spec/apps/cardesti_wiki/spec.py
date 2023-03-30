# cardinality estimation, wiki
from dnnv.properties import *

N = Network("N") # symbolic values

Forall(
    x_,
    And(
        MonoDec(x_[0], N(x_)[0]),
        MonoInc(x_[1], N(x_)[0]),
        MonoDec(x_[2], N(x_)[0]),
        MonoInc(x_[3], N(x_)[0]),
    )
)
