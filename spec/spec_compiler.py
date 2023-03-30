import ast
import sys, os
cwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cwd+"/DNNV")

from functools import partial
from pathlib import Path

import onnx
from dnnv import properties, nn
from dnnv.verifiers.common.reductions import (
    IOPolytopeReduction,
    HalfspacePolytope,
    HyperRectangle,
)

from placeholder import create_placeholder_graph



def get_properties(spec_path, *, network_path=None, input_shape=None,
                   output_shape=None, property_invert=True):
    # prepare spec
    phi = properties.parse(spec_path)

    #print("------phi--------")
    #print(repr(phi))
    #print("-----------------")

    # prepare nn
    if network_path is not None:
        dnn = nn.parse(network_path)
        dnn = dnn.simplify()
    elif input_shape is not None and output_shape is not None:
        # from dnnv.nn.opeartions import Input
        #from dnnv.nn.graph import OperationGraph
        from dnnv.nn.parser.onnx import _parse_onnx_model
        # assemble an NN placeholder
        onnx_g = create_placeholder_graph(input_shape=input_shape, output_shape=output_shape)
        #onnx.save(onnx_g, "/tmp/xx.onnx")  # debug
        dnn = _parse_onnx_model(onnx_g)
    else:
        assert False, "No NN information; Cannot generate useful spec"

    # link nn to the spec
    phi.concretize(N=dnn)

    prob = phi.prob
    # transform spec to polytope
    specs = []
    reduction = IOPolytopeReduction()
    if property_invert:
        spec_gen = reduction.reduce_property(~phi)
    else:
        spec_gen = reduction.reduce_property(phi)
    for spec in spec_gen:
        specs.append(spec)

    return specs, prob

def parse_nn_inout_file(nn_path : Path):
    def sandbox_f(code):
        # FIXME: add security checks for code
        global_dic = {}
        exec(code, global_dic)
        assert global_dic["input_shape"] is not None
        assert global_dic["output_shape"] is not None
        return global_dic["input_shape"], global_dic["output_shape"]


    with open(nn_path, "r") as f:
        code = compile(source=f.read(),
                       filename="?", # FIXME
                       mode='exec')

    input_shape, output_shape = sandbox_f(code)
    return input_shape, output_shape



def main(spec_file, nn_file):
    spec_path = Path(spec_file)
    nn_path = Path(nn_file)

    assert spec_path.exists(), "ERROR: spec file doesn't exist"
    assert nn_path.exists(), "ERROR, nn file doesn't exist"

    if nn_path.suffix == ".onnx":
        specs, prob = get_properties(spec_path, network_path=nn_path)
    else:
        # my NN-input-output file
        in_shape, out_shape = parse_nn_inout_file(nn_path)
        specs, prob = get_properties(spec_path, input_shape = in_shape, output_shape=out_shape)

    for i in range(len(specs)):
        print("=== inverted property [%d]" % i)
        spec = specs[i]
        print(spec)
    print("=== prob: [%d]" % prob)



if __name__ == "__main__":
    def usage():
        print("spec_compiler.py <spec_file> <nn_file>")

    if len(sys.argv) not in [3]:
        usage()
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
