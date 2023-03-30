import numpy as np
import onnx
from onnx import helper, TensorProto, checker

def create_placeholder_graph(*, input_shape=[1],
                             output_shape=[1],
                             input_dtype=TensorProto.FLOAT,
                             output_dtype=TensorProto.FLOAT,):

    X = helper.make_tensor_value_info('X', input_dtype, input_shape)
    Y = helper.make_tensor_value_info('Y', output_dtype, output_shape)
    y_shape = np.array(output_shape, dtype=np.int64)

    n1 = helper.make_node(
        'Constant',
        inputs = [],
        outputs = ['Y_shape'],
        value = helper.make_tensor(
            name='y_shape_const',
            data_type=TensorProto.INT64,
            dims=y_shape.shape,
            vals=y_shape.flatten().astype(int)
        )
    )

    n2 = helper.make_node(
        'Reshape',                  # op_type
        inputs = ['X', 'Y_shape'],
        outputs = ['Y']
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [n1, n2],        # nodes
        'placeholder-NN',  # name
        [X],               # inputs
        [Y],               # outputs
    )

    # Create the model (ModelProto)
    model = helper.make_model(graph_def, producer_name='ouroboros')

    return model

if __name__ == "__main__":
    onnx_g = create_placeholder_graph(input_shape=[2,2], output_shape=[4])
    checker.check_model(onnx_g)
    onnx.save(onnx_g, "/tmp/xx.onnx")

    import onnxruntime as rt
    sess = rt.InferenceSession("/tmp/xx.onnx")
    #input_name = sess.get_inputs()[0].name
    pred_onx = sess.run(None, {'X': np.array([[20,21],[1,2]], dtype=np.float32)})
    print(pred_onx)
