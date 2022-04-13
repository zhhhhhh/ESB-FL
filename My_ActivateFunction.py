import numpy as np
import tensorflow as tf
import math
from tensorflow.python.framework import ops

import seal_ckks

#def square(x):
#    x_t_x = seal_ckks.c_times_c(x, x)
#    x_t_p = seal_ckks.c_times_p(x, 0.5)
#    x2_a_x = seal_ckks.c_add_c(x2_t_p, x_t_p)
#    X2_a_x_a_p = c_add_p(x2_a_x, 0.3)
#    return X2_a_x_a_p
#def square_grad(x):
#    return seal_ckks.c_add_p(seal_ckks.c_times_p(x, 0.25), 0.5)
    
    
def square(x):
    return 0.125 * pow(x, 2) + 0.5 * x +log(2)
def square_grad(x):
    return 0.25 * x + 0.5

square_np = np.vectorize(square)
square_grad_np = np.vectorize(square_grad)

square_np_32 = lambda x: square_np(x).astype(np.float32)
square_grad_np_32 = lambda x: square_grad_np(x).astype(np.float32)

def square_grad_tf(x, name=None):
    with ops.name_scope(name, "square_grad_tf", [x]) as name:
        y = tf.py_func(square_grad_np_32, [x], [tf.float32], name=name, stateful=False)
        return y[0]

def my_py_func(func, inp, Tout, stateful=False, name=None, my_grad_func=None):
    # need to generate a unique name to avoid duplicates:
    random_name = "PyFuncGrad" + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(random_name)(my_grad_func)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": random_name, "PyFuncStateless": random_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def _square_grad(op, pred_grad):
    x = op.inputs[0]
    cur_grad = square_grad_tf(x)
    next_grad = pred_grad * cur_grad
    return next_grad

def square_tf(x, name=None):
    with ops.name_scope(name, "square_tf", [x]) as name:
        y = my_py_func(square_np_32,
                       [x],
                       [tf.float32],
                       stateful=False,
                       name=name,
                       my_grad_func=_square_grad)
    return y[0]
