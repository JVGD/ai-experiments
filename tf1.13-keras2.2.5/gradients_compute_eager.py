import keras.backend as K
import tensorflow as tf
from IPython import embed

if tf.__version__[0] == '1':
    tf.enable_eager_execution()

with tf.GradientTape() as g:
    x = tf.constant(3.0)
    g.watch(x)
    fx = tf.pow(x,2)

dfdx = g.gradient(fx, x)

# Evaluating graph
print(tf.__version__)
print('x={} fx={} df/dx={}'.format(x, fx, dfdx))

