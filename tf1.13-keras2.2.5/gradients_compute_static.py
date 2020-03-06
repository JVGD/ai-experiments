import keras.backend as K
from IPython import embed

x = K.variable(3)
fx = K.pow(x, 2)
dfdx = K.gradients(fx, x)

# Evaluating graph
x = K.eval(x)
fx = K.eval(fx)
dfdx = K.eval(dfdx[0])
print('x={} fx={} df/dx={}'.format(x, fx, dfdx))

