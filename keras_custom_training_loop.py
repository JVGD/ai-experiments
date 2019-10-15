import tensorflow as tf
import numpy as np
import keras.backend as K
from tqdm import tqdm
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense
from IPython import embed

# Setting seeds
np.random.seed(1)
tf.set_random_seed(1)


# Random dataset
samples = np.random.random((100,20))
targets = np.random.random((100,1))

# Model
x = Input(shape=[20])
y = Dense(units=1, activation='sigmoid')(x)
model = Model(x, y)

# Compiling
optimizer = Adam(lr=1e-4)

# Loss
def loss_fn(y_true, y_pred):
    loss_value = K.sum(K.pow((y_true - y_pred), 2))
    return loss_value

# Creatin training graphs
y_true = Input(shape=[20])
y_pred = model(x)
loss = loss_fn(y_pred, y_pred)
updates_op = optimizer.get_updates(params=model.trainable_weights, loss=loss)

train_fn = K.function(
    inputs=[x], 
    outputs=[y_pred], 
    updates=updates_op)

# Training Loop
i = 0
sample = np.expand_dims(samples[i], axis=0)
target = targets[i]
sample = K.variable(sample)
target = K.variable(target)

# Training
train_fn([sample])

for idx in tqdm(range(len(samples))):
    sample = samples[idx]
    target = targets[idx]

    # Adding batch dim since batch=1
    sample = np.expand_dims(sample, axis=0)

    # To tensors
    sample = K.constant(sample)
    target = K.constant(target)

    # Training
    target_pred = train_fn([sample])
    target_pred = target_pred[0]

    # Loss
    loss = loss_fn(target, target_pred)
    print(K.eval(loss))


# Or
# model.train_fn = K.function(
#     inputs=[x], 
#     outputs=[y_pred], 
#     updates=updates_op)


# embed()