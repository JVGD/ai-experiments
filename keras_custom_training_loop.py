import tensorflow as tf
import numpy as np
import keras.backend as K
from tqdm import tqdm
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense
from IPython import embed

# Setting seeds
np.random.seed(0)
tf.set_random_seed(0)

# Sum 2 numbers from 0 to 10 dataset
samples = np.random.randint(0, 9, size=(100,2))
targets = np.sum(samples, axis=-1)

# Features
features = samples.shape[1]

# Model
x = Input(shape=[features])
y = Dense(units=1)(x)
model = Model(x, y)

# Loss
def loss_fn(y_true, y_pred):
    loss_value = K.sum(K.pow((y_true - y_pred), 2))
    return loss_value

# Compiling
optimizer = Adam(lr=1e-4)

# Graph creation
# Creatin training graphs
y_true = Input(shape=[0])
y_pred = model(x)
loss = K.sum(K.pow((y_true - y_pred), 2))
updates_op = optimizer.get_updates(params=model.trainable_weights, loss=loss)

train = K.function(
    inputs=[x, y_true], 
    outputs=[loss], 
    updates=updates_op)

test = K.function(
    inputs=[x, y_true], 
    outputs=[loss])


# Training Loop
i = 0
sample = np.expand_dims(samples[i], axis=0)
target = targets[i]
sample = K.variable(sample)
target = K.variable(target)

embed()


# Training
# train([sample])

# Training loop
epochs = 100

for epoch in range(epochs):
    pbar = tqdm(range(len(samples)))
    for idx in pbar:
        sample = samples[idx]
        target = targets[idx]

        # Adding batch dim since batch=1
        sample = np.expand_dims(sample, axis=0)

        # To tensors
        sample = K.constant(sample)
        target = K.constant(target)

        # Training
        loss_value = train([sample, target])

        pbar.set_description('Loss %s' % loss_value[0])
        # print(K.eval(loss))

    # # Testing
    # samples_test = np.random.randint(0, 9, size=(3,2))
    # for sample in samples_test:
    #     # Adding batch dim since batch=1
    #     sample = np.expand_dims(sample, axis=0)

    #     # To tensors
    #     sample = K.constant(sample)

    #     # Prediction
    #     target_pred = model(sample)

    #     # To numpy
    #     sample = K.eval(sample)
    #     target_pred = K.eval(target_pred)
    #     print("Test: sample: %s Pred: %s" % (sample, target_pred))




# Or
# model.train_fn = K.function(
#     inputs=[x], 
#     outputs=[y_pred], 
#     updates=updates_op)


# embed()