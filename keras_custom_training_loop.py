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

# Samples for testing
samples_test = np.random.randint(0, 9, size=(10,2))
targets_test = np.sum(samples_test, axis=-1)


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

# embed()

# Training loop
epochs = 100

for epoch in range(epochs):
    print('Epoch %s:' % epoch)
    pbar = tqdm(range(len(samples)))
    losses_train = []
    for idx in pbar:
        sample = samples[idx]
        target = targets[idx]

        # Adding batch dim since batch=1
        sample = np.expand_dims(sample, axis=0)
        target = np.expand_dims(target, axis=0)

        # To tensors
        sample = K.constant(sample)
        target = K.constant(target)

        # Evaluation of train graph
        loss_train = train([sample, target])
        
        # Compute loss mean
        losses_train.append(loss_train[0])
        loss_train_mean = np.mean(losses_train)
        
        # Update progress bar
        pbar.set_description('Train Loss %s' % loss_train_mean)

    # Testing
    losses_test = []
    for idx in range(len(samples_test)):
        sample_test = samples_test[idx]
        target_test = targets_test[idx]

        # Adding batch dim since batch=1
        sample_test = np.expand_dims(sample_test, axis=0)
        target_test = np.expand_dims(target_test, axis=0)

        # To tensors
        sample_test = K.constant(sample_test)
        target_test = K.constant(target_test)
        
        # Evaluation test graph
        loss_test = test([sample_test, target_test])
        # print("Test Loss: %s" % loss_test[0])
        # print("Test: sample: %s Pred: %s" % (K.eval(sample_test), K.eval(target_test)))
        
        # Compute test loss mean
        losses_test.append(loss_test[0])
    
    loss_test_mean = np.mean(losses_test)
    print('Test Loss: %s' % loss_test_mean)
