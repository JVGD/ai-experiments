import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow_core.python import keras
from tensorflow_core.python.keras.layers import Input, Dense, Flatten, Conv2D
from tensorflow_core.python.keras import Model
from tensorflow_core.python.keras.optimizers import Adam


"""Training with Gradient Tape

    Pros:
        Very explicit forward and backward pass
        Crazy custom loss function
        That code is debuggable

    Cons:
        Batch size / shuffle implementation
        Custom callbacks implementation
        Custom metrics implementation
        No tensorboar / or linking manually
        Speed: 18s per epoch
"""


# Sanity check
if tf.__version__ != '2.1.0':
    raise ValueError('You need TensorFlow 2.1.0 to run this')


@tf.function
def loss_compute(y_true, y_pred):
    return tf.square(y_true - y_pred)


@tf.function
def forward_pass(model, sample, target):
    # Needs to be recorded by gradient tape
    with tf.GradientTape() as tape:
        target_pred = model(sample)
        loss = loss_compute(target, target_pred)

    # compute gradients w.r.t. the loss
    gradients = tape.gradient(loss, model.trainable_weights)

    return loss, gradients


@tf.function
def backward_pass(model, gradients, optimizer):
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))


def train():

    # Learn to sum 20 nums
    train_samples = tf.random.normal(shape=(10000, 20))
    train_targets = tf.reduce_sum(train_samples, axis=-1)
    test_samples = tf.random.normal(shape=(100, 20))
    test_targets = tf.reduce_sum(test_samples, axis=-1)

    # Model Functional API
    x = Input(shape=[20])
    h = Dense(units=20, activation='relu')(x)
    h = Dense(units=10, activation='relu')(h)
    y = Dense(units=1)(h)
    model = Model(x,y)

    # Training loop
    epochs = 10
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    for epoch in range(epochs):

        # Fancy progress bar
        pbar = tqdm(range(len(train_samples)))

        # Metrics
        loss_metric = keras.metrics.Mean()

        # Batches iteration, batch_size = 1
        for batch_id in pbar:

            # Getting sample target pair
            sample = train_samples[batch_id]
            target = train_targets[batch_id]

            # Adding batch dim since batch=1
            sample = tf.expand_dims(sample, axis=0)
            target = tf.expand_dims(target, axis=0)

            # This is computed in graph mode
            # Computing loss and gradients w.r.t the loss
            loss, gradients = forward_pass(model, sample, target)
            # Updaing model weights
            backward_pass(model, gradients, optimizer)

            # Tracking progress
            loss_metric(loss)
            pbar.set_description('Training Loss: %.3f' % 
                                 loss_metric.result().numpy())

        # At the end of the epoch test the model
        test_targets_pred = model(test_samples)
        test_loss = loss_compute(test_targets, test_targets_pred)
        test_loss_avg = tf.reduce_mean(test_loss)
        print('Validation Loss: %.3f' % test_loss_avg)


if __name__ == '__main__':
    train()
