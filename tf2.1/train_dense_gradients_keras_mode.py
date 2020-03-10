import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam


"""Training with Gradient Tape

    Pros:
        Very compact
        Lots of functionality implicit
        Speed: 11s per epoch

    Cons:
        Code not debuggeable
        Code not explicit
        Not flexible at all
"""


# Sanity check
if tf.__version__ != '2.1.0':
    raise ValueError('You need TensorFlow 2.1.0 to run this')


def loss(y_true, y_pred):
    return tf.square(y_true - y_pred)


if __name__ == '__main__':

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

    # Compiling model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=loss,
        metrics=['mse'])

    # Training
    model.fit(
        x=train_samples, y=train_targets,
        batch_size=1, 
        epochs=10,
        validation_data=(test_samples, test_targets),
        shuffle=True)