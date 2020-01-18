import numpy as np
from keras.layers import LSTM, Input, Dense
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.optimizers import Adam
from keras.models import Model
from IPython import embed

# Conf data
time_step = 2
shuffle = False
batch_size = 1
features = 1

# Data should be 2D and first dim is time dimension
# seq_data : [TIME x FEATURE]
seq_data = np.vstack(np.arange(0, 10))
generator = TimeseriesGenerator(seq_data, seq_data, 
                                length=time_step, 
                                shuffle=shuffle, 
                                batch_size=batch_size)
# Generator test
# d0 : [BATCH x TIME x FEATURE]
d0, t0 = generator[0]
d1, t1 = generator[1]
embed()

# Model
x = Input(shape=[time_step, features])
h = LSTM(units=100)(x)
y = Dense(units=1)(h)
model = Model(x, y)

# Compiling
model.compile(optimizer='adam', loss='mse')

# Training
model.fit_generator(
    generator=generator, 
    epochs=50, 
    shuffle=True)

