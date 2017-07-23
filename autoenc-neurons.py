#!/usr/bin/env python3

'''
!pip3 install -U numpy tensorflow-gpu keras
'''

import random

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv1D, Dense, MaxPooling1D, UpSampling1D
from keras.models import Model, Sequential
from keras.optimizers import RMSprop

# Cap maximum length of lines, for simplicity. Must be a multiple of pool_size ^ (num pool layers)
max_len = 80

# 0 for CPU, or >0 for GPUs
gpu_count = 0
# Dimension of embedding for individual characters in first, last layesr
embedding_dim = 32
# Filter depth of first 1D convolution
num_filters_1 = 30
# Filter depth of second 1D convolution
num_filters_2 = 20
# Filter depth of third 1D convolution
num_filters_3 = 10
# Kernel size for all convolutions
kernel_size = 5
# Pool size for all pooling
pool_size = 2

# Set up device to use for training
if gpu_count > 0:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    session = tf.Session(config=config)
    K.set_session(session)
    device1 = '/gpu:0'
    if gpu_count > 1:
        device2 = '/gpu:1'
    else:
        device2 = '/gpu:0'
else:
    device1 = '/cpu:0'
    device2 = '/cpu:0'


# Read lines, filter chars
lines = []
seen_chars = set()
with open('31100.txt', encoding='latin-1') as text_file:
    for line in text_file:
        if len(line) > 0:
            if len(line) > max_len:
                line = line[:max_len]
            lines.append(line)
            # record which characters appear in input
            for c in line:
                seen_chars.add(c)
# Shuffle input for good measure
random.shuffle(lines)

# Establish character mapping to index and its reverse
char_to_index = dict(zip(seen_chars, range(0, len(seen_chars))))
index_to_char = {i: c for c, i in char_to_index.items()}
num_chars = len(char_to_index)

print('Text has {} chars'.format(num_chars))

num_lines = len(lines)
# One-hot encode the input
one_hot = np.zeros([num_lines, max_len, num_chars])
for i in range(0, num_lines):
    line = lines[i]
    line_len = len(line)
    for j in range(0, line_len):
        one_hot[i, j, char_to_index.get(line[j])] = 1
    for j in range(line_len, max_len):
        one_hot[i, j, char_to_index.get(' ')] = 1

# See https://blog.keras.io/building-autoencoders-in-keras.html
# for more background on convolutional autoencoders in Keras

model = Sequential()
with tf.device(device1):
    # Learn embedding from one-hot encoding to vector space
    model.add(Dense(name='embedding',
                    input_shape=(max_len, num_chars),
                    units=embedding_dim,
                    activation='relu'))
    # Encode convolution + pool 1
    model.add(Conv1D(name='encoder_conv_1',
                     filters=num_filters_1,
                     kernel_size=kernel_size,
                     padding='same',
                     activation='relu'))
    model.add(MaxPooling1D(name='encoder_pool_1',
                           pool_size=pool_size))
    # Encode convolution + pool 2
    model.add(Conv1D(name='encoder_conv_2',
                     filters=num_filters_2,
                     kernel_size=kernel_size,
                     padding='same',
                     activation='relu'))
    model.add(MaxPooling1D(name='encoder_pool_2',
                           pool_size=pool_size))
    # Encode convolution + pool 3
    model.add(Conv1D(name='encoder_conv_3',
                     filters=num_filters_3,
                     kernel_size=kernel_size,
                     padding='same',
                     activation='relu'))
    model.add(MaxPooling1D(name='encoder_pool_3',
                           pool_size=pool_size))
with tf.device(device2):
    # Decode convolution + pool 3
    model.add(Conv1D(name='decoder_conv_3',
                     filters=num_filters_3,
                     kernel_size=kernel_size,
                     padding='same',
                     activation='relu'))
    model.add(UpSampling1D(name='decoder_unpool_3',
                           size=pool_size))
    # Decode convolution + pool 2
    model.add(Conv1D(name='decoder_conv_2',
                     filters=num_filters_2,
                     kernel_size=kernel_size,
                     padding='same',
                     activation='relu'))
    model.add(UpSampling1D(name='decoder_unpool_2',
                           size=pool_size))
    # Decode convolution + pool 1
    model.add(Conv1D(name='decoder_conv_1',
                     filters=num_filters_1,
                     kernel_size=kernel_size,
                     padding='same',
                     activation='relu'))
    model.add(UpSampling1D(name='decoder_unpool_1',
                           size=pool_size))
    # Learn mapping back from vector space to one-hot encoding
    # TODO weight tying?
    model.add(Dense(name='unembedding',
                    units=num_chars,
                    activation='softmax'))

model.summary()

# Compile and fit model
model.compile(optimizer=RMSprop(lr=0.01),
              loss='categorical_crossentropy',
              metrics=['acc'])

history = model.fit(one_hot, one_hot,
                    epochs=200,
                    batch_size=32,
                    verbose=2)

# Round-trip encode-decode the first line
predicted = model.predict(one_hot[:1])


# Examine the activations in the first encoder convolutional layer
activation_model = Model(inputs=[model.input], outputs=[model.layers[1].output])
activations = activation_model.predict(one_hot[:1])[0]

# Find scale of activations
min_act = np.min(activations)
act_range = np.max(activations) - min_act
levels = 5
cutoffs = list(map(lambda i: min_act + act_range * (i / (levels - 1)), range(0, levels)))
print(cutoffs)
display_chars = " .-=*@"
assert len(display_chars) == levels + 1

def print_activation(f):
    for i in range(0, levels):
        if f <= cutoffs[i]:
            return display_chars[i]
    return display_chars[-1]
  
# Print line, and its decoded form, and pretty-print neurons
decoded = ''.join(map(lambda a: index_to_char.get(a), np.argmax(predicted[0], axis=1)))
  
for neuron in range(0, num_filters_1):
    print(lines[0])
    print(decoded)
    print(''.join(map(print_activation, activations[:,neuron])))
    print()