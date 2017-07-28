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

# Cap maximum length of lines, for simplicity. Must be a multiple of pool_size ^ (num pool layers)
max_len = 80

# 0 for CPU, or 1-2 for GPUs
gpu_count = 2
# Dimension of embedding for individual characters in first, last layers
embedding_dim = 32
# Filter depth of first 1D convolution
num_filters_1 = 50
# Filter depth of second 1D convolution
num_filters_2 = 30
# Kernel size for all convolutions
kernel_size = 3
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
        line = line.rstrip('\r\n')
        if len(line) > 0:
            if len(line) > max_len:
                line = line[:max_len]
            lines.append(line)
            # record which characters appear in input
            for c in line:
                seen_chars.add(c)
            if len(lines) >= 20000:
                break #TODO remove limit
# Shuffle input for good measure
random.shuffle(lines)

# Establish character mapping to index and its reverse
char_to_index = dict(zip(seen_chars, range(0, len(seen_chars))))
index_to_char = {i: c for c, i in char_to_index.items()}
num_chars = len(char_to_index)

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

model = Sequential()
with tf.device(device1):
    # Learn embedding from one-hot encoding to vector space
    model.add(Dense(name='embedding',
                    input_shape=(max_len, num_chars),
                    units=embedding_dim,
                    activation=None))
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
with tf.device(device2):
    # Decode convolution + pool 2
    model.add(UpSampling1D(name='decoder_unpool_2',
                           size=pool_size))
    model.add(Conv1D(name='decoder_conv_2',
                     filters=num_filters_2,
                     kernel_size=kernel_size,
                     padding='same',
                     activation='relu'))
    # Decode convolution + pool 1
    model.add(UpSampling1D(name='decoder_unpool_1',
                           size=pool_size))
    model.add(Conv1D(name='decoder_conv_1',
                     filters=num_filters_1,
                     kernel_size=kernel_size,
                     padding='same',
                     activation='relu'))

    # Learn mapping back from vector space to one-hot encoding
    # TODO weight tying?
    model.add(Dense(name='unembedding',
                    units=num_chars,
                    activation='softmax'))

model.summary()

# Compile and fit model
model.compile(optimizer='adadelta',
              loss='categorical_crossentropy',
              metrics=['acc'])

model.fit(one_hot, one_hot,
          epochs=20,
          batch_size=64,
          verbose=2)

lines_to_predict = 5
# Round-trip encode-decode the first line
predicted = model.predict(one_hot[:lines_to_predict])

# Examine the activations in the first encoder convolutional layer
activation_model = Model(inputs=[model.input],
                         outputs=[model.layers[1].output, model.layers[3].output])
activations = activation_model.predict(one_hot[:lines_to_predict])
activations_1 = activations[0]
activations_2 = activations[1]

# Find scale of activations
min_act_1 = np.min(activations_1)
act_range_1 = np.max(activations_1) - min_act_1
min_act_2 = np.min(activations_2)
act_range_2 = np.max(activations_2) - min_act_2

# Number of different activation strengths to display differently
levels = 5
# Max activations for each activation strength bucket
cutoffs_1 = list(map(lambda i: min_act_1 + act_range_1 * (i / (levels - 1)), range(0, levels)))
cutoffs_2 = list(map(lambda i: min_act_2 + act_range_2 * (i / (levels - 1)), range(0, levels)))
# Chars to use to render each activation
display_chars = " .-=*@"
assert len(display_chars) == levels + 1


def print_activation(f, cutoffs):
    for i in range(0, levels):
        if f <= cutoffs[i]:
            return display_chars[i]
    return display_chars[-1]

# How many different neurons/filtering in each conv layer to show
filters_to_show = 10

# Show first conv layer activations
for l in range(0, lines_to_predict):
    print(lines[l])
    for n in range(0, filters_to_show):
        print(''.join(map(lambda a: print_activation(a, cutoffs_1), activations_1[l,:,n])))
    print(''.join(map(lambda a: index_to_char.get(a), np.argmax(predicted[l], axis=1))))
    print()

# Show second conv layer activations
for l in range(0, lines_to_predict):
    print(lines[l])
    for n in range(0, filters_to_show):
        # * pool_size because next conv layer follows downsampling from pool
        # Each activation is a function of multiple inputs
        print(''.join(map(lambda a: print_activation(a, cutoffs_2) * pool_size, activations_2[l,:,n])))
    print(''.join(map(lambda a: index_to_char.get(a), np.argmax(predicted[l], axis=1))))
    print()