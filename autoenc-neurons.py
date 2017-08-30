#!/usr/bin/env python3

'''
!pip3 install -U numpy tensorflow-gpu keras
'''

import random

from IPython.display import display, HTML
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
embedding_dim = 40
# Filter depth of first 1D convolution
num_filters_1 = 32
# Filter depth of second 1D convolution
num_filters_2 = 24
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
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

model.fit(one_hot, one_hot,
          epochs=20,
          batch_size=64,
          shuffle=True,
          verbose=2)

lines_to_predict = 5
# Round-trip encode-decode some lines
first_line = random.randrange(0, len(one_hot) - lines_to_predict)
predict_range = range(first_line, first_line + lines_to_predict)
one_hot_to_predict = one_hot[predict_range]
predicted = model.predict(one_hot_to_predict)

# Examine the activations in the first encoder convolutional layer
activation_model = Model(inputs=[model.input],
                         outputs=[model.layers[1].output, model.layers[3].output])
(activations_1, activations_2) = activation_model.predict(one_hot_to_predict)

# Find scale of activations
min_act_1 = np.min(activations_1)
act_range_1 = np.max(activations_1) - min_act_1
min_act_2 = np.min(activations_2)
act_range_2 = np.max(activations_2) - min_act_2


def print_activation(letter, act, minact, actrange):
    max_lum = 240
    lum = max_lum - int(max_lum * (act - minact) / actrange)
    return '''<span style="color:rgb({},{},{})">{}</span>'''.format(lum, lum, lum, letter)


#def print_activation_1(l, i, n, line):
#    return print_activation(line[i], activations_1[l, i, n], min_act_1, act_range_1)


def print_activation_2(l, i, n, line):
    return print_activation(line[i], activations_2[l, i // pool_size, n], min_act_2, act_range_2)


# Show second conv layer activations
for n in range(0, num_filters_2):
    print('Filter {}:'.format(n))
    for l in predict_range:
        line = lines[l]
        marked_up_letters = ''.join(map(lambda i: print_activation_2(l - first_line, i, n, line), range(0, len(line))))
        display(HTML('''<span style="font-family:monospace">''' + marked_up_letters + '''</span>'''))
    print()

# Print round-trip unencoded text as a sanity check
for p in predicted:
    print(''.join(map(lambda a: index_to_char.get(a), np.argmax(p, axis=1))))
