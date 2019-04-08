"""
Trains a simple quadruplet cross-song encoder on music similarity.
"""

import numpy as np
from random import shuffle
from data.playlists import load_playlists
from helper.dataset_tools import OutputHelper
from helper.prepare_quadruplets import create_training_data_for_quadruplet_loss
from models.simple_genre_model import build_model
from helper.losses_similarity import Losses
from keras.optimizers import SGD

input_shape = (250, 40, 1)
input_lenght = np.prod(input_shape)
decoder_output_lenght = 180
embedding_lenght = 20

output_helper = OutputHelper(embedding_lenght, decoder_output_lenght)

decoder_factor = 0.6

lr = 0.0015
momentum = 0.99

epochs = 20
batch_size = 20
samples_per_epoch = 10000
number_test_samples = 5000
train_ratio = 0.7

losses = Losses(output_helper, decoder_factor=decoder_factor)

playlists = load_playlists()
shuffle(playlists)
data_train = playlists[:int(train_ratio * len(playlists))]
data_test = playlists[int(train_ratio * len(playlists)):]

def trainModel():
    print("Start with (lr, momentum):", lr, momentum)
    model = build_model(input_shape, embedding_lenght, decoder_output_lenght)
    model.compile(SGD(lr, momentum=momentum), losses.quadruplet_loss, metrics=[losses.quadruplet_metric])

    for e in range(epochs):
        (x_test, y_test) = create_training_data_for_quadruplet_loss(model, data_test, number_test_samples, input_shape, output_helper, use_min_distance=False)
        (x_train, y_train) = create_training_data_for_quadruplet_loss(model, data_train, number_test_samples, input_shape, output_helper, use_min_distance=False)
        print(model.evaluate(x_train, y_train, verbose=0)[1], model.evaluate(x_test, y_test, verbose=0)[1])

        for b in range(samples_per_epoch // batch_size):
            (x_train, y_train) = create_training_data_for_quadruplet_loss(model, data_train, batch_size, input_shape, output_helper)
            model.fit(x_train, y_train, epochs=1, verbose=0)

trainModel()
