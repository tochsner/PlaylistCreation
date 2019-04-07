"""
Trains a simple quadruplet cross-digit encoder on music similarity.
"""

import numpy as np
from data.playlists import load_songfeatures_grouped
from helper.prepare_triplets import create_training_data_for_quadruplet_loss
from models.simple_genre_model import build_model
from helper.losses_similarity import Losses
from helper.dataset_tools import OutputHelper
from keras.optimizers import SGD

input_shape = (276, 1,1)
input_lenght = np.prod(input_shape)
embedding_lenght = 20

output_helper = OutputHelper(embedding_lenght, input_lenght)

decoder_factor = 0.6

lr = 0.0015
momentum = 0.9
epochs = 20
samples_per_epoch = 10000
batch_size = 20
number_test_samples = 5000
train_ratio = 0.7

losses = Losses(output_helper, decoder_factor=decoder_factor)

data_train, data_test = load_songfeatures_grouped(input_lenght, train_ratio)


def trainModel():
    print("Start with", lr, momentum)
    model = build_model(input_shape, embedding_lenght)
    model.compile(SGD(lr, momentum=momentum), losses.quadruplet_loss, metrics=[losses.quadruplet_metric])

    for e in range(epochs):
        (x_test, y_test) = create_training_data_for_quadruplet_loss(model, data_test, number_test_samples, output_helper, uses_min_distance=False)
        (x_train, y_train) = create_training_data_for_quadruplet_loss(model, data_train, number_test_samples, output_helper, uses_min_distance=False)
        print(model.evaluate(x_train, y_train, verbose=0)[1], model.evaluate(x_test, y_test, verbose=0)[1])

        for b in range(samples_per_epoch // batch_size):
            (x_train, y_train) = create_training_data_for_quadruplet_loss(model, data_train, batch_size, output_helper)
            model.fit(x_train, y_train, epochs=1, verbose=0)

lr = 0.0001
momentum = 0
trainModel()

lr = 0.0001
momentum = 0.5
trainModel()

lr = 0.0001
momentum = 0.9
trainModel()

lr = 0.0001
momentum = 0.99
trainModel()

lr = 0.0003
momentum = 0
trainModel()

lr = 0.0003
momentum = 0.5
trainModel()

lr = 0.0003
momentum = 0.9
trainModel()

lr = 0.0003
momentum = 0.99
trainModel()

lr = 0.0003
momentum = 0
trainModel()

lr = 0.0003
momentum = 0.5
trainModel()

lr = 0.0003
momentum = 0.9
trainModel()

lr = 0.0003
momentum = 0.99
trainModel()

lr = 0.0006
momentum = 0
trainModel()

lr = 0.0006
momentum = 0.5
trainModel()

lr = 0.0006
momentum = 0.9
trainModel()

lr = 0.0006
momentum = 0.99
trainModel()

lr = 0.001
momentum = 0
trainModel()

lr = 0.001
momentum = 0.5
trainModel()

lr = 0.001
momentum = 0.9
trainModel()

lr = 0.001
momentum = 0.99
trainModel()

lr = 0.002
momentum = 0
trainModel()

lr = 0.002
momentum = 0.5
trainModel()

lr = 0.002
momentum = 0.9
trainModel()

lr = 0.002
momentum = 0.99
trainModel()

lr = 0.004
momentum = 0
trainModel()

lr = 0.004
momentum = 0.5
trainModel()

lr = 0.004
momentum = 0.9
trainModel()

lr = 0.004
momentum = 0.99
trainModel()