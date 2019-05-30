"""
Trains a simple quadruplet cross-song encoder on music similarity.
"""

import numpy as np
from random import shuffle
from data.playlists import load_playlists
from helper.dataset_tools import OutputHelper, split_list
from helper.prepare_quadruplets import TriletCreator
from models.simple_songfeatures_model import build_model
from helper.losses_similarity import Losses
from keras.optimizers import SGD, Adam
from helper.hyperparameters import HyperparameterLogger

input_shape = (406, 1, 1)
input_lenght = np.prod(input_shape)
decoder_output_lenght = input_lenght
embedding_lenght = 20

output_helper = OutputHelper(embedding_lenght, decoder_output_lenght)

decoder_factor = 0.6

lr = 0.01
momentum = 0.8
decay = 0.0001

epochs = 30
batch_size = 32
samples_per_epoch = 10000
number_test_samples = 5000
train_ratio = 0.7

losses = Losses(output_helper, decoder_factor=decoder_factor)

playlists = load_playlists()
data_train, data_test = split_list(playlists, train_ratio)

def train_model():
	print("Start with adam")
	model = build_model(input_shape, embedding_lenght)
	model.compile(Adam(0.01), losses.quadruplet_loss, metrics=[losses.quadruplet_metric])

	triplet_creator = TriletCreator(model, input_shape, output_helper)

	for e in range(epochs):
		(x_test, y_test) = triplet_creator.create_training_data_for_quadruplet_loss(data_test, number_test_samples, use_min_distance=False)
		(x_train, y_train) = triplet_creator.create_training_data_for_quadruplet_loss(data_train, number_test_samples, use_min_distance=False)
	
		for b in range(samples_per_epoch // batch_size):
			(x_train, y_train) = triplet_creator.create_training_data_for_quadruplet_loss(data_train, batch_size)
			model.fit(x_train, y_train, epochs=1, verbose=0)

train_model()
