""" 
Loads the 2821-playlists data set based on Spotify playlists containing over 100'000 songs.
Data gets formatted for the use with Keras and few-shot learning.
"""

import keras
from keras import utils
import numpy as np
import random
#from skimage.io import imread

spectrogram_path = "/home/tobia/Documents/ML/Data MA Sono"
spectrogram_type = ".png"
playlists_path = "data/Playlists.csv"
songfeatures_path = "data/GenreOld.csv"

def imread(path):
    return np.zeros((250, 40, 1))

"""
Imports a list of the songs in each playlist.
"""
def load_playlists():
    with open(playlists_path, "r") as f:
        return [np.array(x.strip().split(','))[2:] for x in f.readlines()]

"""
Loads the spectrogram for a specific Spotify URI (id).
"""
def load_spectrogram(uri, input_shape):
    spectrogram = imread(spectrogram_path + "/" + str(uri) + spectrogram_type) / 256

    max_height = input_shape[0]

    spectrogram = spectrogram[-max_height:]

    return spectrogram.reshape(input_shape)


features = {}

def load_songfeatures(uri, input_shape):
    global features

    if len(features) == 0:
        with open(songfeatures_path, "r") as f:
            for x in f.readlines():
                a = x.strip().split(',')                
                features[a[1]] = np.array(a[2:], dtype=np.float).reshape(input_shape) 
                features[a[1]] = np.minimum(1, np.power(np.maximum(0, features[a[1]]), 1/2))                

        print(len(features))

    return features[uri]


def load_songfeatures_grouped(train_ratio):
    playlists = load_playlists()

    random.shuffle(playlists)

    data_train = playlists[:int(train_ratio * len(playlists))]
    data_test = playlists[int(train_ratio * len(playlists)):]

    return data_train, data_test