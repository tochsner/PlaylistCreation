""" 
Loads the 2821-playlists dataset based on Spotify playlists containing over 100'000 songs.
Data gets formatted for the use with Keras and few-shot learning.
"""

import keras
from keras import utils
import numpy as np
import random
#from skimage.io import imread

def imread(path):
    return np.zeros((250, 40, 1))

spectrogram_path = "/home/tobia/Documents/ML/Data MA Sono"
spectrogram_type = ".png"
playlists_path = "data/Playlists.csv"
songfeatures_path = "data/songfeatures_sigmoid_small.csv"


"""
Imports a list of the songs in each playlist.
"""
def load_playlists():
    with open(playlists_path, "r") as f:
        return [np.array(x.strip().split(','))[2:] for x in f.readlines()]


"""
Loads the spectrogram for a specific Spotify URI.
"""
def load_spectrogram(uri, input_shape):
    spectrogram = imread(spectrogram_path + "/" + str(uri) + spectrogram_type) / 256

    max_height = input_shape[0]

    spectrogram = spectrogram[-max_height:]

    return spectrogram


def load_random_slice_of_spectrogram(uri, input_shape):
    return load_songfeatures(uri, input_shape)    
    
    spectrogram = load_spectrogram(uri, input_shape)

    width = spectrogram.shape[1]
    start = int(random.random() * (width - input_shape[1]))

    return spectrogram[:, start : start + input_shape[1]]


songfeatures = {}

"""
Loads the songfeatures for a specific Spotify URI.
"""
def load_songfeatures(uri, input_shape):
    global songfeatures

    if len(songfeatures) == 0:
        with open(songfeatures_path, "r") as f:
            for x in f.readlines():
                a = x.strip().split(',')                
                songfeatures[a[1]] = np.array(a[2:], dtype=np.float).reshape(input_shape)                                                

        print(len(songfeatures), "songfeatures loaded.")

    return songfeatures[uri]