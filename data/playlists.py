""" 
Loads the 2821-playlists data set based on Spotify playlists containing over 100'000 songs.
Data gets formatted for the use with Keras and few-shot learning.
"""

import keras
from keras import utils
import numpy as np
import random

spectrogram_path = "/home/tobia/Documents/ML/Data MA Sono"
spectrogram_type = ".png"
playlists_path = "data/Playlists.csv"
songfeatures_path = "data/GenreOld.csv"

"""
Imports a list of the songs in each playlist.
"""
def load_playlists():
    with open(playlists_path, "r") as f:
        return [np.array(x.strip().split(','))[2:] for x in f.readlines()]

def load_songfeatures_grouped(input_lenght, train_ratio):
    playlists = load_playlists()

    features = {}

    with open(songfeatures_path, "r") as f:
        for x in f.readlines():
            a = x.strip().split(',')                
            features[a[1]] = np.array(a[2:], dtype=np.float).reshape((1, input_lenght, 1)) 
            features[a[1]] = np.minimum(1, np.power(np.maximum(0, features[a[1]]), 1/2))

        print(len(features)) 

    songfeatures_grouped = [[] for p in range(len(playlists))]

    for i, playlist in enumerate(playlists):
        for song in playlist:
            try:            
                songfeatures_grouped[i].append(np.array(features[song]).reshape((input_lenght,1,1)))
            except:
                pass
    
    random.shuffle(songfeatures_grouped)

    data_train = songfeatures_grouped[:int(train_ratio * len(songfeatures_grouped))]
    data_test = songfeatures_grouped[int(train_ratio * len(songfeatures_grouped)):]

    return data_train, data_test