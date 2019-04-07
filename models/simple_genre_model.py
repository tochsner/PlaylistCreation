"""
Builds a simple convnet for music similarity with a quadruplet cross-playlist encoder.
Output format of the Keras model: Embedding ; Decoder Output
"""

import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Concatenate, BatchNormalization

def build_model(input_shape, embedding_length):
    input_layer = Input(shape=input_shape)
    dense = Flatten()(input_layer)
    dense = Dense(250, activation='relu')(dense)    
    dense = BatchNormalization()(dense)

    encoder_output_layer = Dense(embedding_length, activation='sigmoid')(dense)
    
    decoder_dense = Dense(250, activation='relu')(encoder_output_layer)    
    decoder_dense = BatchNormalization()(decoder_dense)
    decoder_output_layer = Dense(np.prod(input_shape), activation='sigmoid')(decoder_dense)    

    output_layer = Concatenate()([encoder_output_layer, decoder_output_layer])

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
