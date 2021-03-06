"""
Builds a feed-forward for music similarity with a quadruplet cross-playlist encoder.
Output format of the Keras model: Embedding ; Decoder Output ; Target Decoder Output.
"""

import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Concatenate, BatchNormalization, Lambda
from keras.regularizers import l2

def build_model(input_shape, embedding_length):
    input_layer = Input(shape=input_shape)
    flatten_input = Flatten()(input_layer)
    dense = Dense(100, activation='relu')(flatten_input)    
   
    encoder_output_layer = Dense(embedding_length, activation='sigmoid')(dense)
    
    decoder_dense = Dense(100, activation='relu')(encoder_output_layer)         
    decoder_output_layer = Dense(np.prod(input_shape), activation='sigmoid')(decoder_dense)    

    output_layer = Concatenate()([encoder_output_layer, decoder_output_layer, flatten_input])

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
