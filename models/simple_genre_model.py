from keras import backend as k
from keras.models import Input, Model
from keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Lambda, Concatenate, BatchNormalization

"""
Builds a convnet for msuic similarity learning with spectrograms.
"""
def std_layer(x):
    return k.var(x, axis=2, keepdims=True)
def min_layer(x):
    return k.min(x, axis=2, keepdims=True)

def build_model(input_shape, embedding_lenght, decoder_lenght):
    height = input_shape[0]
    width = input_shape[1]

    inputLayer = Input(input_shape)
    convLayer1 = Conv2D(60, (height, 1), activation='relu', name='conv1')(inputLayer)
    convLayer1 = BatchNormalization()(convLayer1)
    convLayer2 = Conv2D(60, (1, 4), activation='relu', name='conv2')(convLayer1)
    convLayer2 = BatchNormalization()(convLayer2)
    convLayer3 = Conv2D(60, (1, 4), activation='relu', name='conv3')(convLayer2)
    convLayer3 = BatchNormalization()(convLayer3)

    avg_layer1 = AveragePooling2D((1, width))(convLayer1)    
    avg_layer2 = AveragePooling2D((1, width - 3))(convLayer2)
    
    avgLayer = AveragePooling2D((1, width - 6))(convLayer3)
    maxLayer = MaxPooling2D((1, width - 6))(convLayer3)
    stdLayer = Lambda(std_layer)(convLayer3)

    concatenatedAvg = Concatenate()([avg_layer1, avg_layer2, avgLayer])
    concatenatedAvg = Flatten()(concatenatedAvg)

    concatenated = Concatenate()([avgLayer, stdLayer, maxLayer])
    flatten = Flatten()(concatenated)
    dense = Dense(120, activation='relu', name='dense1')(flatten)
    dense = BatchNormalization()(dense)
    encoder_ouput = Dense(embedding_lenght, activation='sigmoid', name='dense2')(dense)

    decoder_dense = Dense(120, activation='relu')(encoder_ouput)
    decoder_output = Dense(decoder_lenght, activation='relu')(decoder_dense)

    output = Concatenate()([encoder_ouput, decoder_output, concatenatedAvg])

    model = Model(inputs=inputLayer, outputs=output)    

    return model
