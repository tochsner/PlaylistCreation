import numpy as np
import random

from helper.losses import MeanSquareCostFunction
from data.playlists import load_spectrogram

"""
Generates input and output pairs for performing similarity learning with Keras.
Based on quadruplet-selection if use_min_distance is True, otherwise on triplet-selection.
Output format of the Keras model: Embedding ; Decoder Output ; Target Decoder Output
Format of y_train: Similar Embedding ; Dissimilar Embedding ; Similar Decoder Output
"""
def create_training_data_for_quadruplet_loss(model, grouped_data, num_samples, input_shape, output_helper, use_min_distance = True):
    mse = MeanSquareCostFunction()
    
    num_classes = len(grouped_data)
   
    indexes = list(range(num_classes))
        
    x_shape = (num_samples,) + input_shape
    y_shape = (num_samples, 2 * output_helper.embedding_length + output_helper.decoder_output_length)    

    x_data = np.zeros(x_shape)
    y_data = np.zeros(y_shape)

    for sample in range(num_samples // 2):
        exception = True

        while exception:
            try:
                main_index = random.choice(indexes)
                second_index = random.choice([index for index in indexes if index != main_index])

                main_sample1 = load_spectrogram(random.choice(grouped_data[main_index]), input_shape)
                main_sample2 = load_spectrogram(random.choice(grouped_data[main_index]), input_shape)
                second_sample1 = load_spectrogram(random.choice(grouped_data[second_index]), input_shape)
                second_sample2 = load_spectrogram(random.choice(grouped_data[second_index]), input_shape)

                exception = False
            except:
                pass
        
        outputs = model.predict(np.array([main_sample1, main_sample2, second_sample1, second_sample2]))

        main_embedding_1 = output_helper.get_embedding(outputs[0])
        main_embedding_2 = output_helper.get_embedding(outputs[1])
        second_embedding_1 = output_helper.get_embedding(outputs[2])
        second_embedding_2 = output_helper.get_embedding(outputs[3])

        main_target_decoder_output_1 = output_helper.get_target_decoder_output(outputs[0])
        main_target_decoder_output_2 = output_helper.get_target_decoder_output(outputs[1])
        second_target_decoder_output_1 = output_helper.get_target_decoder_output(outputs[2])
        second_target_decoder_output_2 = output_helper.get_target_decoder_output(outputs[3])                

        costs = (mse.get_cost(main_embedding_1, second_embedding_1),
                 mse.get_cost(main_embedding_1, second_embedding_2),
                 mse.get_cost(main_embedding_2, second_embedding_1),
                 mse.get_cost(main_embedding_2, second_embedding_2))

        argmin = np.argmin(costs)

        if not use_min_distance:
            argmin = 0

        if argmin == 0:
            # mainSample 1
            x_data[2 * sample] = main_sample1    
            y_data[2 * sample] = output_helper.get_target_output(main_embedding_2, second_embedding_1, main_target_decoder_output_2)            

            # secondSample 1
            x_data[2 * sample + 1] = second_sample1           
            y_data[2 * sample + 1] = output_helper.get_target_output(second_embedding_2, main_embedding_1, second_target_decoder_output_2)          
        elif argmin == 1:
            # mainSample 1
            x_data[2 * sample] = main_sample1    
            y_data[2 * sample] = output_helper.get_target_output(main_embedding_2, second_embedding_2, main_target_decoder_output_2)

            # secondSample 2
            x_data[2 * sample + 1] = second_sample2            
            y_data[2 * sample + 1] = output_helper.get_target_output(second_embedding_1, main_embedding_1, second_target_decoder_output_1)
        elif argmin == 2:
            # mainSample 2
            x_data[2 * sample] = main_sample2   
            y_data[2 * sample] = output_helper.get_target_output(main_embedding_1, second_embedding_1, main_target_decoder_output_1)

            # secondSample 1
            x_data[2 * sample + 1] = second_sample1
            y_data[2 * sample + 1] = output_helper.get_target_output(second_embedding_2, main_embedding_2, second_target_decoder_output_2)
        elif argmin == 3:
            # mainSample 2
            x_data[2 * sample] = main_sample2   
            y_data[2 * sample] = output_helper.get_target_output(main_embedding_1, second_embedding_2, main_target_decoder_output_1)

            # secondSample 2
            x_data[2 * sample + 1] = second_sample2
            y_data[2 * sample + 1] = output_helper.get_target_output(second_embedding_1, main_embedding_2, second_target_decoder_output_1)

    return (x_data, y_data)
