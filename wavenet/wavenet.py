from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import sys, warnings, math
import operator

import numpy as np
import keras.backend as K

from keras import metrics
from keras.callbacks import ModelCheckpoint
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.utils import to_categorical

from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Dropout, Flatten
from keras.layers import add, multiply, concatenate
from keras.layers.pooling import AveragePooling1D
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.core import Activation

sys.path.append("..")
from dl_utilities.datasets import dataset_utils as ds_utils
from dl_utilities.general import general as gen_utils


DEFAULT_WEIGHT_DECAY=1E-4
DEFAULT_EPSILON=1E-5

INIT_FILTER_SIZE=3
RECUR_FILTER_SIZE=2
DILATION_EXP_BASE=2


'''
    Implementation of WaveNet architecture in Keras.
        
    Formatted as a function returning a WaveNet module capable of 
        accepting an input embedding and an optional context embedding. 
        
    Function accepts these parameters to customize the WaveNet module:
        -output_dim:
            -final dimension of return tensor
        -residual_filters:
            -number of filters for residual (dilated) layers
        -dilation_steps (default=6):
            -number of dilation steps before restarting
            -therefore, maximum dilation is 2**dilation_steps
        -dilation_blocks (default=3):
            -number of restarts of dilation rates
            -therefore, if dilation_blocks is 3, it dilation rates will restart twice
        -final_activation (default='softmax'):
            -activation on top of final output layer before returning it
        -multi_context (default=False):
            -feature logits from multiple kernel sizes in residual layers
                -provides multiple receptive fields 
            -large receptive fields will output fewer channels
                -since longer range dependencies need less detail
        -regressor (default=False):
            -if true, model will perform casusal convolutions
                -useful for scenerios like music and pixel generation
                    where future output can only be conditioned on 
                    previous time steps
                -in this case, training sequences can be trimmed down to 
                    be the size of the receptive field (to allow for 
                    an increased batch size)
            -if false, model will perform 'same' padding convolutions
                -useful for scenerios like review sentiment analysis 
                    or language transation where it makes sense and 
                    may even be necessary to consider entire input 
                    before outputting anything
        
    References:
        https://arxiv.org/abs/1609.03499
'''


# A single set (e.g. residual layer) of Wavenet operations
def WaveNetBlock(new_filters, filter_size, dilation_rate, 
                    padding, multi_context=False, 
                    use_pooling=False):
                    
    def func(tensors):
        # Break down inputs 
        x = tensors[0]
        if len(tensors) == 1:
            context = None
        else:
            context = tensors[1]
                    
        
        # Set number of iterations of residual connections
        if multi_context:
            num_iter = 3
        else:
            num_iter = 1

            
        # Get needed residual layers
        input = Activation('relu')(x)
        
        combined_layers = []    
        for i in range(num_iter):
            tmp_filters = int(new_filters * (1 - (i * 0.25)))
            
            # Tanh component
            tanh_out = Conv1D(tmp_filters, (filter_size + i),
                                           padding=padding,
                                           dilation_rate=dilation_rate)(input)
                               
            if context and not use_pooling:
                tanh_context_out = Conv1D(tmp_filters, (filter_size + i),
                                               padding=padding,
                                               dilation_rate=dilation_rate)(context)                
                tanh_out = add([tanh_out, tanh_context_out])
            
            tanh_out = Activation('tanh')(tanh_out)            
            
            
            # Sigmoid component
            sigmoid_out = Conv1D(tmp_filters, (filter_size + i),
                                            padding=padding,
                                            dilation_rate=dilation_rate)(input)

            if context and not use_pooling:
                sigmoid_context_out = Conv1D(tmp_filters, (filter_size + i),
                                               padding=padding,
                                               dilation_rate=dilation_rate)(context)                
                sigmoid_out = add([sigmoid_out, sigmoid_context_out])
             
            sigmoid_out = Activation('sigmoid')(sigmoid_out) 
             
             
            # Get layers multiplied together      
            combined = multiply([tanh_out, sigmoid_out])
            combined_layers.append(combined)
        
        
        # If multi-context, concatenate all the filters
        if multi_context:
            final_combined = concatenate(combined_layers)
        else:
            final_combined = combined_layers[0]
            
            
        # Transform them to have "input_dim" filters so that they can be added as a residual    
        input_dim = K.int_shape(x)[-1]

        skip_layer = Conv1D(input_dim, 1, padding='same')(final_combined)
        skip_layer = BatchNormalization(axis=-1, epsilon=DEFAULT_EPSILON)(skip_layer)
        
        out_layer = add([x, skip_layer])
        
        
        # Return both skip and output layer
        return out_layer, skip_layer
    

    # Return function so that it mimics a normal Keras layer
    return func


# Function operating on an input to yield an output after passing through WaveNet model   
def WaveNet(output_dim, residual_filters=64, 
                dilation_steps=6, dilation_blocks=3, 
                final_activation='softmax',
                multi_context=False, regressor=True,
                use_pooling=False):
    
    # Module accepts an input embedding with an optional context vector
    def func(input_tensors):
    
        # Ensure that input is a Keras tensor
        if type(input_tensors) is not list:
            input_tensors = [ input_tensors ]
        elif len(input_tensors) > 2:
            raise ValueError('WaveNet model can accept at most two inputs - '
                                'an input embedding and a context embedding.')
            
        for tensor in input_tensors:
            if not K.is_keras_tensor(tensor):
                raise ValueError('WaveNet model requires Keras tensors as an inputs.')
        
        # Input should ideally have final dimension of 1
        input = input_tensors[0]
        input_dim = K.int_shape(input)[-1]  
        
        if input_dim > 1:
            warnings.warn('The input layer to a WaveNet model should typically have a '
                                  'dimensionality of 1.\nThis recurrent model is designed to discern'
                                  'long-term dependencies on\nscalar, continuous, high-resolution inputs.\n'
                                  'While it accepts inputs with a higher dimensionality, it is likely that\n'
                                  'this model is sub-optimal for a problem requiring such inputs.\n'
                                  'It should be noted that 1-hot vectors should not be passed as '
                                  'inputs to this model; if multi-dimension inputs are used, they should '
                                  'be in embedding form.\n')
                    
                    
        # Set parameters rated to filter size and dilation
        init_filter_size=INIT_FILTER_SIZE
        recurrent_filter_size=RECUR_FILTER_SIZE
        init_dilation_rate=DILATION_EXP_BASE
        
        if regressor:
            padding='causal'
            
            if use_pooling:
                raise ValueError('The "use_pooling" parameter is not usable '
                                    'in the regressor version of Wavenet.')
        else:
            padding='same'  
            
        def wavenet_dilation(cur_iter):
            exponent_val = (cur_iter % dilation_steps)
            return int(math.pow(init_dilation_rate, exponent_val))
                    
                    
        # Begin processing input by passing it through initial 1D conv to get desired number of filters
        out = Conv1D(residual_filters, init_filter_size, padding=padding)(input)
        out = BatchNormalization(axis=-1, epsilon=DEFAULT_EPSILON)(out)
        

        # Feed rsulting tensor through designated number of WaveNet dilation blocks
        skip_connections = [ ]
        
        cur_filters = residual_filters
        downsize_iters = int(dilation_steps * math.ceil(dilation_blocks / 4))
        
        for i in range(dilation_steps * dilation_blocks):
            # Perform WaveNet block
            input_tensors[0] = out
            out, skip = WaveNetBlock(cur_filters, recurrent_filter_size, 
                                        wavenet_dilation(i + 1),
                                        padding, multi_context)(input_tensors)
                                                    
                                                    
            # Reduce resolution on occasion by average pooling and increasing ilters
            if (use_pooling and i % downsize_iters == 0 and 
                (i + 1) != (dilation_steps * dilation_blocks)):
                
                cur_filters *= 2

                out = Conv1D(cur_filters, 1)(out)
                out = BatchNormalization(axis=-1, epsilon=DEFAULT_EPSILON)(out)
                out = AveragePooling1D(pool_size=2)(out)
                    
                skip = Conv1D(cur_filters, 1)(skip)
                skip = BatchNormalization(axis=-1, epsilon=DEFAULT_EPSILON)(skip)
                skip = AveragePooling1D(pool_size=2)(skip)
                
                new_skips = [ ]
                
                for s_in in skip_connections:
                    s_out = Activation('relu')(s_in)
                    s_out = Conv1D(cur_filters, 1)(s_out)
                    s_out = BatchNormalization(axis=-1, epsilon=DEFAULT_EPSILON)(s_out)
                
                    new_skips.append(AveragePooling1D(pool_size=2)(s_out))
                
                skip_connections = new_skips
               
               
            # Add skip connection
            skip_connections.append(skip)
            
            
        # Sum all skip connections and get final output        
        out = add(skip_connections)
        out = Activation('relu')(out)
        out = Conv1D(residual_filters, 1)(out)
        out = BatchNormalization(axis=-1, epsilon=DEFAULT_EPSILON)(out)
        out = Activation('relu')(out)    
        out = Conv1D(residual_filters, 1)(out)
        out = BatchNormalization(axis=-1, epsilon=DEFAULT_EPSILON)(out)
        out = Activation('relu')(out)
        
        if regressor:
            final_layer = Conv1D(output_dim, 1, activation=final_activation)(out)
        else:
           out = Flatten()(out)
           final_layer = Dense(output_dim, activation=final_activation)(out)
           
        
        # Return the last layer with softmax values
        return final_layer

        
    # Return function taking inputs to pass through WaveNet
    return func
    

    
#############   MAIN ROUTINE   #############	
if __name__ == '__main__':        
    # Model parameters
    USE_CHAR_VERSION=True
    
    if USE_CHAR_VERSION:
        time_steps = 2500
        top_words = None
        total_options = ds_utils.get_total_possible_chars()
        
        input_dim = 16
        hidden_dim = 24
        output_dim = 2

        dilation_steps = 5
        dilation_blocks = 4
    
    else:
        time_steps = 500
        top_words = 5000
        total_options = top_words
        
        input_dim = 16
        hidden_dim = 32
        output_dim = 2
        
        dilation_steps = 3
        dilation_blocks = 6
        
        
    # Print receptive field based on parameters above
    filters = [ INIT_FILTER_SIZE ]
    recurrent_filters = [ RECUR_FILTER_SIZE ] * (dilation_steps * dilation_blocks)
    filters.extend(recurrent_filters)
    
    dilation_rates = [ 0 ]
    for i in range(dilation_steps * dilation_blocks):
        dilation_rates.append(math.pow(DILATION_EXP_BASE, (i % dilation_steps)))
        
    final_layer_rf = gen_utils.get_effective_receptive_field(filters, dilation_rates)
    print("The receptive field of the final layer of current model is:  %d\n" % final_layer_rf)
    
    
    # Get IMDB training/test dataset
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

    
    # Convert to from word-based to char-based dataset
    if USE_CHAR_VERSION:
        sorted_keys = sorted(imdb.get_word_index().items(), key=operator.itemgetter(1))

        X_train = ds_utils.convert_word_dataset_to_char_dataset(X_train, sorted_keys)
        X_test = ds_utils.convert_word_dataset_to_char_dataset(X_test, sorted_keys)    


    # Truncate and pre-pad (e.g to the left) input sequences (depending on length)
    max_review_length = time_steps
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

    print(y_train.shape)
    y_train = to_categorical(y_train, output_dim)
    y_test = to_categorical(y_test, output_dim)
    
    print(y_train.shape)
    quit()
    
    # Set-up input embedding for all RNN models
    input = Input((time_steps, ))
    true_output = Input((output_dim, ))

    
    embedding_vecor_length = input_dim
    embedding = Embedding(total_options, embedding_vecor_length, 
                            input_length=max_review_length)(input)
    embedding = BatchNormalization(axis=-1, gamma_regularizer=l2(DEFAULT_WEIGHT_DECAY), 
                                    beta_regularizer=l2(DEFAULT_WEIGHT_DECAY))(embedding)
    embedding = Activation('relu')(embedding)

    
    # Get final layer of logits
    softmax_logits = WaveNet(output_dim, hidden_dim, 
                                dilation_steps=dilation_steps, 
                                dilation_blocks=dilation_blocks, 
                                multi_context=True, regressor=False, 
                                use_pooling=True)(embedding)

                                
    # Define model
    model = Model(inputs=input, outputs=softmax_logits)
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                        metrics=['accuracy'])
    model.summary()
    
    
    # Begin training
    num_epochs=8
    batch_size=8
    
    percent_val = 1.0
    num_examples = X_train.shape[0]
    train_subsection_i = int(batch_size * ((num_examples * percent_val) // batch_size))

    
    model.fit(X_train[:train_subsection_i], y_train[:train_subsection_i], 
                    epochs=num_epochs, batch_size=batch_size)
    
    
    # Evaluate at the end of training
    print("\n\nEvaluation...\n")
    scores = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
    print("\nAccuracy: %.2f%%" % (scores[1] * 100))
    
    
    # Exit successfully
    exit(0)