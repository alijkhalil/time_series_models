from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings, math

import keras.backend as K

from keras import metrics
from keras.callbacks import ModelCheckpoint
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.utils import to_categorical

from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Flatten
from keras.layers import add, multiply, concatenate
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.core import Activation


DEFAULT_WEIGHT_DECAY=1E-4


def WaveNetBlock(new_filters, filter_size, dilation_rate, 
                    use_BN=False, multi_context=False, regressor=False):
    def func(x):
        input = Activation('relu')(x)
        input_dim = K.int_shape(x)[-1]
        
        
        # Set necessary WaveNet block parameters
        if regressor:
            padding='causal'
        else:
            padding='same'
            
        if multi_context:
            num_iter = 3
        else:
            num_iter = 1

            
        # Get needed residual layers
        combined_layers = []    
        for i in range(num_iter):
            tanh_out = Conv1D(new_filters, (filter_size + (i * 2)),
                                           padding=padding,
                                           dilation_rate=dilation_rate,
                                           activation='tanh')(input)
                                           
            sigmoid_out = Conv1D(new_filters, (filter_size + (i * 2)),
                                            padding=padding,
                                            dilation_rate=dilation_rate,
                                            activation='sigmoid')(input)
                                          
            combined = multiply([tanh_out, sigmoid_out])
            combined_layers.append(combined)
        
        if multi_context:
            final_combined = concatenate(combined_layers)
        else:
            final_combined = combined_layers[0]
            
            
        # Transform them to have "input_dim" filter so that they can be added as a residual    
        skip_layer = Conv1D(input_dim, 1, padding='same')(final_combined)
        skip_layer = BatchNormalization(axis=-1, epsilon=1E-5)(skip_layer)
        
        out_layer = add([input, skip_layer])
        
        
        # Return both skip and output layer
        return out_layer, skip_layer
        
    return func

    
def WaveNet(input_tensor, output_dim, skip_filters=64, 
                dilation_steps=6, dilation_blocks=3, final_activation='softmax',
                use_BN=False, multi_context=False, regressor=False):
                
    # Ensure that input into Keras tensor
    if not K.is_keras_tensor(input_tensor):
        raise ValueError('WaveNet model requires a Keras tensor as an input.')
    
    # Input should ideally have final dimension of 1
    input = input_tensor
    input_dim = K.int_shape(input)[-1]    
    
    if input_dim > 1:
        warnings.warn('The input layer to a WaveNet model should typically have a '
                              'dimensionality of 1.\nThis recurrent model is designed to '
                              'discern long-term dependencies on scalar, high-resolution inputs.\n' 
                              'While it accepts inputs with a higher dimensionality, it is likely\n'
                              'that this model is sub-optimal for this particular problem.')
                
    # Set parameters rated to filter size and dilation
    init_filter_size=2
    init_dilation_rate=2

    def wavenet_dilation(cur_iter):
        exponent_val=(1 + (cur_iter % dilation_steps))
        return int(math.pow(init_dilation_rate, exponent_val))
                
                
    # Begin processing input and feed through WaveNet dilation blocks
    out, skip = WaveNetBlock(skip_filters, init_filter_size, 2,
                                use_BN, multi_context, regressor)(input)
    
    skip_connections = [skip]
    for i in range(1, (dilation_steps * dilation_blocks)):
        out, skip = WaveNetBlock(skip_filters, init_filter_size, wavenet_dilation(i),
                                    use_BN, multi_context, regressor)(out)
                                    
        skip_connections.append(skip)
        
        
    # Sum all skip connections and get final output
    out = add(skip_connections)
    out = Activation('relu')(out)
    out = Conv1D(input_dim, 1)(out)
    out = BatchNormalization(axis=-1, epsilon=1E-5)(out)
    out = Activation('relu')(out)    
    out = Conv1D(input_dim, 1)(out)
    out = Flatten()(out)
    final_layer = Dense(output_dim, activation=final_activation)(out)
    
    
    # Return the last layer with softmax values
    return final_layer

    
    

#############   MAIN ROUTINE   #############	
if __name__ == '__main__':
    # Model parameters    
    time_steps=500

    input_dim=64
    hidden_dim=256
    output_dim=2


    # Get IMDB training/test dataset
    top_words = 5000
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)


    # Truncate and pad input sequences (depending on length)
    max_review_length = time_steps
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length) 

    y_train = to_categorical(y_train, output_dim)
    y_test = to_categorical(y_test, output_dim)


    # Set-up input embedding for all RNN models
    input = Input((time_steps, ))
    true_output = Input((output_dim, ))

    embedding_vecor_length = input_dim
    embedding = Embedding(top_words, embedding_vecor_length, 
                            input_length=max_review_length)(input)
    embedding = BatchNormalization(axis=-1, gamma_regularizer=l2(DEFAULT_WEIGHT_DECAY), 
                                    beta_regularizer=l2(DEFAULT_WEIGHT_DECAY))(embedding)
    
    softmax_logits = WaveNet(embedding, output_dim, hidden_dim, 
                                        dilation_steps=5, dilation_blocks=4, 
                                        use_BN=False, multi_context=False, regressor=False)
                            
    # Define model
    model = Model(inputs=input, outputs=softmax_logits)
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                        metrics=['accuracy'])
    model.summary()
    
    
    # Begin training
    num_epochs=8
    batch_size = 64
    percent_val = 1.0
    
    num_examples = X_train.shape[0]
    train_subsection_i = int(batch_size * ((num_examples * percent_val) // batch_size))
    
    model.fit(X_train[:train_subsection_i], y_train[:train_subsection_i], epochs=num_epochs, batch_size=batch_size)
    
    
    # Evaluate at the end of training
    print("\nEvaluation...\n")
    scores = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
    print("Accuracy: %.2f%%" % (scores[1] * 100))    
    
    
    # Exit successfully
    exit(0)  