# Import statements
import sys, argparse
sys.path.append("..")

import tensorflow as tf
import numpy as np

import keras.backend as K

from keras import metrics
from keras.callbacks import ModelCheckpoint
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.utils import to_categorical

from keras.models import Model, Sequential
from keras.layers import Input, BatchNormalization, Dense
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.layers.core import Activation

from rhn import rhn
from act import act
from custom_rnns import custom_rhn, stacked_gru

from dl_utilities.layers import custom_loss as loss_layers



# Model options 
ACT_KEYWORD='act'
RHN_KEYWORD='rhn'
CUSTOM_RHN_KEYWORD='custom_rhn'
GRU_KEYWORD='stacked_gru'

CUSTOM_LOSS_LAYERS = [ACT_KEYWORD]



#############   MAIN ROUTINE   #############	
if __name__ == '__main__':

    # Set up argument parser 
	#
	# Format: 
	#		tester.py <model_type> [--dropout [<float>]] [--recurrent_dropout [<float>]]    
	#							[--depth [<num_of_layers_per_time_step>]]
	#							[--input_dim [<dimension_of_input_embedding>]]
	#							[--hidden_dim [<dimension_of_hidden_state>]]
	#							[--epochs [<num_of_training_epochs>]] [--save_weights]
	#
    parser = argparse.ArgumentParser(description='Test module for a variety of RNN models on the IMBD dataset.')
    
    parser.add_argument('model_type', choices=[ACT_KEYWORD, RHN_KEYWORD, CUSTOM_RHN_KEYWORD, GRU_KEYWORD],
                            help='RNN model of choice (required)')
    parser.add_argument('--dropout', nargs='?', default=0.0, const=0.1, type=float, metavar='dropout',
                            help='apply dropout to embedding input of model at this rate (flag alone defaults to 0.1)')
    parser.add_argument('--recurrent_dropout', nargs='?', default=0.0, const=0.25, type=float, metavar='recurrent_dropout',
                            help='apply dropout recurrent hidden state at this rate (flag alone defaults to 0.25)')                            
    parser.add_argument('--depth', nargs='?', default=5, const=5, type=int, metavar='depth',
                            help='maximum depth of each time step in RNN (no flag defaults to 5)')
    parser.add_argument('--input_dim', nargs='?', default=128, const=128, type=int, metavar='input_dim',
                            help='dimensionality of the input embeddings to the RNN (no flag defaults to 128)')
    parser.add_argument('--hidden_dim', nargs='?', default=512, const=512, type=int, metavar='hidden_dim',
                            help='dimensionality of the hidden states in the RNN (no flag defaults to 512)')                            
    parser.add_argument('--epochs', nargs='?', default=8, const=8, type=int, metavar='num_training_epochs',
                            help='number of training epochs on the IMDB dataset (no flag defaults to 8)')                            
    parser.add_argument('--save_weights', action='store_true', help='flag to save weights in the following format: ' \
                            '<model_dir>/weights-imdb-<dropout>-<recurrent_dropout>-<depth>-<input_dim>-<hidden_dim>-<epochs>.hdf5')
    
    args = parser.parse_args()

    
    # Set up TF session
    num_cores = 6
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True,
                                intra_op_parallelism_threads=num_cores,
                                inter_op_parallelism_threads=num_cores)

    # https://github.com/tensorflow/tensorflow/blob/30b52579f6d66071ac7cdc7179e2c4aae3c9cb88/tensorflow/core/protobuf/config.proto#L35
    # If true, the allocator does not pre-allocate the entire specified
    # GPU memory region, instead starting small and growing as needed.
    config.gpu_options.allow_growth=True

    sess = tf.Session(config=config)
    K.set_session(sess)


    # Model parameters    
    time_steps=500

    input_dim=args.input_dim
    hidden_dim=args.hidden_dim
    output_dim=2

    input_dropout = args.dropout
    rec_dropout = args.recurrent_dropout


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
    embedding = BatchNormalization(axis=-1, gamma_regularizer=l2(1E-4), 
                            beta_regularizer=l2(1E-4))(embedding)
    embedding = Activation('relu')(embedding)
    
    
    # Set model-specific variables and get output layer
    model_type = args.model_type
    model_depth = args.depth
    
    custom_loss_layer = None
        
    if model_type == ACT_KEYWORD:
        weight_dirname = "./act/"
        
       # Should eventually be CalculateACTLoss (requires override of "call" fn with states returned)
        custom_loss_layer = loss_layers.CalculateNormalXEntropyLoss

        out_layer = act.ACT_Cell(hidden_dim, output_units=output_dim, 
                                        max_computation_iters=model_depth,
                                        dropout=input_dropout, 
                                        recurrent_dropout=rec_dropout)(embedding)
                
                
    elif model_type == RHN_KEYWORD:
        weight_dirname = "./rhn/"
        
        out_layer = rhn.RHN(hidden_dim, depth=model_depth, dropout=input_dropout, 
                                recurrent_dropout=rec_dropout)(embedding)                        
        out_layer = Dense(output_dim)(out_layer)
        
        
    elif model_type == CUSTOM_RHN_KEYWORD:
        weight_dirname = "./custom_rnns/"

        out_layer = custom_rhn.CustomRHN(hidden_dim, depth=model_depth, 
                                            dropout=input_dropout, 
                                            recurrent_dropout=rec_dropout)(embedding)                        
        out_layer = Dense(output_dim)(out_layer)
        
        
    elif model_type == GRU_KEYWORD:
        weight_dirname = "./custom_rnns/"

        out_layer = stacked_gru.StackedGRU(hidden_dim, embedding, 
                                            depth=model_depth, 
                                            dropout=input_dropout, 
                                            recurrent_dropout=rec_dropout)
        out_layer = Dense(output_dim)(out_layer)
        
        
    # Get prediction, compile, and set-up inputs and outputs
    predictions = Activation('softmax', name="predictions")(out_layer)
    
    num_examples = X_train.shape[0]    
    if model_type in CUSTOM_LOSS_LAYERS:
        final_loss = custom_loss_layer(name="loss_val")([true_output, predictions])
    
        model = Model([input, true_output], final_loss)
        model.compile(loss=(lambda y_true, y_pred: y_pred), optimizer='adam')
        
        inputs = [ X_train, y_train ]
        output = np.random.random((num_examples, 1))
        
    else:
        model = Model(input, predictions)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])        

        inputs = [ X_train ]
        output = y_train
        
        
        
    # Print model summary
    model.summary()
    
    
    # Set up weight saving callbacks if requested                             
    save_weights = args.save_weights
    num_epochs = args.epochs
    
    callbacks = [ ]
    if save_weights:
        weight_path = weight_dirname + "weights-imdb-" + model_type + "-"
        weight_path += (str(input_dropout) + "-" + str(rec_dropout) + 
                            "-" + str(model_depth) + "-" + str(input_dim) +
                            "-" + str(hidden_dim) + str(num_epochs) + 
                            ".hdf5.alt")
        
        print(weight_path)
        callbacks.append(ModelCheckpoint(weight_path, 
                            monitor="loss", period=int(num_epochs),
                            save_best_only=False, save_weights_only=True))

    
    # Conduct training
    batch_size = 32
    percent_val = 1.0
    
    train_subsection_i = int(batch_size * ((num_examples * percent_val) // batch_size))
    
    final_inputs = [ i_arr[:train_subsection_i] for i_arr in inputs ]
    final_output = output[:train_subsection_i]

    model.fit(final_inputs, final_output, epochs=num_epochs, batch_size=batch_size)
    
    
    # Evaluate model and validation accuracy
    pred_layer = model.get_layer("predictions")                    
    new_model = Model(inputs=model.inputs[0], outputs=pred_layer.output)
    new_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("\n")
    scores = new_model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    
    
    # Exit successfully
    exit(0)  