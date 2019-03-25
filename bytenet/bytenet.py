from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import sys, gc, os, warnings, math
import operator

import numpy as np
import tensorflow as tf
import keras.backend as K

from os.path import isfile

from keras import metrics
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.utils import to_categorical

from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Dropout, Flatten, Lambda
from keras.layers import add, multiply, concatenate
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.core import Activation

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../")

from dl_utilities.datasets import dataset_utils as ds_utils  # Requires 'sys.path.append' call above
from dl_utilities.general import general as gen_utils  # Requires 'sys.path.append' call above
from dl_utilities.bytenet import bytenet_utils as bn_utils  # Requires 'sys.path.append' call above


DEFAULT_WEIGHT_DECAY=1E-4
DEFAULT_EPSILON=1E-5
        
INIT_FILTER_SIZE=2
DILATION_EXP_BASE=2    
      
  
  
'''
    Implementation of ByteMet architecture in Keras.
        
    Formatted as a function returning a ByteNet module capable of 
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
            -if false, model will perform 'same' padding convolutions
                -useful for scenerios like review sentiment analysis 
                    or language transation where it makes sense and 
                    may even be necessary to consider entire input 
                    before outputting anything
        
    References:
        https://arxiv.org/abs/1609.03499
'''
            
        

###############     Model functions     ##################


# A single set (e.g. residual layer) of Wavenet operations
def ByteNetBlock(new_filters, filter_size, dilation_rate, 
                    padding, multi_context=False):
    def func(input):
        # Set necessary WaveNet block parameters
        if multi_context:
            num_iter = 3
        else:
            num_iter = 1

            
        # Get needed residual layers
        scale_down_filters = int(new_filters * 0.5)
        
        out = Activation('relu')(input)
        out = Conv1D(scale_down_filters, 1)(out)
        out = BatchNormalization(axis=-1, epsilon=DEFAULT_EPSILON)(out)
        out = Activation('relu')(out)
        
        combined_layers = []    
        for i in range(num_iter):
            tmp_filters = int(scale_down_filters * (1 - (i * 0.25)))
        
            tmp_out = Conv1D(tmp_filters, (filter_size + (i * 2)),
                                            padding=padding,
                                            dilation_rate=dilation_rate)(out)
            tmp_out = BatchNormalization(axis=-1, epsilon=DEFAULT_EPSILON)(tmp_out)
            
            combined_layers.append(tmp_out)
            
            
        # If multi-context, concatenate all the filters
        if multi_context:
            final_combined = concatenate(combined_layers)
        else:
            final_combined = combined_layers[0]
            
            
        # Transform final layers to scale back up to "new_filters"
        skip_layer = Activation('relu')(final_combined)
        skip_layer = Conv1D(new_filters, 1)(skip_layer)
        skip_layer = BatchNormalization(axis=-1, epsilon=DEFAULT_EPSILON)(skip_layer)
        
        final_layer = add([input, skip_layer])
        
        
        # Return both skip and output layer
        return final_layer
    

    # Return function so that it mimics a normal Keras layer
    return func
    
    
# Function operating on an input to yield an output after passing through ByteNet model   
def ByteNet_helper(output_dim, residual_filters=64, 
                dilation_steps=5, dilation_blocks=3, 
                final_activation='softmax',
                multi_context=False, encoder=False):
    
    # Module accepts an input embedding with an optional context vector
    def func(input):
    
        # Ensure that the inputs have been passed through an embedding layer
        input_dim = K.int_shape(input)[-1]
        if input_dim < 2:
            raise ValueError('The input layer to a ByteNet model must have a dimensionality of '
                                  'more than 1.\nThis recurrent model is designed to discern '
                                  'long-term dependencies on embeddings of high-resolution inputs.')
        
        
        # Set parameters rated to filter size and dilation
        filter_size=INIT_FILTER_SIZE
        init_dilation_rate=DILATION_EXP_BASE

        if encoder:
            padding='same'            
        else:
            padding='causal'          
        
        def bytenet_dilation(cur_iter):
            exponent_val = (cur_iter % dilation_steps)
            return int(math.pow(init_dilation_rate, exponent_val))
                    
                    
        # Begin processing input and feed through ByteNet dilation blocks
        out = Conv1D(residual_filters, 1)(input)
        out = BatchNormalization(axis=-1, epsilon=DEFAULT_EPSILON)(out)
        
        for i in range(dilation_steps * dilation_blocks):
            out = ByteNetBlock(residual_filters, filter_size, 
                                        bytenet_dilation(i),
                                        padding, multi_context)(out)
            
            
        # Transform output layer to get correct number of output filters (per time step)
        out = Activation('relu')(out)
        out = Conv1D(output_dim, 1)(out)
        final_layer = BatchNormalization(axis=-1, epsilon=DEFAULT_EPSILON)(out)
        
        if not encoder:
            final_layer = Activation('relu')(final_layer)    
            final_layer = Conv1D(output_dim, 1, activation=final_activation)(final_layer)    
        
        
        # Return the last layer with softmax values
        return final_layer

        
    # Return function taking inputs to pass through ByteNet
    return func
    

# Encoder component of ByteNet    
def ByteNetEncoder(output_dim, residual_filters=64, 
                dilation_steps=5, dilation_blocks=3, 
                multi_context=False):

    return ByteNet_helper(output_dim, residual_filters,
                            dilation_steps, dilation_blocks,
                            None, multi_context, encoder=True)
    
    
# Decoder component of ByteNet
def ByteNetDecoder(output_dim, residual_filters=64, 
                dilation_steps=5, dilation_blocks=3, 
                final_activation='softmax',
                multi_context=False):                

    return ByteNet_helper(output_dim, residual_filters,
                            dilation_steps, dilation_blocks,
                            final_activation, multi_context, 
                            encoder=False)                
 

# Overall ByteNet function 
def ByteNet(time_steps, 
                input_dim,
                hidden_dim,
                dilation_steps,
                dilation_blocks,
                total_num_char,
                use_multi_context=True,
                encoder_output_dim=None,
                decoder_only=True):
        
    # Build decoder only model        
    if decoder_only:
        # Define input and get input embedding
        shifted_targets = Input((time_steps, ))
        
        
        in_embedding = Embedding(total_num_char, input_dim, input_length=time_steps)(shifted_targets)
        in_embedding = BatchNormalization(axis=-1, gamma_regularizer=l2(DEFAULT_WEIGHT_DECAY), 
                                    beta_regularizer=l2(DEFAULT_WEIGHT_DECAY))(in_embedding)
        in_embedding = Activation('relu')(in_embedding)

        
        # Pass through ByteNet decoder
        out_logits = ByteNetDecoder(total_num_char, hidden_dim, 
                                            dilation_steps=dilation_steps, 
                                            dilation_blocks=dilation_blocks, 
                                            multi_context=use_multi_context)(in_embedding)

                                            
        # Define overall model
        model = Model(inputs=shifted_targets, outputs=out_logits)
    

    # Build encoder-decoder version of ByteNet model
    else:
        # Set-up input embedding for all RNN models
        input = Input((time_steps, ))
        shifted_targets = Input((time_steps, ))

        
        in_embedding = Embedding(total_num_char, input_dim, input_length=time_steps)(input)
        in_embedding = BatchNormalization(axis=-1, gamma_regularizer=l2(DEFAULT_WEIGHT_DECAY), 
                                        beta_regularizer=l2(DEFAULT_WEIGHT_DECAY))(in_embedding)
        in_embedding = Activation('relu')(in_embedding)

        
        # Pass through encoder function to get encoder embedding
        if type(encoder_output_dim) is not int:
            raise ValueError("The encoder-decoder ByteNet model requires a value for 'encoder_output_dim'.")
            
        encode_embedding = ByteNetEncoder(encoder_output_dim, hidden_dim, 
                                            dilation_steps=dilation_steps, 
                                            dilation_blocks=dilation_blocks, 
                                            multi_context=use_multi_context)(in_embedding)
        encode_embedding = Lambda(lambda x: x, name="encode_out")(encode_embedding)
                            
                            
        # Pass through decoder to get encoder embedding
        target_embedding = Embedding(total_num_char, input_dim, 
                                        input_length=time_steps)(shifted_targets)
        target_embedding = BatchNormalization(axis=-1, gamma_regularizer=l2(DEFAULT_WEIGHT_DECAY), 
                                                beta_regularizer=l2(DEFAULT_WEIGHT_DECAY), 
                                                name="target_embed")(target_embedding)
        
        
        decoder_input = concatenate([encode_embedding, target_embedding])
        
        
        # Create decoder model
        decoder_in_shape = K.int_shape(decoder_input)[1:]
        tmp_input = Input(decoder_in_shape)
        
        decode_logits = ByteNetDecoder(total_num_char, hidden_dim, 
                                            dilation_steps=dilation_steps, 
                                            dilation_blocks=dilation_blocks, 
                                            multi_context=use_multi_context)(tmp_input)

        decoder_model = Model(inputs=tmp_input, outputs=decode_logits, name="decoder_model")
        
        
        # Pass decoder input through decoder model
        out_logits = decoder_model(decoder_input)
        
        
        # Define overall encoder/decoder model
        model = Model(inputs=[input, shifted_targets], outputs=out_logits)        


    
    # Return final model
    return model
    
    
            
    
    
#############   MAIN ROUTINE   #############	

if __name__ == '__main__':        
    # Model parameters
    DO_TRANSLATION=False
    DO_TRAINING=False
    SAVE_WEIGHTS=True
    PRELOAD_WEIGHTS=True

    

    ##############  Language translation                
    if DO_TRANSLATION:        
        # Define variables
        time_steps = 250
        min_time_steps = int(time_steps * 0.25)

        input_dim = 32
        hidden_dim = 128
        encoder_output_dim = 64
        
        dilation_steps = 6
        dilation_blocks = 4
         
         
        # Print receptive field based on parameters above
        filters = [ 1 ]
        recurrent_filters = [ INIT_FILTER_SIZE ] * (dilation_steps * dilation_blocks)
        filters.extend(recurrent_filters)
        
        dilation_rates = [ 0 ]
        for i in range(dilation_steps * dilation_blocks):
            dilation_rates.append(math.pow(DILATION_EXP_BASE, (i % dilation_steps)))
            
        final_layer_rf = gen_utils.get_effective_receptive_field(filters, dilation_rates)
        print("The receptive field of the final layer of current model is:  %d\n" % final_layer_rf)
         
         
        # Get translation dataset         
        desired_lang = 'EN-FR'
        src, target, decoder = ds_utils.get_translated_corpus_chars(langs=[desired_lang],
                                                                    min_len=min_time_steps, 
                                                                    max_len=time_steps)
        
        # Split dataset into train and test subsets
        num_ex = len(src)
        perm = np.random.permutation(num_ex)
        
        src_train, src_test = gen_utils.split_list_data(src, rank=2, 
                                        percent_split=0.8, desired_permutation=perm)
        
        tgt_train, tgt_test = gen_utils.split_list_data(target, rank=2, 
                                        percent_split=0.8, desired_permutation=perm)
    

        # Get hot-one encoding of the target char's for evaluation
        total_num_char = len(decoder)
        amp_factor = int(time_steps // 50)
        
        one_hot_tgt_train = bn_utils.one_hot_conversion(tgt_train, total_num_char, amp_factor)
        one_hot_tgt_test = bn_utils.one_hot_conversion(tgt_test, total_num_char, amp_factor)
        

        # Shift train/test target over 1
        token_train_shifted = np.zeros_like(tgt_train)
        token_test_shifted = np.zeros_like(tgt_test)
        
        for i in range(tgt_train.shape[0]):
            for j in reversed(range(1, time_steps)):
                token_train_shifted[i, j] = tgt_train[i, j - 1]
            token_train_shifted[i, 0] = 0
        
        for i in range(tgt_test.shape[0]):
            for j in reversed(range(1, time_steps)):
                token_test_shifted[i, j] = tgt_test[i, j - 1]
            token_test_shifted[i, 0] = 0
        
        
        # Define overall encoder/decoder model
        model = ByteNet(time_steps=time_steps, 
                            input_dim=input_dim,
                            hidden_dim=hidden_dim,
                            dilation_steps=dilation_steps,
                            dilation_blocks=dilation_blocks,
                            total_num_char=total_num_char,
                            use_multi_context=True,
                            encoder_output_dim=encoder_output_dim,
                            decoder_only=False)
        
        
        # Preload weights if requested
        weight_filename = "translator_weights." + desired_lang + ".h5"
        
        if PRELOAD_WEIGHTS:
            if isfile(weight_filename):
                model.load_weights(weight_filename)
            else:
                warnings.warn("Could not find weight file in the "
                                "current directory.", UserWarning)
                                

        # Begin training if training flag is set
        context_len = int(min_time_steps * 0.5)
        if DO_TRAINING:
            num_epochs = 30     # Should be greater than 'save_period' and also divisible by it
            batch_size = 16
            save_period = 3
            
            percent_val = 1.0
            num_examples = src_train.shape[0]
            train_subsection_i = int(batch_size * ((num_examples * percent_val) // batch_size))
            
            
            # Set up callbacks for training (e.g. save weights and update generator weights)            
            callbacks = []
            if SAVE_WEIGHTS:
                callbacks.append(ModelCheckpoint(weight_filename, 
                                    monitor="acc", period=int(num_epochs // save_period),
                                    save_best_only=False, save_weights_only=True))
                    
                    
            # Provide actual target lengths as a hint to predict length
            act_tgt_lens = np.ndarray(shape=(len(tgt_train), ))
            for i, tgt in enumerate(tgt_train):
                act_tgt_lens[i] = bn_utils.get_encoded_str_len(tgt)
                    
                    
            # Prepare variables for training (in cirriculum fashion)
            cur_difficulty = 0
            nthreads = 6
            
            max_queue_size=int((train_subsection_i * 0.5) // batch_size)
            subsection_per_thread = int((1.5 / nthreads) * train_subsection_i)
            subsection_per_thread = int(batch_size * (subsection_per_thread // batch_size))

            act_tgt_lens = act_tgt_lens[:train_subsection_i]
            src_train = src_train[:train_subsection_i]
            token_train_shifted = token_train_shifted[:train_subsection_i]
            one_hot_tgt_train = one_hot_tgt_train[:train_subsection_i]
            tgt_train = tgt_train[:train_subsection_i]
            
            
            # Define helper functions for the generator           
            def default_bytenet_model():
                return ByteNet(time_steps=time_steps, 
                                    input_dim=input_dim,
                                    hidden_dim=hidden_dim,
                                    dilation_steps=dilation_steps,
                                    dilation_blocks=dilation_blocks,
                                    total_num_char=total_num_char,
                                    use_multi_context=True,
                                    encoder_output_dim=encoder_output_dim,
                                    decoder_only=False)
            
            
            def get_encoder_func(model):
                enc_model, _, _ = bn_utils.get_subcomponents_of_enc_decode_bytenet(model)
                return enc_model
            
            
            def one_hot_conversion_func(orig_tgt):
                return bn_utils.one_hot_conversion(orig_tgt, total_num_char, amp_factor)
            
            
            def tmp_predictions_func(model, difficulty_level):
                # Function getting model predictions
                def get_pred_wrapper(shifted_targets, encoder_output, model):
                    model_components = bn_utils.get_subcomponents_of_enc_decode_bytenet(model)
                    encode_model, tgt_model, decoder_model = model_components
                                            
                    return bn_utils.get_prediction_of_enc_decode_bytenet(shifted_targets, 
                                                                            encoder_output,
                                                                            tgt_model, decoder_model, 
                                                                            batch_size=int(batch_size // 2))                  
                
                
                # Return "generator_pred_wrapper" with appropriate parameters baked in
                return bn_utils.generator_pred_wrapper(model, difficulty_level,              
                                                        batch_size=int(batch_size // 2),
                                                        num_examples=subsection_per_thread,
                                                        encoder_inputs=src_train,
                                                        get_encoder_func=get_encoder_func,
                                                        shifted_inputs=token_train_shifted, 
                                                        get_shifted_preds_func=bn_utils.get_shifted_predictions,
                                                        final_str_len_hints=act_tgt_lens,    
                                                        actual_targets=tgt_train,
                                                        input_processing_func=get_pred_wrapper,                 
                                                        label_conversion_func=one_hot_conversion_func,
                                                        context_len=context_len, 
                                                        init_beam_size=1)
            
            
            # Start multi-thread generator after functions are defined, imported, and/or in-scope 
            final_num_examples = train_subsection_i
            seq_gen = bn_utils.multi_thread_seq_gen(queue_size=max_queue_size, nthreads=nthreads)
            seq_gen.start_activity(default_bytenet_model, 
                                        tmp_predictions_func, 
                                        subsection_per_thread, 
                                        final_num_examples, 
                                        batch_size, cur_difficulty)
             
            callbacks.append(seq_gen.weight_callback)

            
            # Compile model after creating generator (to avoid transferring TF context to other threads)            
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            
            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            sess = tf.Session(config=config)
            K.set_session(sess)
            
            
            model.compile(loss='categorical_crossentropy', optimizer='adam', 
                                metrics=['accuracy'])
            model.summary()            
            
            
            # Begin cirriculum training
            inc_loss_val = 1.0
            
            for i in range(0, num_epochs, save_period): 
                # Conduct training
                history = model.fit_generator(seq_gen, 
                                                steps_per_epoch=(train_subsection_i // batch_size), 
                                                epochs=(i + save_period), 
                                                initial_epoch=i,
                                                callbacks=callbacks)            
                
                
                # Increase difficulty if too easy
                last_loss = history.history['loss'][-1]

                print(last_loss)
                if cur_difficulty > 0:
                    quit()

                if cur_difficulty < MAX_DIFF_LEVEL and inc_loss_val > last_loss:
                    cur_difficulty += 1
                    seq_gen.set_new_difficulty_value(cur_difficulty)

                    
                    
        # Stop generator
        seq_gen.stop_activity()
        seq_gen.stop_all_threads()
        
                    
        # Perform qualitative evaluation
        print("\nEvaluation...\n")        
        bn_utils.do_translation_evaluation(model, src_test, 
                                    tgt_test, test_num_examples=5, 
                                    context_len=context_len, 
                                    beam_size=12, total_num_char=total_num_char,
                                    decoder=decoder)        

         

         
    ##############  Character sequence generation        
    else:
        # Define variables
        desired_examples = 20000
        
        min_time_steps = 75
        time_steps = 250        

        input_dim = 64
        hidden_dim = 196
        
        dilation_steps = 4
        dilation_blocks = 6
                
        
        # Print receptive field based on parameters above
        filters = [ 1 ]
        recurrent_filters = [ INIT_FILTER_SIZE ] * (dilation_steps * dilation_blocks)
        filters.extend(recurrent_filters)
        
        dilation_rates = [ 0 ]
        for i in range(dilation_steps * dilation_blocks):
            dilation_rates.append(math.pow(DILATION_EXP_BASE, (i % dilation_steps)))
            
        final_layer_rf = gen_utils.get_effective_receptive_field(filters, dilation_rates)
        print("The receptive field of the final layer of current model is:  %d\n" % final_layer_rf)
        
        
        # Get English text excerpts
        corpus_gen = ds_utils.gen_corpus_examples(time_steps, 
                                                    min_example_size=min_time_steps, 
                                                    use_paragraphs=False)

        language_toks = []
        actual_examples = 0
        
        print("Getting corpus tokens...\nOn example:   0")
        for decoder, tok in corpus_gen:
            language_toks.append(tok)

            actual_examples += 1
            if actual_examples == desired_examples:
                break
            elif actual_examples % 2500 == 0:
                print("\t\t%d" % actual_examples)
                
        if actual_examples < desired_examples:
            warnings.warn("Insufficient number of language tokens found.\n"
                            "Either use a different corpus or decrease number "
                            "of desired examples.", UserWarning)


        # Split dataset into train and test subsets
        token_train, token_test = gen_utils.split_list_data(language_toks, rank=2, 
                                                                percent_split=0.8)

        
        # Shift train/test input over 1 to get labels
        token_train_shifted = np.zeros_like(token_train)
        token_test_shifted = np.zeros_like(token_test)
        
        for i in range(token_train.shape[0]):
            for j in reversed(range(1, time_steps)):
                token_train_shifted[i, j] = token_train[i, j - 1]
            token_train_shifted[i, 0] = 0
        
        for i in range(token_test.shape[0]):
            for j in reversed(range(1, time_steps)):
                token_test_shifted[i, j] = token_test[i, j - 1]
            token_test_shifted[i, 0] = 0

            
        # Get hot-one encoding of the target char's for evaluation
        total_num_char = len(decoder)
        
        one_hot_tgt_train = gen_utils.transform_into_one_hot(token_train, total_num_char)
        one_hot_tgt_test = gen_utils.transform_into_one_hot(token_test, total_num_char)
        
        one_hot_tgt_train = gen_utils.remove_padding_from_one_hot(one_hot_tgt_train)
        one_hot_tgt_test = gen_utils.remove_padding_from_one_hot(one_hot_tgt_test)

                                            
        # Define model
        model = ByteNet(time_steps=time_steps, 
                            input_dim=input_dim,
                            hidden_dim=hidden_dim,
                            dilation_steps=dilation_steps,
                            dilation_blocks=dilation_blocks,
                            total_num_char=total_num_char,
                            use_multi_context=True,
                            decoder_only=True)
                            
        model.compile(loss='categorical_crossentropy', optimizer='adam', 
                            metrics=['accuracy'])
        model.summary()                                                 
        

        # Preload weights if requested
        weight_filename = "generator_weights.h5"
        
        if PRELOAD_WEIGHTS:
            if isfile(weight_filename):
                model.load_weights(weight_filename)
            else:
                warnings.warn("Could not find weight file in the "
                                "current directory.", UserWarning)

                                
        # Begin training if training flag is set
        if DO_TRAINING:
            num_epochs = 20     # Should be greater than 'save_period'
            batch_size = 16
            save_period = 3
            
            percent_val = 1.0
            num_examples = token_train_shifted.shape[0]
            train_subsection_i = int(batch_size * ((num_examples * percent_val) // batch_size))
             
            callbacks = []
            callbacks.append(ModelCheckpoint(weight_filename, 
                                monitor="acc", period=int(num_epochs // save_period),
                                save_best_only=False, save_weights_only=True))
                            
            model.fit(token_train_shifted[:train_subsection_i], 
                            one_hot_tgt_train[:train_subsection_i], 
                            epochs=num_epochs, batch_size=batch_size,
                            callbacks=callbacks)

                
        # Perform qualitative evaluation
        print("\nEvaluation...\n")
        bn_utils.do_char_gen_evaluation(model, token_test, test_num_examples=5, 
                                            context_len=int(0.5 * min_time_steps), 
                                            beam_size=12, total_num_char=total_num_char,
                                            decoder=decoder)
    
    
    # Exit successfully
    exit(0)
