# IMDB dataset
# requires tranformation in character-wise dataset
# can be using:
#    from keras.datasets import imdb
#    imdb.get_word_index(path='imdb_word_index.json'):

# Overall model would:
#   -parse dataset into words
#   -parse words into character embeddings
#   -insert following tokens:
#           -spaces tokens between words
#           -start sequence token 
#           -UNK sequence token for OVV words
#               -EITHER of a certain "normal" word len or do it per actual character 
#               -agruments:
#                  -would be providing extra info (e.g. word length)
#                   -word/char not directly comparable either way
#   -DOES DATA HAVE SPECIAL MARKER FOR PERIODS, COMMAS, ETC
#
#   When tested with old recurrent RNNs
#       -function to convert char embedding into N-grams to make word embeddings
#       -link:  https://arxiv.org/pdf/1508.06615.pdf
#
#   Model
#       -make char embedding or maybe just take the straight number
#           -how would it work with char embeddings? don't think it would!
#       -pass into wavenet model and process


# Look at: https://github.com/usernaamee/keras-wavenet/blob/master/simple-generative-model-regressor.py
#       -generative version very similar to regressive in implementation
#       -should be the same
#           -generative used for Text-To-Speech, music generation
#           -regressor used for prediction of next audio waves

# Consider differences with ByteNet
#       -operates on character
#       -translator uses encoder/decoder framework
#       -Generator uses only decoder
#
#       -Encoder can take context too 
#       -Casual mask done on decoder
#           -But also takes last token (via addition) too


# Replaced by "Conv1D" with "dilation_rate" parameter
# Should be kwargs['dilation_rate'] = rate (e.g. atrous_rate)
# atrous_rate is distance between nodes in the filter (should be exponent of 2)
class CausalAtrousConvolution1D(AtrousConvolution1D):
    def __init__(self, nb_filter, filter_length, init='glorot_uniform', activation=None, weights=None,
                 border_mode='valid', subsample_length=1, atrous_rate=1, W_regularizer=None, b_regularizer=None,
                 activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, causal=False, **kwargs):
        
        super(CausalAtrousConvolution1D, self).__init__(nb_filter, filter_length, init, activation, weights,
                                                        border_mode, subsample_length, atrous_rate, W_regularizer,
                                                        b_regularizer, activity_regularizer, W_constraint, b_constraint,
                                                        bias, **kwargs)
        self.causal = causal
        if self.causal and border_mode != 'valid':
            raise ValueError("Causal mode dictates border_mode=valid.")

    def get_output_shape_for(self, input_shape):
        input_length = input_shape[1]

        if self.causal:
            input_length += self.atrous_rate * (self.filter_length - 1)

        # def conv_output_length(input_length, filter_size,
        #                           padding, stride, dilation=1):
        length = conv_output_length(input_length,
                                    self.filter_length,
                                    self.border_mode,
                                    self.subsample[0],  # Should just be baked in as stride of 1
                                    dilation=self.atrous_rate)

        return (input_shape[0], length, self.nb_filter)

    def call(self, x, mask=None):
        # For prepending input with padding (on one side)
        if self.causal:
            x = K.asymmetric_temporal_padding(x, self.atrous_rate * (self.filter_length - 1), 0)
        
        # No mask parameter in call
        # May need to mask out prior... WITH WHAT FUNCTION TO GENERATE THE MASK?
        #                                Make my own layer to pass it through
        return super(CausalAtrousConvolution1D, self).call(x, mask)


def WaveNet(fragment_length, nb_filters, nb_output_bins, dilation_depth, nb_stacks, use_skip_connections,
                learn_all_outputs, _log, desired_sample_rate, use_bias, res_l2, final_l2):
    def residual_block(x):
        original_x = x
        
        # Note: The AtrousConvolution1D with the 'causal' flag is implemented in github.com/basveeling/keras#@wavenet.
        tanh_out = CausalAtrousConvolution1D(nb_filters, 2, atrous_rate=2 ** i, border_mode='valid', causal=True,
                                             bias=use_bias,
                                             name='dilated_conv_%d_tanh_s%d' % (2 ** i, s), activation='tanh',
                                             W_regularizer=l2(res_l2))(x)
        sigm_out = CausalAtrousConvolution1D(nb_filters, 2, atrous_rate=2 ** i, border_mode='valid', causal=True,
                                             bias=use_bias,
                                             name='dilated_conv_%d_sigm_s%d' % (2 ** i, s), activation='sigmoid',
                                             W_regularizer=l2(res_l2))(x)
        x = layers.Merge(mode='mul', name='gated_activation_%d_s%d' % (i, s))([tanh_out, sigm_out])

        res_x = layers.Convolution1D(nb_filters, 1, border_mode='same', bias=use_bias,
                                     W_regularizer=l2(res_l2))(x)
        skip_x = layers.Convolution1D(nb_filters, 1, border_mode='same', bias=use_bias,
                                      W_regularizer=l2(res_l2))(x)
                                      
        res_x = layers.Merge(mode='sum')([original_x, res_x])
        return res_x, skip_x
        
        
        
    input = Input(shape=(fragment_length, nb_output_bins), name='input_part')
    out = input
    skip_connections = []
    out = CausalAtrousConvolution1D(nb_filters, 2, atrous_rate=1, border_mode='valid', causal=True,
                                    name='initial_causal_conv')(out)
    for s in range(nb_stacks):
        for i in range(0, dilation_depth + 1):
            out, skip_out = residual_block(out)
            skip_connections.append(skip_out)

    if use_skip_connections:
        out = layers.Merge(mode='sum')(skip_connections)
    out = layers.Activation('relu')(out)
    out = layers.Convolution1D(nb_output_bins, 1, border_mode='same',
                               W_regularizer=l2(final_l2))(out)
    out = layers.Activation('relu')(out)
    out = layers.Convolution1D(nb_output_bins, 1, border_mode='same')(out)

    if not learn_all_outputs:
        raise DeprecationWarning('Learning on just all outputs is wasteful, now learning only inside receptive field.')
        out = layers.Lambda(lambda x: x[:, -1, :], output_shape=(out._keras_shape[-1],))(
            out)  # Based on gif in deepmind blog: take last output?

    out = layers.Activation('softmax', name="output_softmax")(out)
    model = Model(input, out)

    receptive_field, receptive_field_ms = compute_receptive_field()

    _log.info('Receptive Field: %d (%dms)' % (receptive_field, int(receptive_field_ms)))
    return model        