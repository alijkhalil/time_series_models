# Import statements
import sys, os
import numpy as np

from keras import backend as K

from keras import activations
from keras import constraints
from keras import initializers
from keras import regularizers

from keras.models import Model, Sequential
from keras.engine.topology import InputSpec, Layer
from keras.layers import Input, Dense, Lambda
from keras.layers import Recurrent, LSTM, GRU
from keras.layers import add, multiply, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import _time_distributed_dense
from keras.regularizers import l2
from keras.legacy import interfaces
from keras.layers.core import Activation

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../")

from dl_utilities.layers import general as dl_layers  # Requires 'sys.path.append' call above


# Global variables
DEFAULT_DEPTH=4


    
# RHN Cell definition
class RHN(Recurrent):
    """
    Implementation of Recurrent Highway Network (RHN).
    
    By design, this RNN cell was devised to increase the 
    expressiveness of the model by adding more depth at 
    each time step (relative to a generic RNN - like a 
    GRU or LSTM cell).  In other words, the number of 
    layers between arbitrary hidden states is greater 
    for a RHN compared to a stacked LSTM, for instance.
    
    # Issues:
        -Often requires LN for stability
            -seems that "no var" LN works significantly better than standard "var" LN
        -Does not seem to be as efficient as possible unless you tweak RHN algorithm
            -the current and previous hidden gating needs to be binary between the two components
            -commented out original RHN algorithm (in 'step' function) though for use if desired
                -reason being that does not work consistently
            -to use strict RHN implementation, uncomment lines with starting with '#'
                -also comment out lines with '#COMMENT OUT' notation
        -LN layers seemed to help a bit with efficacy of training
            -however, should not be necessary
        
    # Arguments
        units: Positive integer, dimensionality of the output space.
        depth: Number of intermediary hidden state layers before 
                outputing final state.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
            
    # References
        - [RHN] (https://arxiv.org/pdf/1607.03474.pdf)
    """

    @interfaces.legacy_recurrent_support
    def __init__(self, 
                 units,
                 depth=DEFAULT_DEPTH,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',   
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
                 
                 
        super(RHN, self).__init__(**kwargs)
        
        self.units = units
        self.depth = depth
        
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        
        self.internal_layers = {}

        
    # Function override to ensure that it is never called in this type of recurrent layer
    def add_weight(self, shape, initializer,
                   name=None,
                   trainable=True,
                   regularizer=None,
                   constraint=None):
                   
        raise ValueError("The 'add_weight' function is not allowed for this " \
                            "particular RNN layer. Use 'add_layer' function instead.")  
                   
    
    
    # Layer functions
    def add_layer(self, layer, name, input_shape):
        self.internal_layers[name] = layer
        self.internal_layers[name].build(input_shape)
        
        
    def get_layer(self, name):
        return self.internal_layers[name]
    
    
    # Weight override functions 
    @property
    def trainable_weights(self):
        tmp_weights = []
        
        for i_layer in self.internal_layers.values():
            tmp_weights.extend(i_layer.trainable_weights)
    
        return tmp_weights

        
    @property
    def non_trainable_weights(self):
        tmp_weights = []
        
        for i_layer in self.internal_layers.values():
            tmp_weights.extend(i_layer.non_trainable_weights)
    
        return tmp_weights
    
    
    # Should contain all layers with weights associated with them
    def build(self, input_shape):
        # Get relavent shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        
        batch_size = input_shape[0] if self.stateful else None
        input_dim = input_shape[2]
        
        input_shape = (batch_size, input_dim)
        hidden_input_shape = (batch_size, self.units) 
        
        
        # Set input dimension and spec values 
        self.input_dim = input_dim
        self.input_spec = InputSpec(shape=(batch_size, None, self.input_dim)) 
        self.state_spec = InputSpec(shape=(batch_size, self.units))

                           
        # Initialize states to None
        self.states = [None]
        if self.stateful:
            self.reset_states()

            
        # Define needed layers
        input_layer = Dense(self.units * 3,	
                                use_bias=False,
                                kernel_initializer=self.kernel_initializer,
                                kernel_regularizer=self.kernel_regularizer,
                                kernel_constraint=self.kernel_constraint,
                                name='input_kernel')
                                
        self.add_layer(input_layer, 'input_layer', input_shape)

        
        for i in range(self.depth):
            layer_key_str = ("recurrent_layer%d" % (i+1))
            recurrent_layer = Dense(self.units * 3,
                                        use_bias=self.use_bias,
                                        kernel_initializer=self.recurrent_initializer,
                                        kernel_regularizer=self.recurrent_regularizer,
                                        kernel_constraint=self.recurrent_constraint,
                                        name=layer_key_str)
            
            self.add_layer(recurrent_layer, layer_key_str, hidden_input_shape)
        
        
            layer_key_str = ("LN_layer%d" % (i+1))
            LN_layer = dl_layers.LN(use_variance=False, name=layer_key_str)
            
            self.add_layer(LN_layer, layer_key_str, hidden_input_shape)
       
       
        # Set built flag
        self.built = True
        

    # Called immediately before RNN step as part of set-up process
    # Passed as "state" element (after output and intermediary states) as result of RNN iteration
    # Normally used to pass dropout masks
    def get_constants(self, inputs, training=None):
        constants = []
        input_dim = K.int_shape(inputs)[-1]
        
        if 0.0 < self.dropout < 1.0:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, input_dim))
            
            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            dp_mask = K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training)
                                                
            constants.append(dp_mask)
            
        else:
            dp_mask = K.cast_to_floatx(1.)
            constants.append(dp_mask)
            
            
        if 0 < self.recurrent_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))
            
            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)

            rec_dp_mask = [ K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training)
                            for _ in range(self.depth - 1) ]
                                                
            constants.append(rec_dp_mask)
            
        else:
            rec_dp_mask = [ K.cast_to_floatx(1.) for _ in range(self.depth - 1) ]
            constants.append(rec_dp_mask)

            
        return constants

        
    def step(self, inputs, states):
        s_tm1 = states[0]       # from states (returned below in order)
        
        input_dp_mask = states[1]		# from "get_constants"
        rec_dp_mask = states[2]		# from "get_constants"

        
        # Apply dropout and get initial hidden state and input layers
        dp_input = inputs * input_dp_mask

        cur_input_layer = self.get_layer('input_layer')
        cur_rec_layer = self.get_layer('recurrent_layer1')       
        ln_layer = self.get_layer('LN_layer1')       
        
        print(cur_input_layer, cur_rec_layer)
        
        
        # RHN logic (for processing previous hidden state plus input
        matrix_x = cur_input_layer(dp_input)
        matrix_inner = cur_rec_layer(s_tm1)

        h_x = matrix_x[:, :self.units]
        h_inner = matrix_inner[:, :self.units]
        h_final = self.activation(h_x + h_inner)

        t_x = matrix_x[:, self.units: 2 * self.units]
        t_inner = matrix_inner[:, self.units: 2 * self.units]
        #t_final = self.recurrent_activation(t_x + t_inner)
        t_final = Activation('relu')(t_x + t_inner)       # COMMENT OUT
        
        c_x = matrix_x[:, 2 * self.units:]
        c_inner = matrix_inner[:, 2 * self.units:]
        #c_final = self.recurrent_activation(c_x + c_inner)
        c_final = Activation('relu')(c_x + c_inner)       # COMMENT OUT
        
        z = self.recurrent_activation(t_final - c_final)    # COMMENT OUT
        #s_next = c_final * s_tm1 + t_final * h_final
        s_next = z * s_tm1 + (1 - z) * h_final      # COMMENT OUT
        s_next = ln_layer(s_next)
        
        
        # For loop with core functionality for remaining "deeper" hidden states
        for i in range(self.depth - 1): 
            # Apply dropout and get current layer's weights
            dp_rec_hidden = s_next * rec_dp_mask[i]

            layer_key_str = ("recurrent_layer%d" % (i+2))            
            cur_rec_layer = self.get_layer(layer_key_str)
            
            
            # RHN logic (for recursively processing state)
            matrix_inner = cur_rec_layer(dp_rec_hidden)

            h = matrix_inner[:, :self.units]
            h_final = self.activation(h)

            t = matrix_inner[:, self.units: 2 * self.units]
            #t_final = self.recurrent_activation(t)
            t_final = Activation('relu')(t)     # COMMENT OUT
            
            c = matrix_inner[:, 2 * self.units:]
            #c_final = self.recurrent_activation(t)
            c_final = Activation('relu')(c)     # COMMENT OUT

            z = self.recurrent_activation(t_final - c_final)    # COMMENT OUT
            
            #s_next = c_final * s_next + t_final * h_final
            s_next = z * s_next + (1 - z) * h_final     # COMMENT OUT

            
            # Normalize layer
            LN_key_str = ("LN_layer%d" % (i+2))
            ln_layer = self.get_layer(LN_key_str)
            
            s_next = ln_layer(s_next)
            
            
        # Set learning phase flag
        if 0 < self.dropout + self.recurrent_dropout:
            s_next._uses_learning_phase = True
        
        
        # Return output and states
        return s_next, [s_next]

        
    def get_config(self):
        config = {'units': self.units,
                  'depth': self.depth,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
                  
        base_config = super(RHN, self).get_config()
        
        return dict(list(base_config.items()) + list(config.items()))
