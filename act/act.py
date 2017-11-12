# Import statements
import sys
import numpy as np
import tensorflow as tf

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

sys.path.append("..")
from dl_utilities.layers import act as act_layers
from dl_utilities.layers import general as dl_layers


# Global variables
DEFAULT_EPSILON=0.01
DEFAULT_MAX_ITERS=25
DEFAULT_PONDER_COST=1E-7



# ACT Cell definition
class ACT_Cell(Recurrent):
    """
    This ACT cell represents the core logic for the Adaptive Computation Time 
    model.  In theory, it is a RNN designed to learn the appropiate amount 
    of time needed to process a recurrent input at particular time step.  To 
    do that, the model weigh the intermediary states/output and also assign 
    a penalty to over-computation so that each input is not simply processed 
    for the maximum allowed time.
    
    # Issues:
        -Keras does not have a backend function for looping based on a predicate
            -approximated using 'loop_layer' (from 'general' layers in the 'dl_utilities' repo)
        -Instability of training RNN's on Keras
            -particularly with high dropout rate
        
    # Arguments
        output_units: Positive integer, dimensionality of the hidden state space.
        hidden_units: Positive integer, dimensionality of the output space.
        eplison_val: Smallest acceptable weight for an hidden/output state at an iteration.
        max_computation_iters: Maximum number of iterations per time step.
        ponder_cost: Weight of ponder cost component of specialized ACT loss function.
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
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Setting it to true will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            Use a forget gate with a positive bias to add gradient flow initially.
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
        - [ACT] (https://arxiv.org/pdf/1603.08983v4.pdf)
    """
    
    @interfaces.legacy_recurrent_support
    def __init__(self, 
                 hidden_units,
                 output_units=None, 
                 eplison_val=DEFAULT_EPSILON,
                 max_computation_iters=DEFAULT_MAX_ITERS,
                 ponder_cost=DEFAULT_PONDER_COST,
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
         
        super(ACT_Cell, self).__init__(**kwargs)
        
        if output_units is None:
            raise ValueError("The 'output_units' variable must be an integer.")
                
        self.hidden_units = hidden_units
        self.output_units = output_units
        
        self.eplison_val = eplison_val
        self.max_computation_iters = max_computation_iters
        self.ponder_cost = ponder_cost        
        
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
        self.return_state = True

        
    # Override to ensure that implementation does rely on Keras to have 'return_states' support
    def call(self, inputs, mask=None, initial_state=None, training=None):
        if initial_state is not None:
            if not isinstance(initial_state, (list, tuple)):
                initial_states = [initial_state]
            else:
                initial_states = list(initial_state)
                
        if isinstance(inputs, list):
            initial_states = inputs[1:]
            inputs = inputs[0]
        elif self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_state(inputs)

        if len(initial_states) != len(self.states):
            raise ValueError('Layer has ' + str(len(self.states)) +
                             ' states but was passed ' +
                             str(len(initial_states)) +
                             ' initial states.')
                             
        input_shape = K.int_shape(inputs)
        if self.unroll and input_shape[1] is None:
            raise ValueError('Cannot unroll a RNN if the '
                             'time dimension is undefined. \n'
                             '- If using a Sequential model, '
                             'specify the time dimension by passing '
                             'an `input_shape` or `batch_input_shape` '
                             'argument to your first layer. If your '
                             'first layer is an Embedding, you can '
                             'also use the `input_length` argument.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a `shape` '
                             'or `batch_shape` argument to your Input layer.')
                             
        constants = self.get_constants(inputs, training=None)
        preprocessed_input = self.preprocess_input(inputs, training=None)
        last_output, outputs, states = K.rnn(self.step,
                                             preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=input_shape[1])
                                             
        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_update(updates, inputs)

        if self.return_sequences:
            output = outputs
        else:
            output = last_output

        # Properly set learning phase
        if getattr(last_output, '_uses_learning_phase', False):
            output._uses_learning_phase = True

        if not isinstance(states, (list, tuple)):
            states = [states]
        else:
            states = list(states)
            
        # Return output and states combined (in a list)    
        return [output] + states

    
    # No suppport for "add_weight" - instead use higher-level Layers and "add_layer" function
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
    
    
    # Weight override function 
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
        
    
    # Override generic RNN functions because states are unique for this cell
    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
            
        if self.return_sequences:
            output_shape = (input_shape[0], input_shape[1], self.output_units)
        else:
            output_shape = (input_shape[0], self.output_units)

        state_shape = [ ]
        for i_spec in self.state_spec:
            cur_dim = i_spec.shape[-1]
            state_shape.append((input_shape[0], cur_dim))
            
        return [output_shape] + state_shape
    
            
    def get_initial_state(self, inputs):
        # Build an all-zero tensor of shape (samples, 1)        
        initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        
        # Build zero-ed intermediate states by getting dimension of each state
        initial_states = []
        for i_spec in self.state_spec:
            cur_dim = i_spec.shape[-1]
            tmp_state = K.tile(initial_state, [1, cur_dim])  # (samples, cur_dim)
            
            initial_states.append(tmp_state)
        
        return initial_states
    
    
    def reset_states(self, states_value=None):
        if not self.stateful:
            raise AttributeError('Layer must be stateful.')
            
        if not self.input_spec:
            raise RuntimeError('Layer has never been called '
                               'and thus has no states.')
                               
        batch_size = self.input_spec.shape[0]
        if not batch_size:
            raise ValueError('If a RNN is stateful, it needs to know '
                             'its batch size. Specify the batch size '
                             'of your input tensors: \n'
                             '- If using a Sequential model, '
                             'specify the batch size by passing '
                             'a `batch_input_shape` '
                             'argument to your first layer.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a '
                             '`batch_shape` argument to your Input layer.')
                             
        if states_value is not None:
            if not isinstance(states_value, (list, tuple)):
                states_value = [states_value]
                
            if len(states_value) != len(self.states):
                raise ValueError('The layer has ' + str(len(self.states)) +
                                 ' states, but the `states_value` '
                                 'argument passed '
                                 'only has ' + str(len(states_value)) +
                                 ' entries')
                                 
        if self.states[0] is None:
            self.states = []
            for i_spec in self.state_spec:
                cur_dim = i_spec.shape[-1]
                self.states.append(K.zeros((batch_size, cur_dim)))
                
            if not states_value:
                return
                
        for i, state_tuple in enumerate(zip(self.states, self.state_spec)):
            state, state_spec = state_tuple
            state_dim = state_spec.shape[-1]
            
            if states_value:
                value = states_value[i]

                if value.shape != (batch_size, state_dim):
                    raise ValueError(
                        'Expected state #' + str(i) +
                        ' to have shape ' + str((batch_size, state_dim)) +
                        ' but got array with shape ' + str(value.shape))
            else:
                value = np.zeros((batch_size, state_dim))
                
            K.set_value(state, value)
    
    
    # Should contain all layers with weights associated with them
    def build(self, input_shape):
        # Get relavent shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        
        batch_size = input_shape[0] if self.stateful else None
        input_dim = input_shape[2]
        
        GRU_input_shape = (batch_size, None, input_dim + 1)
        hidden_input_shape = (batch_size, self.hidden_units) 
        output_shape = (batch_size, self.hidden_units)                                        
        
        # Set input dimension and spec values 
        self.input_dim = input_dim
        self.input_spec = InputSpec(shape=(batch_size, None, self.input_dim)) 
        self.state_spec = [InputSpec(shape=(batch_size, self.hidden_units)),    # states
                           InputSpec(shape=(batch_size, 1)),                    # counters
                           InputSpec(shape=(batch_size, 1))]                    # remainders


                           
        # Initialize states to None
        self.states = [None, None, None]
        if self.stateful:
            self.reset_states()

            
        # Define needed layers
        input_layer = Dense(self.hidden_units * 3,	
                                use_bias=self.use_bias,
                                kernel_initializer=self.kernel_initializer,
                                kernel_regularizer=self.kernel_regularizer,
                                kernel_constraint=self.kernel_constraint,
                                name='input_kernel')
                                
        self.add_layer(input_layer, 'input_layer', GRU_input_shape)

        
        recurrent_layer1 = Dense(self.hidden_units * 2,
                                    use_bias=self.use_bias,
                                    kernel_initializer=self.recurrent_initializer,
                                    kernel_regularizer=self.recurrent_regularizer,
                                    kernel_constraint=self.recurrent_constraint,
                                    name='recurrent_kernel1')
                                 
        self.add_layer(recurrent_layer1, 'recurrent_layer1', hidden_input_shape)

        
        recurrent_layer2 = Dense(self.hidden_units,
                                    use_bias=self.use_bias,
                                    kernel_initializer=self.recurrent_initializer,
                                    kernel_regularizer=self.recurrent_regularizer,
                                    kernel_constraint=self.recurrent_constraint,
                                    name='recurrent_kernel2')
                                 
        self.add_layer(recurrent_layer2, 'recurrent_layer2', hidden_input_shape)
            
        output_layer = Dense((self.output_units + 1),
                                    use_bias=self.use_bias,
                                    kernel_initializer=self.kernel_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    kernel_constraint=self.kernel_constraint,
                                    name='output_layer')
                                 
        self.add_layer(output_layer, 'output_layer', output_shape)
            
            
        # Set built flag
        self.built = True
        

    # Called immediately before RNN step as part of set-up process
    # Passed as "state" element (after output and intermediary states) as result of RNN iteration
    # Normally used to pass dropout masks
    def get_constants(self, inputs, training=None):
        constants = []
        
        # Set ones tensor with shape of hidden layer
        ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
        ones = K.tile(ones, (1, self.hidden_units))
        
        # Get input and recurrent dropout masks
        if 0.0 < self.dropout < 1.0:            
            dp_mask = K.in_train_phase(K.dropout(ones, self.dropout),
                                            ones,
                                            training=training)                                                
                                            
        else:
            dp_mask = ones
        
        constants.append(dp_mask)
            
            
        if 0 < self.recurrent_dropout < 1:            
            rec_dp_mask = K.in_train_phase(K.dropout(ones, self.dropout),
                                            ones,
                                            training=training)
                                            
        else:
            rec_dp_mask = ones
                                                
        constants.append(rec_dp_mask)
        
        
        # Return them        
        return constants

        
    def step(self, inputs, states):
        # Break down previous output/states
        s_tm1 = states[0]       # from states (returned below in order)
        init_counter = states[1]       # from states (returned below in order)
        init_remainder = states[2]       # from states (returned below in order)
        
        dp_mask = states[3]		# from "get_constants"
        rec_dp_mask = states[4]		# from "get_constants"

        
        # Input flag and counters
        new_input_flag = K.variable(1, dtype='float32')

        counter = act_layers.ResetLayer()(init_counter)
        counter_condition = act_layers.SetterLayer(1.0)(init_counter)

        prob = act_layers.ResetLayer()(init_remainder)
        prev_not_done_mask = act_layers.SetterLayer(1.0)(init_counter)

        s_cur = s_tm1
        acclum_state = act_layers.ResetLayer()(s_tm1)
        acclum_output = act_layers.CreateCustomShapeLayer(self.output_units)(inputs)
        
        
        # Define loop condition and step functions
        def cond_func(inputs, new_input_flag, acclum_output, 
                        s_cur, acclum_state,
                        prob, counter, counter_condition,
                        dp_mask, rec_dp_mask, not_done_mask):
            
            final_cond = multiply([not_done_mask, counter_condition])
            return K.any(final_cond)
        
        
        def step_func(inputs, new_input_flag, acclum_output, 
                        s_cur, acclum_state,
                        prob, counter, counter_condition,
                        dp_mask, rec_dp_mask, not_done_mask):
                        
            # Constants that can be generate on each iteration (in tf.while_loop) if needed
            x_bin_init = concatenate([inputs, K.ones_like(counter)], axis=-1)        
            x_bin_non_init = concatenate([inputs, K.zeros_like(counter)], axis=-1)
                    
            one_minus_eps = act_layers.SetterLayer(1.0 - self.eplison_val)(counter)
            max_computation = act_layers.SetterLayer(self.max_computation_iters)(counter)
            
                    
            # Augment input with binary flag and pass input/hidden state to transition RNN
            x_bin = act_layers.FlagLayer(new_input_flag)([x_bin_init, x_bin_non_init])
            
            
            # Normalize hidden state (with zero mean) and then apply dropout to it (during training)
            s_cur = s_cur - K.mean(s_cur, [1], keepdims=True)
            dp_hidden = s_cur * dp_mask
            dp_rec_hidden = s_cur * rec_dp_mask
                        
                        
            # GRU logic (for recursively mixing state with input)
            matrix_x = self.get_layer('input_layer')(x_bin)
            matrix_inner = self.get_layer('recurrent_layer1')(dp_rec_hidden)

            x_z = matrix_x[:, :self.hidden_units]
            x_r = matrix_x[:, self.hidden_units: 2 * self.hidden_units]
            recurrent_z = matrix_inner[:, :self.hidden_units]
            recurrent_r = matrix_inner[:, self.hidden_units: 2 * self.hidden_units]

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            x_h = matrix_x[:, 2 * self.hidden_units:]
            recurrent_h = self.get_layer('recurrent_layer2')(r * dp_rec_hidden)
            hh = self.activation(x_h + recurrent_h)
            s_next = z * s_cur + (1 - z) * hh
            
            
            # Use the hidden state to get output and stop confidence
            init_output = self.get_layer('output_layer')(dp_hidden)
            output = dl_layers.crop(0, self.output_units)(init_output)
        
            halting_unit = dl_layers.crop(self.output_units)(init_output)
            new_percent = Activation('sigmoid')(halting_unit)
            
            
            # Determine if batch items are done yet to update running probability
            tmp_probs = add([prob, new_percent])
            less_than_limit = act_layers.CompLayer()([tmp_probs, one_minus_eps])
            new_not_done_mask = multiply([less_than_limit, not_done_mask])
            
            add_vals = multiply([new_percent, new_not_done_mask])
            prob = add([prob, add_vals])
            
            
            # Update counter for examples still being processed
            counter = add([counter, new_not_done_mask])
            counter_condition = act_layers.CompLayer()([counter, max_computation])
            
            
            # Designate those elements requiring remainder and those requiring percentage
            final_iter_prob_condition = multiply([less_than_limit, counter_condition])
            final_iter_remainder_condition = act_layers.OneMinusLayer()(final_iter_prob_condition)
            

            # If percentage less than one_minus_ep, use "new_percent", otherwise use prob remainder
            prob_weights = multiply([final_iter_prob_condition, new_percent])
            remainder_weights = multiply([final_iter_remainder_condition, 
                                            act_layers.OneMinusLayer()(prob)])
            
            culm_update_weights = add([prob_weights, remainder_weights])
            culm_update_weights = multiply([culm_update_weights, not_done_mask])
            
            
            # Set accumulated values by adding weighted amount onto previous accumulated value
            new_s_term = multiply([culm_update_weights, s_cur])
            acclum_state = add([acclum_state, new_s_term])
            
            new_output_term = multiply([culm_update_weights, output])
            acclum_output = add([acclum_output, new_output_term])
               
               
            # Set input flag and done mask variables
            s_cur = s_next
            not_done_mask = new_not_done_mask
            new_input_flag = act_layers.ResetLayer()(new_input_flag)

        
            return [inputs, new_input_flag, acclum_output, s_cur, acclum_state, 
                        prob, counter, counter_condition, dp_mask, rec_dp_mask, 
                        not_done_mask]
        
        
        # Call loop layer
        looper = dl_layers.loop_layer(cond_func, step_func)
        loop_inputs = [ inputs, new_input_flag, acclum_output, 
                        s_cur, acclum_state,
                        prob, counter, counter_condition,
                        dp_mask, rec_dp_mask,
                        prev_not_done_mask ]
        
        acclum_output, acclum_state, prob, counter = looper(loop_inputs)

        
        # Get accumulated counter and remainder values
        new_counter = K.cast_to_floatx(self.ponder_cost) * counter
        new_counter = add([init_counter, new_counter])
        
        new_remainders = K.cast_to_floatx(1.) - prob
        new_remainders = K.cast_to_floatx(self.ponder_cost) * new_remainders        
        new_remainders = add([init_remainder, new_remainders])
        
        
        # Set learning phase flag
        if 0 < self.dropout + self.recurrent_dropout:
            acclum_state._uses_learning_phase = True
            acclum_output._uses_learning_phase = True
        
        
        # Return output and states
        return acclum_output, [acclum_state, new_counter, new_remainders]

        
    def get_config(self):
        config = {'hidden_units': self.hidden_units,
                  'output_units': self.output_units,
                  'eplison_val': self.eplison_val,
                  'max_computation_iters': self.max_computation_iters,
                  'ponder_cost': self.ponder_cost,
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
                  
        base_config = super(ACT_Cell, self).get_config()
        
        return dict(list(base_config.items()) + list(config.items()))
