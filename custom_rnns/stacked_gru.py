# Import statements
from keras import backend as K

from keras.layers import Input, GRU


# Global variables
DEFAULT_DEPTH=5


def StackedGRU(hidden_dim, input_tensor=None, depth=DEFAULT_DEPTH, dropout=0., recurrent_dropout=0.):
    # Ensure the input is acceptable
    if input_tensor is not None:
        if not K.is_keras_tensor(input_tensor):
            input = Input(tensor=input_tensor)
        else:
            input = input_tensor
    else:
        raise ValueError("The 'input_tensor' must be provided.")

        
    # Stack designated number of GRU cells
    next_state = input
    
    for i in range(depth):
        ret_seq = True
        if i == (depth-1):
            ret_seq = False
        
        GRU_layer = GRU(hidden_dim, return_sequences=ret_seq, implementation=2, 
                            dropout=dropout, recurrent_dropout=recurrent_dropout, 
                            name=("GRU%d" % i))
        next_state = GRU_layer(next_state)
     
     
    # Return output
    return next_state
