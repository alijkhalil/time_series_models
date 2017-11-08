# time_series_models
Repository with a variety of RNN architectures architectures implemented in Keras.

Each architecture can be tested out-of-the-box with the tester.py script using:
        tester.py <model_type> [--dropout [<float_value>]] [--recurrent_dropout [<float_value>]]    
                                [--depth [<num_of_layers_per_time_step>]]
								[--input_dim [<dimension_of_input_embedding>]]
								[--hidden_dim [<dimension_of_hidden_state>]]
								[--epochs [<num_of_training_epochs>]] [--save_weights]

The test script reqiures the "dl_utilities" package (easily attainable using the "set_up.sh" shell script).

Like in the test script, it is possible to initiate new models and train them by simply importing the needed code.

In other words, each directory should be able to act as its own Python package by importing using:
  from directory_name import filename as <desired_model_name_handle>

This is why each directory has its own empty "__init__.py" file.

Though these packages should be usable with any Keras backend, they have only been tested with a TensorFlow backend.