class RNN_Config(object):

    """

    Class used to configure the recurrent neural network (RNN) object properties.

    """

    # Summaries
    summaries_name_scope = ' '

    # Learning rate
    learning_rate = 0.001

    # Number of inputs, ouputs, hidden units, time steps, layers
    number_inputs = 0
    number_outputs = 0
    number_hidden_units = 0
    number_steps = 0
    number_layers = 0

    # Batch size
    batch_size = 0

    # Configuration (Max gradient norm)
    max_grad_norm = 5

    # Enable to compute the gradient difference loss (GDL)
    enable_gdl_optimization = False
    
    # Cell type
    cell_type = ''

    # Custom cell type
    custom_lstm_cell_type = ''