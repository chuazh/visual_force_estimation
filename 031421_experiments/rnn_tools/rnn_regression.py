from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from rnn_tools.vanilla_lstm import VanillaLSTMCell


class RNN_Regression:

    """

    This class implements a Recurrent Neural Network (RNN) for a regression task.

    """

    def __init__(self, config):

        # --------------------------------------------------------------------------------------------------------------
        # Values from configuration file
        # --------------------------------------------------------------------------------------------------------------

        # Summaries
        self.summaries_name_scope = config.summaries_name_scope

        # Learning rate
        self.learning_rate = config.learning_rate

        # Number of inputs, ouputs, hidden units, time steps, layers.
        self.number_inputs = config.number_inputs
        self.number_outputs = config.number_outputs
        self.number_hidden_units = config.number_hidden_units
        self.number_steps = config.number_steps
        self.number_layers = config.number_layers

        # Batch size
        self.batch_size = config.batch_size

        # Configuration (Max gradient norm)
        self.max_grad_norm = config.max_grad_norm
        
        # Enable to compute the gradient difference loss (GDL).
        self.enable_gdl_optimization = config.enable_gdl_optimization
        
        self.cell_type = config.cell_type

        self.custom_cell_type = config.custom_cell_type

        self.scale_data = {'X-Video': 1, 'X-Tool': 1, 'Y': 1}
        
        self.select_input_signals = {'X-Video': True, 'X-Tool': True}

        self.step_index = 1

        self.start_output_time_step = int(0.5 * self.number_steps)

        # --------------------------------------------------------------------------------------------------------------
        # Objects used by the computational graph
        # --------------------------------------------------------------------------------------------------------------

        # RNN: Initial state
        self.initial_state = None

        # RNN: Reset state
        self.reset_state = None

        # RNN State
        self.state = None

        # RNN Final state
        self.final_state = None

        # RNN Model
        self.cell = None

        # RNN Weights
        self.weights = None

        # RNN Biases
        self.biases = None

        # RNN Output
        self.outputs = None

        # RNN reduced output
        self.reduced_output = None

        # RNN last output
        self.last_output = None

        # Input/output
        self.X = None
        self.Y = None

        # Dropout probability
        self.P_dropout = None

        # Prediction
        self.prediction_op = None

        # Gradients
        self.Gradient_Y = None
        self.Gradient_Prediction =  None

        # Gradient difference loss
        self.gradient_difference_loss = None

        # Residual
        self.residual_op = None

        # Scaled residuals
        self.scaled_residual_op = None

        # L2 norm
        self.l2_norm = {
            'Residuals': None,
            'Ground Truth': None,
            'Prediction': None,
        }

        self.normalized_tensor = {
            'Ground Truth | Normalized': None,
            'Prediction | Normalized': None
        }

        # L1 norm
        self.l1_norm = {
            'Residuals': None,
            'Ground Truth': None,
            'Prediction': None
        }

        # Force Error Vector
        self.force_error_vector = {
            '6D Vector': None,
            'Fx': None,
            'Fy': None,
            'Fz': None,
            'Tx': None,
            'Ty': None,
            'Tz': None
        }

        # Absolute error
        self.absolute_error_op = None

        # Relative error
        self.relative_error_op = None

        # Root Mean Square (RMS) Error
        self.rms_error_op = None

        # Mean Absolute Error (MAE)
        self.mae_error_op = None

        # These constants weigh the contribution of the L2 and GDL losses into the final loss function.
        # loss = lambda_l2 * loss_l2 + lambda_gdl * loss_gdl.
        self.lambda_l2 = 0.75
        self.lambda_gdl = 1.0 - self.lambda_l2

        # Gradients
        self.gradient_update_op = None

        # Cost
        self.cost_op = None
        self.cost_contributions = None
        self.cost_1 = None
        self.cost_2 = None
        self.cost_3 = None
        self.selected_cost_function = None
        self.dual_loss = None
        self.gradients = None

        # Accuracy operation
        self.accuracy_op = None

        # Optimizer
        self.optimizer = None

        # Training operation
        self.train_op = None

        # Epsilon constant
        self.epsilon = 1.0/100.0

        # Tolerance
        self.tolerance = 0.001

    def linear_layer(self, input_tensor, n_hidden, n_output, scope_name):

        """
        Creates a dense layer with a linear activation.

        :param input_tensor: Input tensor.
        :param n_hidden: Number of hidden units.
        :param n_output: Number of output units.
        :param scope_name: Scope name.
        :return: The result of the linear operation: input * weight + bias.
        """

        with tf.variable_scope(scope_name):

            self.weights = tf.Variable(tf.random_normal([n_hidden, n_output]), name='weight')
            self.biases = tf.Variable(tf.random_normal([n_output]), name='bias')
            output = tf.matmul(input_tensor, self.weights) + self.biases

        return output

    def create_rnn_graph(self, X, P_dropout):

        """

        Creates RNN computational graph.

        :param X: Input tensor.
        :param P_dropout: Dropout probability.
        :return:
        """

        print('[ RNN Regression ] Creating RNN Computational Graph -> Start')

        n_input = self.number_inputs
        n_output = self.number_outputs
        n_hidden = self.number_hidden_units
        n_steps = self.number_steps

        # --------------------------------------------------------------------------------------------------------------
        # Reshape input tensor X
        # --------------------------------------------------------------------------------------------------------------

        print('X Shape | Before Reshape Operation: ')
        print('  a) Tensor shape: ', X.get_shape())

        with tf.name_scope('input_data_reshape'):
            X = [tf.squeeze(input_, [1]) for input_ in tf.split(1, n_steps, X)]

        print('X Shape | After Reshape Operation:')
        print('  a) List size: ', len(X))
        print('  b) Tensor shape of each element in the list: ', X[0].get_shape())

        # --------------------------------------------------------------------------------------------------------------
        # RNN Graph definition
        # --------------------------------------------------------------------------------------------------------------

        with tf.name_scope('recurrent_neural_net'):

            # List of cells.
            list_of_cells = self.number_layers * [None]

            # Lstm cell.
            lstm_cell = None

            # Cell type string.
            cell_type_string = ''
            
            # Custom LSTM cell type: 'CIFG', 'Complete'
            custom_lstm_cell_type = self.custom_cell_type
            
            for k in range(0, self.number_layers):
                
                # LSTM cell (from Tensorflow library).
                if self.cell_type == 'LSTM-Tensorflow':
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(
                        num_units=n_hidden,
                        input_size=None,
                        use_peepholes=True,
                        cell_clip=None,
                        initializer=None,
                        num_proj=None,
                        num_unit_shards=1,
                        num_proj_shards=1,
                        forget_bias=1.0
                    )
                    
                    cell_type_string = self.cell_type
                
                # GRU cell (from Tensorflow library)
                elif self.cell_type == 'GRU-Tensorflow':
                    
                    lstm_cell = tf.nn.rnn_cell.GRUCell(num_units=n_hidden)
                    
                    cell_type_string = self.cell_type
                    
                # Custom LSTM
                elif self.cell_type == 'LSTM-Custom':
                    
                    cell_type_string = self.cell_type + ' | ' + custom_lstm_cell_type

                    # Vanilla LSTM Cell
                    if k == 0:
                        lstm_cell = VanillaLSTMCell(input_size=n_input, num_blocks=n_hidden, option=custom_lstm_cell_type)
                    else:
                        lstm_cell = VanillaLSTMCell(input_size=n_hidden, num_blocks=n_hidden, option=custom_lstm_cell_type)
    
                # Dropout layer.
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=P_dropout[k])
    
                # List of cells.
                list_of_cells[k] = lstm_cell
                
            # Multiple cells
            self.cell = tf.nn.rnn_cell.MultiRNNCell(list_of_cells)
            
            print('Cell Type = %s | Number of Layers = %d' % (cell_type_string, self.number_layers))

            # Initial state
            self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)

            # Reset state
            self.reset_state = self.cell.zero_state(self.batch_size, tf.float32)

            # Outputs and state
            self.outputs, self.state = tf.nn.rnn(
                cell=self.cell,
                inputs=X,
                initial_state=self.initial_state
            )

            # ----------------------------------------------------------------------------------------------------------
            # Average the output from RNN (from start_index to end_index).
            # ----------------------------------------------------------------------------------------------------------

            end_index = len(self.outputs)
            out_tensor = None
            start_index = self.start_output_time_step
            output_indices = range(start_index, end_index, 1)

            c = 0
            for i in output_indices:
                if i == start_index:
                    out_tensor = self.outputs[i]
                else:
                    out_tensor += self.outputs[i]
                c += 1

            self.reduced_output = out_tensor / c

            print('Output Tensor Y | Averaged over %d of %d time steps.' % (len(output_indices), end_index))

            # ----------------------------------------------------------------------------------------------------------
            # Last output and final state
            # ----------------------------------------------------------------------------------------------------------

            # Last output of the recurrent neural net.
            self.last_output = self.reduced_output

            # Final state.
            self.final_state = self.state

        # --------------------------------------------------------------------------------------------------------------
        #  Dense layer (with a linear activation) added on top on the RNN.
        # --------------------------------------------------------------------------------------------------------------

        linear_activation = self.linear_layer(
            input_tensor=self.last_output,
            n_hidden=n_hidden,
            n_output=n_output,
            scope_name='output_layer'
        )

        print('[ RNN Regression ] Creating RNN Computational Graph -> End')

        return linear_activation

    def build_rnn(self):

        """

        Builds the RNN computational graph, including input/output placeholders, losses, optimizers, and summaries.

        :return: None.
        """

        print('[ RNN Regression ] Build RNN -> Start')

        # --------------------------------------------------------------------------------------------------------------
        # Input/output tensor variables
        # --------------------------------------------------------------------------------------------------------------

        # RNN input (X)
        self.X = \
            tf.placeholder("float", [self.batch_size, self.number_steps, self.number_inputs], name='X_input')

        # RNN output (Y)
        self.Y = \
            tf.placeholder("float", [self.batch_size, self.number_outputs], name='Y_output')

        # Dropout probability
        self.P_dropout = \
            tf.placeholder("float", [self.number_layers], name='P_drop')

        # Gradient of Y
        self.Gradient_Y = \
            tf.placeholder("float", [self.batch_size, self.number_outputs], name='Y_gradient')

        # Gradient of Prediction
        self.Gradient_Prediction =\
            tf.placeholder("float", [self.batch_size, self.number_outputs], name='Prediction_gradient')

        # --------------------------------------------------------------------------------------------------------------
        # Visualizing the input as a image summary
        # --------------------------------------------------------------------------------------------------------------

        with tf.name_scope('summaries'):
            tf.image_summary('rnn_input', tf.reshape(self.X, [self.batch_size, self.number_steps, self.number_inputs, 1]), 5)

        # --------------------------------------------------------------------------------------------------------------
        # Create the RNN computational graph
        # --------------------------------------------------------------------------------------------------------------

        self.prediction_op = self.create_rnn_graph(
            X=self.X,
            P_dropout=self.P_dropout
        )

        # --------------------------------------------------------------------------------------------------------------
        # Losses and other metrics.
        # --------------------------------------------------------------------------------------------------------------

        # Residual.
        with tf.name_scope('residual'):
            self.residual_op = self.Y - self.prediction_op

        # Relative error.
        with tf.name_scope('relative_error'):
            self.relative_error_op = tf.abs(self.residual_op) / tf.constant(self.tolerance)

        # Root-mean squared error (RMS).
        with tf.name_scope('rms_error'):
            self.rms_error_op = tf.sqrt(
                tf.reduce_mean(tf.square(self.residual_op), reduction_indices=1),
                name='root_mean_square_error'
            )

        # Mean absolute error (MAE).
        with tf.name_scope('mae_error'):
            self.mae_error_op = tf.reduce_mean(
                tf.abs(self.residual_op),
                reduction_indices=1,
                name='mean_absolute_error'
            )

        # Gradient difference loss (GDL).
        with tf.name_scope('gradient_difference_loss'):
            self.gradient_difference_loss = tf.reduce_sum(
                tf.abs(tf.abs(self.Gradient_Y) - tf.abs(self.Gradient_Prediction)),
                reduction_indices=1,
                name='gradient_difference_loss'
            )

        # Error per force component ('Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz').
        with tf.name_scope('force_error_per_component'):

            # Similarity per force component
            self.force_error_vector['6D Vector'] = tf.sqrt(
                tf.reduce_sum(tf.square(self.residual_op), reduction_indices=0), name='6d_error_vector')
            if self.number_outputs == 6:
                force_string = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
            else:
                force_string = ['Fx', 'Fy', 'Fz']
            

            for index in range(0, len(force_string)):
                self.force_error_vector[force_string[index]] = self.force_error_vector['6D Vector'][index]

        # L2-norm of residuals.
        with tf.name_scope('l2_norm'):
            self.l2_norm['Residuals'] = \
                tf.sqrt(tf.reduce_sum(tf.square(self.residual_op), reduction_indices=1), name='l2_norm_residuals')

        # L1-norm of residuals.
        with tf.name_scope('l1_norm'):
            self.l1_norm['Residuals'] = \
                tf.reduce_sum(tf.abs(self.residual_op), reduction_indices=1, name='l1_norm_residuals')

        # Loss contributions.
        with tf.name_scope('loss_contributions'):

            if self.enable_gdl_optimization:
                print('[ RNN Regression ] Gradient Difference Loss Optimization -> Enabled')
            else:
                print('[ RNN Regression ] Gradient Difference Loss Optimization -> Disabled')

            print('  a) Weight for L2 cost: ', self.lambda_l2)
            print('  b) Weight for GDL cost: ', self.lambda_gdl)

            # RMS Error.
            with tf.name_scope('cost_1_loss_l2norm'):
                self.cost_1 = tf.reduce_sum(self.rms_error_op)

            # Gradient difference loss (GDL)
            with tf.name_scope('cost_2_loss_gdl'):
                self.cost_2 = tf.reduce_sum(self.gradient_difference_loss)

            if self.lambda_l2 == 1.0:
                self.cost_contributions = self.lambda_l2 * self.cost_1
            else:
                self.cost_contributions = self.lambda_l2 * self.cost_1 + self.lambda_gdl * self.cost_2

        # Total loss.
        with tf.name_scope('cost'):
            self.cost_op = self.cost_contributions

        # Accuracy: Mean relative error.
        # (You could add other metrics here).
        with tf.name_scope('accuracy'):
            self.accuracy_op = tf.reduce_mean(self.relative_error_op, name='accuracy')

        # --------------------------------------------------------------------------------------------------------------
        # Optimizer.
        # --------------------------------------------------------------------------------------------------------------

        with tf.name_scope('optimizer'):

            # Get trainable variables.
            tvars = tf.trainable_variables()

            # Compute gradients w.r.t. trainable variables.
            self.gradients = tf.gradients(self.cost_op, tvars)

            # Clip gradients.
            grads, _ = tf.clip_by_global_norm(
                self.gradients,
                self.max_grad_norm
            )

            # Optimizer.
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)

            # Trainable object.
            self.train_op = self.optimizer.apply_gradients(zip(grads, tvars))

        # --------------------------------------------------------------------------------------------------------------
        # Tensorflow summaries.
        # --------------------------------------------------------------------------------------------------------------

        with tf.name_scope('summaries'):

            # Cell output
            No = len(self.outputs)
            samples = 16
            if No < 16:
                samples = No
            indices_vector = np.linspace(0, No - 1, samples, dtype=int)
            for i in indices_vector:
                tf.histogram_summary('cell_'+str(No)+'/output_' + str(i), self.outputs[i])

            tf.histogram_summary('cell/reduced_output', self.last_output)
            tf.histogram_summary('cell/state', self.final_state)

            # Network output.
            tf.histogram_summary('output/activation', self.prediction_op)
            tf.histogram_summary('output/weights', self.weights)
            tf.histogram_summary('output/bias', self.biases)

            # Root Mean Square Error (Force Vector): 6D Vector
            tf.histogram_summary('force_error/Force_Vector', self.force_error_vector['6D Vector'])
            tf.scalar_summary('force_error/Fx', self.force_error_vector['Fx'])
            tf.scalar_summary('force_error/Fy', self.force_error_vector['Fy'])
            tf.scalar_summary('force_error/Fz', self.force_error_vector['Fz'])
            if self.number_outputs == 6:
                tf.scalar_summary('force_error/Tx', self.force_error_vector['Tx'])
                tf.scalar_summary('force_error/Ty', self.force_error_vector['Ty'])
                tf.scalar_summary('force_error/Tz', self.force_error_vector['Tz'])

            # Cost
            tf.scalar_summary('cost/total', self.cost_op)
            tf.scalar_summary('cost/term_l2norm', self.cost_1)
            tf.scalar_summary('cost/term_gdl', self.cost_2)

            # Accuracy: Mean Relative Error (MRE)
            tf.scalar_summary('accuracy/mean_relative_error', self.accuracy_op)

            # Similarity between predicted and ground truth vectors:
            # a) L1 norm
            # b) L2 norm
            tf.histogram_summary('residuals/l1_norm', self.l1_norm['Residuals'])
            tf.histogram_summary('residuals/l2_norm', self.l2_norm['Residuals'])

            # Residuals:
            #   a) Mean Relative Error (MRE)
            #   b) Root Mean Square Error (RMSE)
            #   c) Mean Absolute Error (MAE)
            #   d) Gradient difference loss.
            tf.histogram_summary('residuals/mean_relative_error', self.relative_error_op)
            tf.histogram_summary('residuals/root_mean_square_error', self.rms_error_op)
            tf.histogram_summary('residuals/mean_absolute_error', self.mae_error_op)
            tf.histogram_summary('residuals/gradient_difference_loss', self.gradient_difference_loss)

            # Gradients updates for optimizing the cost function
            No = len(self.gradients)
            step = 1
            indices_vector = np.arange(0, No, step, dtype=int)
            for i in indices_vector:
                tf.histogram_summary('gradients_'+ str(No) + '/variable_' + str(i), self.gradients[i])

        print('[ RNN Regression ] Build RNN -> End')

    def inference(self):

        """
            Inference (your implementation).
            :return:
        """

        print("[ RNN Regression ] Inference.")

        return None

    def train(self):

        """
            Training (your implementation).
            :return:
        """

        print("[ RNN Regression ] Training.")

        return None
