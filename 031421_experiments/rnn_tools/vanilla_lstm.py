import tensorflow as tf


class VanillaLSTMCell(tf.nn.rnn_cell.RNNCell):

    """

    Vanilla LSTM Cell. There are two variants:

        - option = "Complete", corresponds to the full LSTM network implementation.
        - option = "CIFG", corresponds to the LSTM network with coupled input-forget gates (CIFG).

    """

    def __init__(self, input_size, num_blocks, option='Complete'):
        self._num_blocks = num_blocks
        self._input_size = input_size
        self.option = option

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._num_blocks

    @property
    def state_size(self):
        return 2 * self._num_blocks

    def __call__(self, inputs, state, scope=None):

        with tf.variable_scope(scope or type(self).__name__):
            initializer = tf.random_uniform_initializer(-0.1, 0.1)

            def get_variable(name, shape):
                return tf.get_variable(name, shape, initializer=initializer, dtype=inputs.dtype)

            c_prev, y_prev = tf.split(1, 2, state)

            W_z = get_variable("W_z", [self.input_size, self._num_blocks])
            W_i = get_variable("W_i", [self.input_size, self._num_blocks])

            if self.option == 'Complete':
                W_f = get_variable("W_f", [self.input_size, self._num_blocks])
            elif self.option == 'CIFG':
                W_f = None

            W_o = get_variable("W_o", [self.input_size, self._num_blocks])

            R_z = get_variable("R_z", [self._num_blocks, self._num_blocks])
            R_i = get_variable("R_i", [self._num_blocks, self._num_blocks])

            if self.option == 'Complete':
                R_f = get_variable("R_f", [self._num_blocks, self._num_blocks])
            elif self.option == 'CIFG':
                R_f = None

            R_o = get_variable("R_o", [self._num_blocks, self._num_blocks])

            b_z = get_variable("b_z", [1, self._num_blocks])
            b_i = get_variable("b_i", [1, self._num_blocks])

            if self.option == 'Complete':
                b_f = get_variable("b_f", [1, self._num_blocks])
            elif self.option == 'CIFG':
                b_f = None

            b_o = get_variable("b_o", [1, self._num_blocks])

            p_i = get_variable("p_i", [self._num_blocks])

            if self.option == 'Complete':
                p_f = get_variable("p_f", [self._num_blocks])
            elif self.option == 'CIFG':
                p_f = None

            p_o = get_variable("p_o", [self._num_blocks])

            g = h = tf.tanh

            z = g(tf.matmul(inputs, W_z) + tf.matmul(y_prev, R_z) + b_z)
            i = tf.sigmoid(tf.matmul(inputs, W_i) + tf.matmul(y_prev, R_i) + tf.mul(c_prev, p_i) + b_i)

            if self.option == 'Complete':
                f = tf.sigmoid(tf.matmul(inputs, W_f) + tf.matmul(y_prev, R_f) + tf.mul(c_prev, p_f) + b_f)
            elif self.option == 'CIFG':
                f = 1 - i

            c = tf.mul(i, z) + tf.mul(f, c_prev)
            o = tf.sigmoid(tf.matmul(inputs, W_o) + tf.matmul(y_prev, R_o) + tf.mul(c, p_o) + b_o)
            y = tf.mul(h(c), o)

            return y, tf.concat(1, [c, y])
