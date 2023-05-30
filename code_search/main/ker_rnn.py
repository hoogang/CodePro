import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.contrib.rnn import  LSTMStateTuple
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging

def _linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(axis=1, values=args), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [output_size], initializer=tf.constant_initializer(bias_start))
    return res + bias_term


class OnLSTMCell(RNNCell):

    """Basic LSTM recurrent network cell.

      The implementation is based on: http://arxiv.org/abs/1409.2329.

      We add forget_bias (default: 1) to the biases of the forget gate in order to
      reduce the scale of forgetting in the beginning of the training.

      It does not allow cell clipping, a projection layer, and does not
      use peep-hole connections: it is the basic baseline.

      For advanced models, please use the full LSTMCell that follows.
      """

    def __init__(self, num_units, chunk_size,forget_bias=1.0,input_size=None,
                 state_is_tuple=True):

        """Initialize the basic LSTM cell.

            Args:
              num_units: int, The number of units in the LSTM cell.
              forget_bias: float, The bias added to forget gates (see above).
              input_size: Deprecated and unused.
              state_is_tuple: If True, accepted and returned states are 2-tuples of
                the `c_state` and `m_state`.  If False, they are concatenated
                along the column axis.  The latter behavior will soon be deprecated.
              activation: Activation function of the inner states.
            """
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self.chunk_size = chunk_size
        self.n_chunk = num_units // chunk_size

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def cumsum(self, x, direction):

        if direction == 'right':
            output = tf.cumsum(tf.nn.softmax(x, -1), -1)
            return output
        elif direction == 'left':
            output = 1 - tf.cumsum(tf.nn.softmax(x, -1), -1)
            return output

    def __call__(self, inputs, state,scope=None):

        with vs.variable_scope(scope or "onlstm_cell"):

            if self._state_is_tuple:
                c, h = state
            else:
                c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)

            concat = _linear([inputs, h], 4 * self._num_units + 2 * self.n_chunk, True, scope=scope)

            f_master_t = concat[:, :self.n_chunk]
            f_master_t = self.cumsum(tf.nn.softmax(f_master_t),'right')
            f_master_t = tf.expand_dims(f_master_t, 2)

            i_master_t = concat[:, self.n_chunk:2 * self.n_chunk]
            i_master_t = self.cumsum(tf.nn.softmax(i_master_t), 'left')
            i_master_t = tf.expand_dims(i_master_t, 2)
            concat = concat[:, 2 * self.n_chunk:]
            concat = tf.reshape(concat, [-1, self.n_chunk * 4, self.chunk_size])

            f_t = tf.nn.sigmoid(concat[:, :self.n_chunk])
            i_t = tf.nn.sigmoid(concat[:, self.n_chunk: 2 * self.n_chunk])
            o_t = tf.nn.sigmoid(concat[:, 2 * self.n_chunk: 3 * self.n_chunk])
            c_t_hat = tf.tanh(concat[:, 3 * self.n_chunk:])

            w_t = f_master_t * i_master_t

            new_c = w_t * (f_t * tf.reshape(c, [-1, self.n_chunk, self.chunk_size]) + i_t * c_t_hat) + \
                    (i_master_t - w_t) * c_t_hat + \
                    (f_master_t - w_t) * tf.reshape(c, [-1, self.n_chunk, self.chunk_size])
            new_h = tf.tanh(new_c) * o_t
            new_c = tf.reshape(new_c, [-1, self._num_units])
            new_h = tf.reshape(new_h, [-1, self._num_units])


            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                new_state = array_ops.concat([new_c, new_h], 1)

            return new_h, new_state