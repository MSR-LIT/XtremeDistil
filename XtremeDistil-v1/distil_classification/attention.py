# 2020 Microsoft Research, Subhabrata Mukherjee
# Code for https://aka.ms/XtremeDistil

from keras import backend as K
from keras import regularizers, constraints, initializers, activations
from keras.engine import InputSpec
from keras.engine import Layer


def time_distributed_dense(x, w, b=None, dropout=None,
                           input_dim=None, output_dim=None,
                           timesteps=None, training=None):
    """Apply `y . w + b` for every temporal slice y of x.
    # Arguments
        x: input tensor.
        w: weight matrix.
        b: optional bias vector.
        dropout: whether to apply dropout (same dropout mask
            for every temporal slice of the input).
        input_dim: integer; optional dimensionality of the input.
        output_dim: integer; optional dimensionality of the output.
        timesteps: integer; optional number of timesteps.
        training: training phase tensor or boolean.
    # Returns
        Output tensor.
    """
    if not input_dim:
        input_dim = K.shape(x)[2]
    if not timesteps:
        timesteps = K.shape(x)[1]
    if not output_dim:
        output_dim = K.shape(w)[1]

    if dropout is not None and 0. < dropout < 1.:
        # apply the same dropout pattern at every timestep
        ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
        dropout_matrix = K.dropout(ones, dropout)
        expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
        x = K.in_train_phase(x * expanded_dropout_matrix, x, training=training)

    # collapse time dimension and batch dimension together
    x = K.reshape(x, (-1, input_dim))
    x = K.dot(x, w)
    if b is not None:
        x = K.bias_add(x, b)
    # reshape to 3D tensor
    if K.backend() == 'tensorflow':
        x = K.reshape(x, K.stack([-1, timesteps, output_dim]))
        x.set_shape([None, None, output_dim])
    else:
        x = K.reshape(x, (-1, timesteps, output_dim))
    return x


class SelfAttention(Layer):
    def __init__(self,
                 units,
                 activation='tanh',
                 name='SelfAttention',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.name = name
        self.supports_masking = True

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        self.batch_size, self.timesteps, self.input_dim = input_shape

        self.V_a = self.add_weight(shape=(self.units,),
                                   name='V_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.U_a = self.add_weight(shape=(self.input_dim, self.units),
                                   name='U_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.b_u = self.add_weight(shape=(self.units,),
                                   name='b_a',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        self.W_a = self.add_weight(shape=(self.input_dim, self.units),
                                   name='W_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)

        self.input_spec = [InputSpec(shape=(self.batch_size, self.timesteps, self.input_dim))]
        self.built = True

    # def compute_mask(self, input, mask=None):
    #    if mask is not None:
    #        return K.any(mask, axis=1)
    #    return mask

    def call(self, x, mask=None):
        if mask is not None:
            mask = K.expand_dims(K.cast(mask, K.floatx()))

        self.x_seq = x
        self._uxpb = time_distributed_dense(x=self.x_seq, w=self.U_a, b=self.b_u,
                                            input_dim=self.input_dim,
                                            timesteps=self.timesteps,
                                            output_dim=self.units)
        self.new_words = []
        self.ats = []

        for i in range(self.timesteps):
            word = K.repeat(self.x_seq[:, i, :], self.timesteps)
            word_tr = K.dot(word, self.W_a)
            et = K.dot(activations.tanh(word_tr + self._uxpb), K.expand_dims(self.V_a))
            at = K.exp(et)
            if mask is not None:
                at = at * mask
            at_sum = K.sum(at, axis=1)
            at_sum_repeated = K.repeat(at_sum, self.timesteps)
            at /= at_sum_repeated
            self.ats.append(at)
            context = K.squeeze(K.batch_dot(at, self.x_seq, axes=1), axis=1)
            self.new_words.append(context)
        self.res = K.stack(self.new_words, axis=1)
        return self.res

    def compute_output_shape(self, input_shape):
        """
            For Keras internal compatability checking
        """
        return (None, self.timesteps, self.input_dim)

    def get_config(self):
        """
            For rebuilding models on load time.
        """
        config = {
            'units': self.units,
        }
        base_config = super(SelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
