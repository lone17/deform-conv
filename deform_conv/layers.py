from __future__ import absolute_import, division


import tensorflow as tf
from keras.layers import Conv2D
from keras.initializers import RandomNormal
from deform_conv.deform_conv import tf_batch_map_offsets


class ConvOffset2D(Conv2D):
    """ConvOffset2D"""

    def __init__(self, filters, channel_wise=True, init_normal_stddev=0.01, 
                 **kwargs):
        """Init"""

        self.filters = filters
        self.channel_wise = channel_wise
        if self.channel_wise:
            super(ConvOffset2D, self).__init__(
                self.filters * 2, (3, 3), padding='same', use_bias=False,
                # TODO gradients are near zero if init is zeros
                kernel_initializer='zeros',
                # kernel_initializer=RandomNormal(0, init_normal_stddev),
                **kwargs
            )
        else:
            super(ConvOffset2D, self).__init__(
                2, (3, 3), padding='same', use_bias=False,
                # TODO gradients are near zero if init is zeros
                kernel_initializer='zeros',
                # kernel_initializer=RandomNormal(0, init_normal_stddev),
                **kwargs
            )

    def call(self, x):
        # TODO offsets probably have no nonlinearity?
        x_shape = tf.shape(x)
        offsets = super(ConvOffset2D, self).call(x)
        
        if self.channel_wise:
            offsets = self._to_bc_h_w_2(offsets, x_shape)
        else:
            offsets = tf.tile(offsets, [x_shape[-1], 1, 1, 1])
        x = self._to_bc_h_w(x, x_shape)
        x_offset = tf_batch_map_offsets(x, offsets)
        x_offset = self._to_b_h_w_c(x_offset, x_shape)
        return x_offset

    def compute_output_shape(self, input_shape):
        return input_shape

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, h, w, 2c) -> (b*c, h, w, 2)"""
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.reshape(x, (-1, x_shape[1], x_shape[2], 2))
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, h, w, c) -> (b*c, h, w)"""
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.reshape(x, (-1, x_shape[1], x_shape[2]))
        return x

    @staticmethod
    def _to_b_h_w_c(x, x_shape):
        """(b*c, h, w) -> (b, h, w, c)"""
        x = tf.reshape(
            x, (-1, x_shape[3], x_shape[1], x_shape[2])
        )
        x = tf.transpose(x, [0, 2, 3, 1])
        return x
