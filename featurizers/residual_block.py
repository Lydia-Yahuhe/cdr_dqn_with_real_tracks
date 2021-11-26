import tensorflow as tf


def residual_block(inputs, channels_in, channels_out, cardinality, is_training, strides=(1, 1), project_shortcut=False):
    """Implements a layer of ResNeXt.

    Based in https://blog.waya.ai/deep-residual-learning-9610bb62c355.

    Args:
        inputs: Input for the block.
        channels_in: Number of channels of radarTracks for the convolutional group.
        channels_out: Number of chhannels generated as output.
        cardinality: Number of convolution groups. Must be divisible by channels_in
        is_training: Placeholder that indicates if the model is being trained
        strides: Strides.
        project_shortcut: Indicates whether the input should be projected to match the output's dimensions.

    Returns:
        A tensor with shape (-1, width, height, channels_out).
    
    """

    shortcut = inputs

    output = tf.layers.conv2d(inputs, filters=channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')
    output = _add_common_layers(output, is_training)

    output = _grouped_convolution(output, cardinality, channels_in, strides)
    output = _add_common_layers(output, is_training)

    output = tf.layers.conv2d(output, filters=channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')
    output = tf.layers.batch_normalization(output, training=is_training)

    if project_shortcut or strides != (1, 1):
        # When the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions).
        shortcut = tf.layers.conv2d(shortcut, filters=channels_out, kernel_size=(1, 1), strides=strides, padding='same')
        shortcut = tf.layers.batch_normalization(shortcut, training=is_training)

    output = tf.nn.relu(tf.add(shortcut, output))

    return output


def _grouped_convolution(inputs, cardinality, channels, strides):
    assert not channels % cardinality

    channels_per_group = channels // cardinality

    # In a grouped convolution layer, input and output channels are divided into `cardinality` groups,
    # and convolutions are separately performed within each group
    groups = []
    for j in range(cardinality):
        group_input = inputs[:, :, :, j * channels_per_group: j * channels_per_group + channels_per_group]
        groups.append(tf.layers.conv2d(group_input, filters=channels_per_group, kernel_size=(3, 3), strides=strides,
                                       padding='same'))

    output = tf.concat(groups, -1)
    return output


def _add_common_layers(inputs, is_training):
    output = tf.layers.batch_normalization(inputs, training=is_training)
    output = tf.nn.relu(output)
    return output
