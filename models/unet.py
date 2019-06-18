from tensorflow.python.keras.layers import *
from tensorflow.python.keras.regularizers import l1_l2
from models.custom_layers import InstanceNormalization
from tensorflow.python.keras.models import Model
import tensorflow as tf
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('init_chan', default=32, help='Number of channels in the first convolution layer')
flags.DEFINE_integer('num_pools', default=4, help='Number of pooling layers')
flags.DEFINE_float('l1', default=0., help='L1 regularization factor')
flags.DEFINE_float('l2', default=0., help='L2 regularization factor')


def _conv2d_3(inputs, filter_num, name=None):
    conv = Conv2D(filters=filter_num, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='he_uniform',
                  padding='same', kernel_regularizer=l1_l2(l1=FLAGS.l1, l2=FLAGS.l2), name=name)(inputs)
    in_norm = InstanceNormalization()(conv)
    relu = ReLU()(in_norm)
    return relu


def _conv_single(inputs, filter_num, name=None):
    conv = Conv2D(filters=filter_num, kernel_size=(1, 1), strides=(1, 1), kernel_initializer='he_uniform',
                  padding='valid', kernel_regularizer=l1_l2(l1=FLAGS.l1, l2=FLAGS.l2), name=name)(inputs)
    return conv


def unet_input(input_shape=(320, 320, 1)):
    with tf.name_scope('Inputs'):
        inputs = Input(shape=input_shape, name='input_data')
    return inputs


def _increase_size(inputs, factor=2, name=None):
    orig_shape = inputs.shape.as_list()
    new_shape = (orig_shape[1] * factor, orig_shape[2] * factor)
    increased = Lambda(lambda inp: tf.image.resize_bilinear(
        images=inp, size=new_shape, align_corners=False), name=name)(inputs)
    return increased


def unet(input_data):

    filter_num = FLAGS.init_chan
    conv_list = list()  # List for storing pooling layers
    depth = FLAGS.num_pools + 1  # Depth of the model
    print('Building UNET')  # Checking whether this is done only once or many times.

    s = filter_num
    checks = [s, s*2, s*4, s*8, s*8, s*4, s*2, s, s]  # A hack. Only for 4 pool models.

    bl_name = 'Block_01'
    with tf.name_scope(bl_name):
        conv1 = _conv2d_3(inputs=input_data, filter_num=filter_num, name=bl_name + '_Conv_1')
        conv2 = _conv2d_3(inputs=conv1, filter_num=filter_num, name=bl_name + '_Conv_2')
        pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv2)
        conv_list.append(conv2)  # Saving conv layers for later use

    for block_num in range(2, depth):
        filter_num *= 2
        bl_name = f'Block_{block_num:02d}'
        assert filter_num == checks[block_num-1], f'{bl_name}: Incorrect filter number'
        with tf.name_scope(bl_name):
            conv1 = _conv2d_3(inputs=pool, filter_num=filter_num, name=bl_name + '_Conv_1')
            conv2 = _conv2d_3(inputs=conv1, filter_num=filter_num, name=bl_name + '_Conv_2')
            pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv2)
            conv_list.append(conv2)

    # The Bottom Block.  # See original benchmark code to see why filter numbers are what they are.
    bl_name = f'Block_{depth:02d}'
    assert filter_num == checks[depth-1], 'Incorrect filter number'
    with tf.name_scope(bl_name):
        conv1 = _conv2d_3(inputs=pool, filter_num=filter_num, name=bl_name + '_Conv_1')
        conv2 = _conv2d_3(inputs=conv1, filter_num=filter_num, name=bl_name + '_Conv_2')

    # Going up.
    for block_num in range(depth + 1, 2 * depth - 1):  # Last up channel is not included.
        filter_num //= 2
        bl_name = f'Block_{block_num:02d}'
        assert filter_num == checks[block_num-1], 'Incorrect filter number'
        with tf.name_scope(bl_name):
            up = _increase_size(inputs=conv2, name=f'{bl_name}_UP')
            merged = Concatenate(name=f'{bl_name}_Merge')([up, conv_list.pop()])
            assert up.shape[-1] * 2 == merged.shape[-1]  # Checking.
            conv1 = _conv2d_3(inputs=merged, filter_num=filter_num, name=bl_name + '_Conv_1')
            conv2 = _conv2d_3(inputs=conv1, filter_num=filter_num, name=bl_name + '_Conv_2')

    # Last block
    bl_name = f'Block_{2*depth-1:02d}'
    assert filter_num == checks[2*FLAGS.num_pools], 'Incorrect filter number'

    with tf.name_scope(bl_name):
        up = _increase_size(inputs=conv2, name=f'{bl_name}_UP')
        merged = Concatenate(name=f'{bl_name}_Merge')([up, conv_list.pop()])
        assert up.shape[-1] * 2 == merged.shape[-1]  # Checking.
        conv1 = _conv2d_3(inputs=merged, filter_num=filter_num, name=bl_name + '_Conv_1')
        conv2 = _conv2d_3(inputs=conv1, filter_num=filter_num, name=bl_name + '_Conv_2')

    assert filter_num == FLAGS.init_chan, 'Incorrect filter number. Check indexing!'

    with tf.name_scope('Outputs'):
        down = _conv_single(inputs=conv2, filter_num=filter_num//2, name='Down_1')
        down = _conv_single(inputs=down, filter_num=1, name='Down_2')
        outputs = _conv_single(inputs=down, filter_num=1, name='Outputs')

    return outputs


def make_unet_model(scope, input_shape):
    with tf.name_scope(scope):
        input_data = unet_input(input_shape=input_shape)
        reconstruction = unet(input_data)
        model = Model(inputs=input_data, outputs=reconstruction)
    return model
