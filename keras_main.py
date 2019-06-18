from utils.run_utils import initialize, save_dict_as_json, get_logger
from train.metrics import batch_ssim, batch_psnr, batch_msssim, batch_nmse
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
from models.unet import make_unet_model
from utils.dataset import HDF5Sequence

import tensorflow as tf
from absl import flags
from pathlib import Path
from time import time
import json


FLAGS = flags.FLAGS

flags.DEFINE_enum('challenge', default='multicoil', enum_values=['singlecoil', 'multicoil'], help='Challenge type')
flags.DEFINE_string('data_dir', default='./data', help='Path to data')
flags.DEFINE_integer('num_epochs', default=2, help='Number of epochs')
flags.DEFINE_string('ckpt_dir', default='./checkpoints', help='Directory to save checkpoints')
flags.DEFINE_string('log_dir', default='./logs', help='Directory to save log files')
flags.DEFINE_float('lr', default=1E-3, help='Learning Rate')
flags.DEFINE_bool('verbose', default=True, help='Whether to print out statistics for each step or just for each epoch')
flags.DEFINE_integer('batch_size', default=16, help='Mini-batch size')
flags.DEFINE_string('restore_dir', default='', help='Directory to folder with the checkpoints to be restored from')
flags.DEFINE_bool('save_best_only', default=True, help='Whether to checkpoint files only when they have improved')
flags.DEFINE_integer('max_images', default=0, help='Maximum number of images to display on Tensorboard.')
flags.DEFINE_integer('max_to_keep', default=5, help='Maximum number of checkpoints to keep in training.')


def main(argv):  # argv are non-flag parameters
    del argv
    start = time()
    tf.print('Tensorflow Engaged')
    run_number, run_name = initialize(FLAGS.ckpt_dir)

    log_path = Path(FLAGS.log_dir)
    log_path.mkdir(exist_ok=True)
    log_path = log_path / run_name
    log_path.mkdir(exist_ok=False)
    logger = get_logger(__name__)

    data_path = Path(FLAGS.data_dir)
    train_path = data_path / f'{FLAGS.challenge}_train'
    val_path = data_path / f'{FLAGS.challenge}_val'

    train_dataset = HDF5Sequence(data_dir=train_path, batch_size=FLAGS.batch_size, training=True, as_tensors=False)
    val_dataset = HDF5Sequence(data_dir=val_path, batch_size=FLAGS.batch_size, training=False, as_tensors=False)

    model = make_unet_model(scope='UNET', input_shape=(320, 320, 1))
    # multi_model = tf.keras.utils.multi_gpu_model(model, gpus=2)
    multi_model = model

    tf.keras.utils.plot_model(model, to_file=log_path / f'model_{run_number:02d}.png', show_shapes=True)
    model.summary()

    ckpt_path = Path(FLAGS.ckpt_dir) / run_name
    ckpt_path.mkdir(exist_ok=True)
    ckpt_path = ckpt_path / '{epoch:04d}.ckpt'
    ckpt_path = str(ckpt_path)

    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.lr)

    checkpoint = ModelCheckpoint(filepath=ckpt_path, verbose=1, save_best_only=True)
    visualizer = TensorBoard(log_dir=log_path, batch_size=FLAGS.batch_size)

    model_config = json.loads(model.to_json())
    # Save FLAGS to a json file next to the tensorboard data
    save_dict_as_json(dict_data=FLAGS.flag_values_dict(), log_dir=str(log_path), save_name=f'{run_name}_FLAGS')
    save_dict_as_json(dict_data=model_config, log_dir=str(log_path), save_name=f'{run_name}_config')

    multi_model.compile(optimizer=optimizer, loss='mae', metrics=[batch_ssim, batch_psnr, batch_msssim, batch_nmse])
    multi_model.fit_generator(train_dataset, epochs=FLAGS.num_epochs, verbose=1, callbacks=[checkpoint, visualizer],
                              validation_data=val_dataset, workers=4, use_multiprocessing=True)

    finish = int(time() - start)
    logger.info(f'Finished Training model. Time: {finish // 3600:02d}hrs {(finish // 60) % 60}min {finish % 60}s')
    return 0


if __name__ == '__main__':
    gpu = 1  # For switching between my 2 GPUs.
    gpu_kwargs = dict(allow_growth=True, per_process_gpu_memory_fraction=0.99, visible_device_list=str(gpu))
    config_kwargs = dict(allow_soft_placement=True, log_device_placement=False)

    gpu_options = tf.GPUOptions(**gpu_kwargs)
    config = tf.ConfigProto(gpu_options=gpu_options, **config_kwargs)
    #
    # tf.enable_eager_execution(config=config)
    # tf.enable_resource_variables()
    # tf.enable_v2_behavior()
    # tf.enable_v2_tensorshape()

    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    FLAGS.set_default('num_epochs', 20)
    FLAGS.set_default('batch_size', 16)
    FLAGS.set_default('verbose', False)
    FLAGS.set_default('lr', 1E-3)
    FLAGS.set_default('save_best_only', True)
    FLAGS.set_default('restore_dir', '')

    tf.app.run(main)
    sess.close()
