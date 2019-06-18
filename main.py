from utils.run_utils import initialize, save_dict_as_json, get_logger
from train.training import train_and_eval
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

    normalize = False

    def loss_func(labels, predictions):
        return 1 - tf.image.ssim(labels, predictions, max_val=15.0)  # Sort of heuristic method...

    train_dataset = HDF5Sequence(data_dir=train_path, batch_size=FLAGS.batch_size, training=True, normalize=normalize)
    val_dataset = HDF5Sequence(data_dir=val_path, batch_size=FLAGS.batch_size, training=False, normalize=normalize)

    model = make_unet_model(scope='UNET', input_shape=(320, 320, 1))
    tf.keras.utils.plot_model(model, to_file=log_path / f'model_{run_number:02d}.png', show_shapes=True)
    model.summary()

    ckpt_path = Path(FLAGS.ckpt_dir) / run_name
    ckpt_path.mkdir(exist_ok=True)

    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)  # Can't change lr easily until TF2.0 update...
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

    if FLAGS.restore_dir:
        latest_checkpoint = tf.train.latest_checkpoint(FLAGS.restore_dir)
        checkpoint.restore(latest_checkpoint).assert_nontrivial_match().assert_existing_objects_matched()
        print(f'Restored model from {latest_checkpoint}')

    manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory=str(ckpt_path), max_to_keep=FLAGS.max_to_keep)
    writer = tf.contrib.summary.create_file_writer(logdir=str(log_path))  # Graph display not possible in eager...

    model_config = json.loads(model.to_json())
    # Save FLAGS to a json file next to the tensorboard data
    save_dict_as_json(dict_data=FLAGS.flag_values_dict(), log_dir=str(log_path), save_name=f'{run_name}_FLAGS')
    save_dict_as_json(dict_data=model_config, log_dir=str(log_path), save_name=f'{run_name}_config')

    with writer.as_default(), tf.contrib.summary.always_record_summaries():
        train_and_eval(model=model, optimizer=optimizer, manager=manager, train_dataset=train_dataset,
                       val_dataset=val_dataset, num_epochs=FLAGS.num_epochs, loss_func=loss_func,
                       save_best_only=FLAGS.save_best_only, use_train_metrics=True, use_val_metrics=True,
                       verbose=FLAGS.verbose, max_images=FLAGS.max_images)

    finish = int(time() - start)
    logger.info(f'Finished Training model. Time: {finish // 3600:02d}hrs {(finish // 60) % 60}min {finish % 60}s')
    return 0


if __name__ == '__main__':
    gpu = 1  # For switching between my 2 GPUs.
    gpu_kwargs = dict(allow_growth=True, per_process_gpu_memory_fraction=0.95, visible_device_list=str(gpu))
    config_kwargs = dict(allow_soft_placement=True, log_device_placement=False)

    gpu_options = tf.GPUOptions(**gpu_kwargs)
    config = tf.ConfigProto(gpu_options=gpu_options, **config_kwargs)

    tf.enable_eager_execution(config=config)
    tf.enable_resource_variables()
    tf.enable_v2_behavior()
    tf.enable_v2_tensorshape()

    FLAGS.set_default('num_epochs', 20)
    FLAGS.set_default('batch_size', 1)
    FLAGS.set_default('verbose', False)
    FLAGS.set_default('lr', 2E-4)
    FLAGS.set_default('save_best_only', True)
    FLAGS.set_default('max_to_keep', 10)
    FLAGS.set_default('max_images', 16)
    FLAGS.set_default('restore_dir', '/home/veritas/PycharmProjects/newFastMRI/checkpoints/Trial 04  2019-04-05 120005')

    tf.app.run(main)
