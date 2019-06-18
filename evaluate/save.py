import tensorflow as tf
from pathlib import Path
from collections import defaultdict
from models.unet import make_unet_model
from utils.dataset import HDF5Sequence, HDF5TestSequence
import numpy as np
import h5py


def fetch_model():  # So I only change these parts instead of all the code.
    return make_unet_model(scope='Restored_Model', input_shape=(320, 320, 1))


def save_reconstructions(reconstructions, save_dir):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.

    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        save_dir (str): Path to the output directory where the reconstructions
            should be saved.
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    print('Starting saving')
    for file_name, recons in reconstructions.items():
        name = save_path / file_name
        print('Saving ', name)
        with h5py.File(name, 'x', libver='latest') as f:
            f.create_dataset('reconstruction', data=recons, chunks=(1, 320, 320), compression='lzf', shuffle=True)


def restore_and_run_model(restore_dir, data_dir, batch_size=16, acc_fac=None, normalize=True, val_set=True):

    # Two factors which keep changing all the time.
    model = fetch_model()

    if val_set:
        dataset = HDF5Sequence(data_dir=data_dir, batch_size=batch_size, training=False, as_tensors=False,
                               acc_fac=acc_fac, normalize=normalize, for_save=True)
    else:
        dataset = HDF5TestSequence(data_dir=data_dir, batch_size=batch_size,
                                   as_tensors=False, acc_fac=acc_fac, normalize=normalize)

    # Same from here on.
    checkpoint = tf.train.Checkpoint(model=model)

    print(len(dataset))
    print(dataset.num_slices)

    latest_checkpoint = tf.train.latest_checkpoint(restore_dir)
    checkpoint.restore(latest_checkpoint).assert_existing_objects_matched()
    print(f'Restored model from {latest_checkpoint}')

    reconstructions = defaultdict(list)

    for count, (data, stddevs, means, fat_supps, file_names, slice_numbers, acc_facs) in enumerate(dataset):
        recons = model(inputs=data, training=False)
        recons = np.squeeze(recons.numpy())

        if normalize:
            recons = recons * stddevs + means
        else:
            recons = recons * stddevs

        print('Recon ', file_names[0].split('/')[-1])
        for idx in range(recons.shape[0]):
            file_name = file_names[idx].split('/')[-1]
            reconstructions[file_name].append((slice_numbers[idx], recons[idx]))

    recon_dict = dict()
    for file_name, slice_predictions in reconstructions.items():
        print('Reformatting ', file_name)
        recon_dict[file_name] = np.stack([pred for _, pred in sorted(slice_predictions)], axis=0)

    print('Finished making recons')
    return recon_dict


def main(argv):
    del argv

    validation = False
    restore_dir = '/home/veritas/PycharmProjects/newFastMRI/checkpoints/Trial 05  2019-04-05 232636'
    normalize = True
    save_dir = './recons/' + 'adam_recon_test_normalized_ssim'

    # validation = False
    # restore_dir = '/home/veritas/PycharmProjects/newFastMRI/checkpoints/Trial 06  2019-04-05 232718'
    # normalize = False
    # save_dir = './recons/' + 'adam_recon_test_amplified_ssim'

    if validation:
        data_dir = '/home/veritas/PycharmProjects/newFastMRI/data/multicoil_val'
    else:
        data_dir = '/home/veritas/PycharmProjects/newFastMRI/data/multicoil_test'

    reconstructions = restore_and_run_model(
        restore_dir=restore_dir, data_dir=data_dir, batch_size=11, normalize=normalize, val_set=validation)

    save_reconstructions(reconstructions=reconstructions, save_dir=save_dir)

    return 0


if __name__ == '__main__':
    gpu = 1  # For switching between my 2 GPUs.
    gpu_kwargs = dict(allow_growth=True, per_process_gpu_memory_fraction=0.90, visible_device_list=str(gpu))
    config_kwargs = dict(allow_soft_placement=True, log_device_placement=False)

    gpu_options = tf.GPUOptions(**gpu_kwargs)
    config = tf.ConfigProto(gpu_options=gpu_options, **config_kwargs)

    tf.enable_eager_execution(config=config)
    tf.enable_resource_variables()
    tf.enable_v2_behavior()
    tf.enable_v2_tensorshape()

    tf.app.run(main)
