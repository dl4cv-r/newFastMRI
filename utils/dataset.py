import tensorflow as tf
from tensorflow.python.keras.utils import Sequence
from random import choice
from pathlib import Path
import numpy as np
import h5py
from math import ceil


# I need to test whether this system actually works without bugs. Check output images!
class HDF5Sequence(Sequence):  # Output must be numpy array not tensor if using multiprocessing.
    def __init__(self, data_dir, batch_size=16, training=True, as_tensors=False,
                 acc_fac=None, normalize=True, for_save=False, drop_last=False):

        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.training = training
        self.acc_fac = acc_fac
        self.as_tensors = as_tensors
        self.normalize = normalize
        self.for_save = for_save
        self.drop_last = drop_last

        if self.acc_fac is not None:  # Use both if self.acc_fac is None.
            assert self.acc_fac in (4, 8), 'Invalid acceleration factor'

        data_path = Path(self.data_dir)
        file_names = [str(h5) for h5 in data_path.glob('*.h5')]
        file_names.sort()

        if not file_names:  # If the list is empty for any reason
            raise OSError("Sorry! No files present in this directory.")

        print(f'Initializing {data_path.stem}. This might take a minute')
        slice_counts = [self.get_slice_number(file_name) for file_name in file_names]
        self.num_slices = sum(slice_counts)

        names_and_slices = list()
        if self.acc_fac is not None:
            for name, slice_num in zip(file_names, slice_counts):
                names_and_slices += [[name, s_idx, self.acc_fac] for s_idx in range(slice_num)]
        else:
            for name, slice_num in zip(file_names, slice_counts):
                names_and_slices += [[name, s_idx, choice((4, 8))] for s_idx in range(slice_num)]

        self.names_and_slices = names_and_slices
        assert self.num_slices == len(names_and_slices), 'Error in length'
        print(f'Finished {data_path.stem} initialization!')
        self.indices = np.random.permutation(self.num_slices) if self.training else np.arange(self.num_slices)

    def __len__(self):  # Always include last if like this.
        if self.drop_last:
            return self.num_slices // self.batch_size
        else:
            return int(ceil(self.num_slices / self.batch_size))

    @staticmethod
    def get_slice_number(file_name):
        with h5py.File(name=file_name, mode='r', swmr=True) as hf:
            return hf['1'].shape[0]

    @staticmethod
    def h5_slice_parse_fn(file_name, slice_num, acc_fac):
        with h5py.File(file_name, 'r', libver='latest', swmr=True) as hf:
            ds_slice_arr = np.asarray(hf[str(acc_fac)][slice_num])
            gt_slice_arr = np.asarray(hf['1'][slice_num])
            fat = hf.attrs['acquisition']  # Fat suppression ('CORPDFS_FBK', 'CORPD_FBK')
            if fat == 'CORPDFS_FBK':
                fat_supp = True
            elif fat == 'CORPD_FBK':
                fat_supp = False
            else:
                raise TypeError('Invalid fat suppression/acquisition type!')
        return ds_slice_arr, gt_slice_arr, fat_supp

    @staticmethod
    def tf_augment_data(data, labels):  # Need to test whether this works.
        lr_seed = np.random.randint(low=0, high=2 ** 30)
        ud_seed = np.random.randint(low=0, high=2 ** 30)
        data = tf.image.random_flip_left_right(data, seed=lr_seed)
        data = tf.image.random_flip_up_down(data, seed=ud_seed)
        labels = tf.image.random_flip_left_right(labels, seed=lr_seed)
        labels = tf.image.random_flip_up_down(labels, seed=ud_seed)
        return data, labels

    @staticmethod
    def np_augment_data(data, labels):
        ud_idxs = np.random.random(labels.shape[0]) < 0.5
        lr_idxs = np.random.random(labels.shape[0]) < 0.5
        data[ud_idxs] = np.flip(data[ud_idxs], axis=1)
        data[lr_idxs] = np.flip(data[lr_idxs], axis=2)
        labels[ud_idxs] = np.flip(labels[ud_idxs], axis=1)
        labels[lr_idxs] = np.flip(labels[lr_idxs], axis=2)
        return data, labels

    @staticmethod
    def tf_process_batch(ds_slices, gt_slices):  # No fat yet.
        data = tf.expand_dims(tf.convert_to_tensor(ds_slices), axis=3)
        data = tf.ensure_shape(data, shape=(None, 320, 320, 1))
        labels = tf.convert_to_tensor(tf.expand_dims(gt_slices, axis=3))
        labels = tf.ensure_shape(labels, shape=(None, 320, 320, 1))
        return data, labels

    @staticmethod
    def np_process_batch(ds_slices, gt_slices):  # No fat yet.
        data = np.expand_dims(ds_slices, axis=3)
        labels = np.expand_dims(gt_slices, axis=3)
        assert data.shape == labels.shape
        assert data.shape[1:] == (320, 320, 1)
        return data, labels

    @staticmethod
    def normalize_batched_slices(ds_slices, gt_slices):  # Same as paper.
        stddevs = np.std(ds_slices, axis=(1, 2), keepdims=True)
        means = np.mean(ds_slices, axis=(1, 2), keepdims=True)
        ds_slices -= means
        ds_slices /= stddevs
        ds_slices = np.clip(ds_slices, a_min=-6, a_max=6)
        gt_slices -= means
        gt_slices /= stddevs
        gt_slices = np.clip(gt_slices, a_min=-6, a_max=6)
        return ds_slices, gt_slices, stddevs, means

    @staticmethod
    def amplify_batched_slices(ds_slices, gt_slices):
        stddevs = np.std(ds_slices, axis=(1, 2), keepdims=True)
        means = np.mean(ds_slices, axis=(1, 2), keepdims=True)  # Kept for API consistency.
        ds_slices /= stddevs
        gt_slices /= stddevs
        return ds_slices, gt_slices, stddevs, means

    def __getitem__(self, item):  # item is an integer from range(len(Sequence))
        start = item * self.batch_size
        idxs = self.indices[start:start+self.batch_size]  # numpy automatically handles index overshooting.

        ds_slices = list()
        gt_slices = list()
        fat_supps = list()
        file_names = list()
        slice_numbers = list()
        acc_facs = list()

        for idx in idxs:
            file_name, slice_num, acc_fac = self.names_and_slices[idx]
            ds_slice, gt_slice, fat_supp = self.h5_slice_parse_fn(file_name, slice_num, acc_fac)
            ds_slices.append(ds_slice)
            gt_slices.append(gt_slice)
            fat_supps.append(fat_supp)

            if self.for_save:
                file_names.append(file_name)
                slice_numbers.append(slice_num)
                acc_facs.append(acc_fac)
        else:
            ds_slices = np.stack(ds_slices, axis=0)
            gt_slices = np.stack(gt_slices, axis=0)
            fat_supps = np.stack(fat_supps, axis=0)

        if self.normalize:
            ds_slices, gt_slices, stddevs, means = self.normalize_batched_slices(ds_slices, gt_slices)
        else:
            ds_slices, gt_slices, stddevs, means = self.amplify_batched_slices(ds_slices, gt_slices)

        if self.as_tensors:
            data, labels = self.tf_process_batch(ds_slices, gt_slices)
            if self.training:
                data, labels = self.tf_augment_data(data, labels)
        else:  # give as numpy
            data, labels = self.np_process_batch(ds_slices, gt_slices)
            if self.training:
                data, labels = self.np_augment_data(data, labels)

        if not self.for_save:
            return data, labels
        else:
            return data, stddevs, means, fat_supps, file_names, slice_numbers, acc_facs  # List type returns for some.

    def on_epoch_end(self):
        if self.training:
            self.indices = np.random.permutation(self.num_slices)


class HDF5TestSequence:
    def __init__(self, data_dir, batch_size=16, as_tensors=False, acc_fac=None, normalize=True):

        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.acc_fac = acc_fac
        self.as_tensors = as_tensors
        self.normalize = normalize

        if self.acc_fac is not None:  # Use both if self.acc_fac is None.
            assert self.acc_fac in (4, 8), 'Invalid acceleration factor'

        data_path = Path(self.data_dir)
        file_names = [str(h5) for h5 in data_path.glob('*.h5')]
        file_names.sort()

        if not file_names:  # If the list is empty for any reason
            raise OSError("Sorry! No files present in this directory.")

        print(f'Initializing {data_path.stem}. This might take a minute')
        slice_counts = [self.get_test_slice_number(file_name) for file_name in file_names]
        self.num_slices = sum(slice_counts)

        names_and_slices = list()
        for name, slice_num in zip(file_names, slice_counts):
            names_and_slices += [[name, s_idx] for s_idx in range(slice_num)]

        self.names_and_slices = names_and_slices
        assert self.num_slices == len(names_and_slices), 'Error in length'
        print(f'Finished {data_path.stem} test initialization!')
        self.indices = np.arange(self.num_slices)

    def __len__(self):  # Always include last if like this.
        return int(ceil(self.num_slices / self.batch_size))

    @staticmethod
    def get_test_slice_number(file_name):
        with h5py.File(name=file_name, mode='r', swmr=True) as hf:
            return hf['data'].shape[0]

    @staticmethod
    def h5_test_slice_parse_fn(file_name, slice_num):  # Haven't implemented acceleration filtering yet.
        with h5py.File(file_name, 'r', libver='latest', swmr=True) as hf:
            ds_slice_arr = np.asarray(hf['data'][slice_num])
            attrs = dict(hf.attrs)
            acc_fac = str(attrs['acceleration'])
            fat = attrs['acquisition']
            if fat == 'CORPDFS_FBK':
                fat_supp = True
            elif fat == 'CORPD_FBK':
                fat_supp = False
            else:
                raise TypeError('Invalid fat suppression/acquisition type!')
            return ds_slice_arr, acc_fac, fat_supp

    @staticmethod
    def tf_process_test_batch(ds_slices):  # No fat yet.
        data = tf.expand_dims(tf.convert_to_tensor(ds_slices), axis=3)
        data = tf.ensure_shape(data, shape=(None, 320, 320, 1))
        return data

    @staticmethod
    def np_process_test_batch(ds_slices):  # No fat yet.
        data = np.expand_dims(ds_slices, axis=3)
        assert data.shape[1:] == (320, 320, 1)
        return data

    @staticmethod
    def normalize_batched_test_slices(ds_slices):  # Same as paper.
        stddevs = np.std(ds_slices, axis=(1, 2), keepdims=True)
        means = np.mean(ds_slices, axis=(1, 2), keepdims=True)
        ds_slices -= means
        ds_slices /= stddevs
        ds_slices = np.clip(ds_slices, a_min=-6, a_max=6)
        return ds_slices, stddevs, means

    @staticmethod
    def amplify_batched_test_slices(ds_slices):
        stddevs = np.std(ds_slices, axis=(1, 2), keepdims=True)
        means = np.mean(ds_slices, axis=(1, 2), keepdims=True)  # Kept for API consistency.
        ds_slices /= stddevs
        return ds_slices, stddevs, means

    def __getitem__(self, item):  # item is an integer from range(len(Sequence))
        start = item * self.batch_size
        idxs = self.indices[start:start+self.batch_size]  # numpy automatically handles index overshooting.

        ds_slices = list()
        fat_supps = list()
        file_names = list()
        slice_numbers = list()
        acc_facs = list()

        for idx in idxs:
            file_name, slice_num = self.names_and_slices[idx]
            ds_slice, acc_fac, fat_supp = self.h5_test_slice_parse_fn(file_name, slice_num)
            ds_slices.append(ds_slice)
            fat_supps.append(fat_supp)
            file_names.append(file_name)
            slice_numbers.append(slice_num)
            acc_facs.append(acc_fac)
        else:
            try:
                ds_slices = np.stack(ds_slices, axis=0)
                fat_supps = np.stack(fat_supps, axis=0)
            except ValueError as e:
                print(len(ds_slices), len(fat_supps), e)

        if self.normalize:
            ds_slices, stddevs, means = self.normalize_batched_test_slices(ds_slices)
        else:
            ds_slices, stddevs, means = self.amplify_batched_test_slices(ds_slices)

        if self.as_tensors:
            data = self.tf_process_test_batch(ds_slices)
        else:  # give as numpy
            data = self.np_process_test_batch(ds_slices)

        return data, stddevs, means, fat_supps, file_names, slice_numbers, acc_facs  # List type returns for some.
