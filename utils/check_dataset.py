from utils.dataset import HDF5Sequence
import matplotlib.pyplot as plt
import numpy as np


val_dir = '/home/veritas/PycharmProjects/newFastMRI/data/multicoil_val'
dataset = HDF5Sequence(data_dir=val_dir, training=True, as_tensors=False, for_save=True)

sdx = 10
plt.gray()
for data, labels, *other in dataset:
    plt.figure(1)
    plt.title('data')
    plt.imshow(np.squeeze(data[sdx]))
    plt.colorbar()

    plt.figure(2)
    plt.title('labels')
    plt.imshow(np.squeeze(labels[sdx]))
    plt.colorbar()

    plt.show()

    for thing in other:
        print(thing[sdx])

    break


