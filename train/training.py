import tensorflow as tf
import tensorflow.contrib.eager as tfe
from train.metrics import batch_ssim, batch_psnr, batch_msssim, batch_nmse
from time import time
from utils.run_utils import get_logger
from absl import flags

FLAGS = flags.FLAGS


"""
I have designed the code to work with tf keras sequences, not tf dataset.
I decided on this because tf dataset cannot cope with HDF5 files, which support single processes only (for most builds)
Please note that this may cause issues if this code is used for anything else, 
since keras sequences are not frequently used.
"""


@tfe.defun
def train_step(model, optimizer, loss_func, data, labels):
    data = tf.convert_to_tensor(data)  # Converting to tensor should be done inside defun to save, etc.
    labels = tf.convert_to_tensor(labels)

    with tf.GradientTape() as tape:
        recon = model(inputs=data, training=True)
        step_loss = loss_func(labels=labels, predictions=recon)

    grads = tape.gradient(step_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step=tf.train.get_global_step())
    return step_loss, recon


@tfe.defun
def val_step(model, loss_func, data, labels):
    data = tf.convert_to_tensor(data)
    labels = tf.convert_to_tensor(labels)

    recon = model(inputs=data, training=False)
    step_loss = loss_func(labels=labels, predictions=recon)  # Fixed loss function for now.
    return step_loss, recon


@tfe.defun
def make_view_images(labels, recons):
    max_val = tf.reduce_max(labels, axis=(1, 2, 3), keepdims=True)
    min_val = tf.reduce_min(labels, axis=(1, 2, 3), keepdims=True)
    val_range = 1 / (max_val - min_val)
    view_recons = (recons - min_val) * val_range
    view_labels = (labels - min_val) * val_range
    return view_labels, view_recons


@tfe.defun
def get_step_metrics_from_imgs(labels, recons):
    view_labels, view_recons = make_view_images(labels, recons)
    return get_step_metrics(view_labels, view_recons)


@tfe.defun
def get_step_metrics(view_labels, view_recons):
    step_msssim = batch_msssim(view_labels, view_recons)
    step_ssim = batch_ssim(view_labels, view_recons)
    step_psnr = batch_psnr(view_labels, view_recons)
    step_nmse = batch_nmse(view_labels, view_recons)
    return step_msssim, step_ssim, step_psnr, step_nmse


# The 2 functions here must be modified to get more or fewer metrics. The rest has been taken care of.
def create_epoch_metrics(epoch):
    epoch_msssim = tfe.metrics.Mean(name=f'val_msssim{epoch:02d}')
    epoch_ssim = tfe.metrics.Mean(name=f'val_ssim{epoch:02d}')
    epoch_psnr = tfe.metrics.Mean(name=f'val_psnr{epoch:02d}')
    epoch_nmse = tfe.metrics.Mean(name=f'val_nmse{epoch:02d}')
    return epoch_msssim, epoch_ssim, epoch_psnr, epoch_nmse


def train_epoch(model, optimizer, dataset, loss_func, epoch, use_metrics=True, verbose=True):

    epoch_loss = tfe.metrics.Mean(name=f'train_loss_{epoch:02d}')
    epoch_metrics = create_epoch_metrics(epoch) if use_metrics else tuple()
    num_metrics = len(epoch_metrics)
    step_metrics = tuple()

    for step, (data, labels) in enumerate(dataset, start=1):
        step_loss, recons = train_step(model=model, optimizer=optimizer, loss_func=loss_func, data=data, labels=labels)
        epoch_loss(step_loss)

        if use_metrics:
            step_metrics = get_step_metrics_from_imgs(labels=labels, recons=recons)
            for mdx in range(num_metrics):
                epoch_metrics[mdx](step_metrics[mdx])

        if verbose:
            tf.print(f'Epoch {epoch:03d} Step {step:04d} loss: ', step_loss)
            if use_metrics:
                for idx, step_metric in enumerate(step_metrics, start=1):
                    tf.print(f'Metric {idx}: ', step_metric)

    epoch_loss = epoch_loss.result(write_summary=False)

    if use_metrics:
        epoch_metrics = [epoch_metric.result(write_summary=False) for epoch_metric in epoch_metrics]

    return epoch_loss, epoch_metrics


def val_epoch(model, dataset, loss_func, epoch, max_images=36, verbose=True, use_metrics=True):

    epoch_loss = tfe.metrics.Mean(name=f'val_loss_{epoch:02d}')
    epoch_metrics = create_epoch_metrics(epoch) if use_metrics else tuple()
    num_metrics = len(epoch_metrics)
    step_metrics = tuple()

    interval = len(dataset) // max_images if max_images > 0 else 0

    for step, (data, labels) in enumerate(dataset, start=1):
        step_loss, recons = val_step(model=model, loss_func=loss_func, data=data, labels=labels)
        epoch_loss(step_loss)

        if use_metrics:
            step_metrics = get_step_metrics_from_imgs(labels=labels, recons=recons)
            for mdx in range(num_metrics):
                epoch_metrics[mdx](step_metrics[mdx])

        if verbose:
            tf.print(f'Epoch {epoch:03d} Step {step:04d} loss: ', step_loss)
            if use_metrics:
                for idx, step_metric in enumerate(step_metrics, start=1):
                    tf.print(f'Metric {idx}: ', step_metric)

        if max_images > 0:
            if (step-1) // interval == 0:
                tf.contrib.summary.image(name='val_recon_imgs', tensor=recons, max_images=max_images, step=epoch)
                tf.contrib.summary.image(name='val_label_imgs', tensor=labels, max_images=max_images, step=epoch)
                tf.contrib.summary.image(name='val_diff_imgs', tensor=recons-labels, max_images=max_images, step=epoch)

    epoch_loss = epoch_loss.result(write_summary=False)

    if use_metrics:
        epoch_metrics = [epoch_metric.result(write_summary=False) for epoch_metric in epoch_metrics]

    return epoch_loss, epoch_metrics


def train_and_eval(model, optimizer, manager, train_dataset, val_dataset, num_epochs, loss_func, save_best_only=True,
                   use_train_metrics=False, use_val_metrics=True, verbose=True, max_images=36):

    logger = get_logger(__name__)

    prev_loss = 2**30

    for epoch in range(1, num_epochs + 1):
        # Training
        tic = time()
        logger.info(f'\nStarting Epoch {epoch:03d} Training')
        train_epoch_loss, train_epoch_metrics = \
            train_epoch(model=model, optimizer=optimizer, dataset=train_dataset, loss_func=loss_func,
                        epoch=epoch, use_metrics=use_train_metrics, verbose=verbose)

        # After Epoch training is over.
        toc = int(time() - tic)
        logger.info(f'Epoch {epoch:03d} Training Finished. Time: {toc // 60}min {toc % 60}s.')

        tf.contrib.summary.scalar(name='train_epoch_loss', tensor=train_epoch_loss, step=epoch)
        logger.info(f'Epoch Training loss: {float(train_epoch_loss):.4f}')

        if use_train_metrics:
            logger.info(f'Epoch Training Metrics:')
            for idx, metric in enumerate(train_epoch_metrics, start=1):
                tf.contrib.summary.scalar(name=f'train_metric_{idx}', tensor=metric, step=epoch)
                logger.info(f'Train Metric {idx}: {float(metric):.4f}')

        # Validation
        tic = time()
        logger.info(f'\nStarting Epoch {epoch:03d} Validation')
        val_epoch_loss, val_epoch_metrics = \
            val_epoch(model=model, dataset=val_dataset, loss_func=loss_func, epoch=epoch,
                      max_images=max_images, verbose=verbose, use_metrics=use_val_metrics)

        # After Epoch validation is over.
        toc = int(time() - tic)

        tf.contrib.summary.scalar(name='val_epoch_loss', tensor=val_epoch_loss, step=epoch)
        logger.info(f'Epoch {epoch:03d} Validation Finished. Time: {toc // 60}min {toc % 60}s.')
        logger.info(f'Epoch Validation loss: {float(val_epoch_loss):.4f}')

        if use_val_metrics:
            logger.info(f'Epoch Validation Metrics:')
            for idx, metric in enumerate(val_epoch_metrics, start=1):
                tf.contrib.summary.scalar(name=f'val_metric_{idx}', tensor=metric, step=epoch)
                logger.info(f'Validation Metric {idx}: {float(metric):.4f}')

        # Checkpoint the Epoch if there has been improvement.  # Not possible when the loss function keeps changing...
        if val_epoch_loss < prev_loss:
            prev_loss = val_epoch_loss
            logger.info('Validation loss has improved from previous epoch')
            logger.info(f'Last checkpoint file: {manager.latest_checkpoint}')
            manager.save(checkpoint_number=epoch)
        else:
            logger.info('Validation loss has not improved from previous epoch')
            logger.info(f'Previous minimum loss: {float(prev_loss):.4f}')
            if not save_best_only:
                manager.save(checkpoint_number=epoch)
