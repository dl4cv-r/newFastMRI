import tensorflow as tf


def batch_ssim(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def batch_msssim(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, max_val=1.0))


def batch_psnr(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))


def batch_nmse(y_true, y_pred):
    return tf.divide(tf.reduce_sum(tf.squared_difference(y_true, y_pred)), tf.reduce_sum(y_true ** 2))
