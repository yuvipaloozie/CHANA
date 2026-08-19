"""Losses used by the historical TensorFlow training notebooks."""

from __future__ import annotations


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise ImportError("Install CHANA with the 'tensorflow' extra to use losses") from exc
    return tf


def focal_tversky_loss(
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 1.5,
    smooth: float = 1e-6,
):
    """Return the focal-Tversky loss used in the training exports.

    ``alpha`` weights false negatives and ``beta`` weights false positives.
    """
    tf = _tensorflow()

    def loss(y_true, y_pred):
        y_true_flat = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        y_pred_flat = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
        true_positive = tf.reduce_sum(y_true_flat * y_pred_flat)
        false_negative = tf.reduce_sum(y_true_flat * (1.0 - y_pred_flat))
        false_positive = tf.reduce_sum((1.0 - y_true_flat) * y_pred_flat)
        tversky = (true_positive + smooth) / (
            true_positive + alpha * false_negative + beta * false_positive + smooth
        )
        return tf.pow(1.0 - tversky, gamma)

    return loss


def soft_binary_crossentropy(label_smoothing: float = 0.1):
    """Return binary cross-entropy with the pseudo/noisy-label smoothing setting."""
    tf = _tensorflow()
    return tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)


def dice_loss(y_true, y_pred, smooth: float = 1e-6):
    return 1.0 - dice_coefficient(y_true, y_pred, smooth=smooth)


def dice_coefficient(y_true, y_pred, smooth: float = 1.0):
    tf = _tensorflow()
    y_true_flat = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_flat = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) + smooth
    )
