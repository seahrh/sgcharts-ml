import math
import warnings

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    tf = None
    keras = None
    warnings.warn("Install tensorflow to use this feature", ImportWarning)

__all__ = ["ArcMarginProduct"]


@keras.utils.register_keras_serializable()
class ArcMarginProduct(keras.layers.Layer):
    """
    Implements large margin arc distance.

    Reference:
        https://arxiv.org/pdf/1801.07698.pdf
        https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/
            blob/master/src/modeling/metric_learning.py
    """

    def __init__(
        self, n_classes, scale=30, margin=0.50, easy_margin=False, ls_eps=0.0, **kwargs
    ):
        super(ArcMarginProduct, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.scale = scale
        self.margin = margin
        self.ls_eps = ls_eps
        self.easy_margin = easy_margin
        self.cos_m = tf.math.cos(margin)
        self.sin_m = tf.math.sin(margin)
        self.th = tf.math.cos(math.pi - margin)
        self.mm = tf.math.sin(math.pi - margin) * margin
        self.W = None

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "n_classes": self.n_classes,
                "scale": self.scale,
                "margin": self.margin,
                "ls_eps": self.ls_eps,
                "easy_margin": self.easy_margin,
            }
        )
        return config

    def build(self, input_shape):
        super(ArcMarginProduct, self).build(input_shape[0])

        self.W = self.add_weight(
            name="W",
            shape=(int(input_shape[0][-1]), self.n_classes),
            initializer="glorot_uniform",
            dtype="float32",
            trainable=True,
            regularizer=None,
        )

    def call(self, inputs, **kwargs):
        X, y = inputs
        y = tf.cast(y, dtype=tf.int32)
        cosine = tf.matmul(
            tf.math.l2_normalize(X, axis=1), tf.math.l2_normalize(self.W, axis=0)
        )
        sine = tf.math.sqrt(1.0 - tf.math.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = tf.where(cosine > 0, phi, cosine)
        else:
            phi = tf.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = tf.cast(tf.one_hot(y, depth=self.n_classes), dtype=cosine.dtype)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.n_classes
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        return output
