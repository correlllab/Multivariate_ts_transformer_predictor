#https://www.tensorflow.org/text/tutorials/transformer#the_transformer
import tensorflow as tf


class GTN_CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,
                 d_model: int,
                 warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps


    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = tf.multiply(step, (tf.pow(self.warmup_steps, -1.5)))

        return tf.multiply(tf.math.rsqrt(self.d_model), tf.math.minimum(arg1, arg2))
