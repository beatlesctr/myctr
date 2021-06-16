from model.esmm import EsmmConfig
import tensorflow as tf
ret = EsmmConfig.from_json_file(json_file="./cfg/esmm.json")
print(ret)
a = tf.constant([[1.,2.],[3.,4.]], dtype=tf.float32)
with tf.variable_scope(name_or_scope='bi-encoder', reuse=tf.AUTO_REUSE):
    aa = tf.layers.dense(inputs=a, units=10, name="a")
    ab = tf.layers.dense(inputs=a, units=10, name="a")
    v = tf.trainable_variables()
    print(v)
