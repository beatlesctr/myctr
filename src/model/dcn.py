import copy
import json

import six
import tensorflow as tf


class DCNConfig(object):
    """Configuration for `TransformerModel`."""

    def __init__(self,
                 emb_size=12,
                 dense_feat_num=512,
                 sparse_feat_space_cfg=[6, 7],
                 cross_layer_num=3,
                 dnn_layer_cfg=[128, 64],
                 label_num=5,
                 dropout_prob=0.1):
        self.emb_size = emb_size,
        self.dense_feat_num = dense_feat_num,
        self.sparse_feat_space_cfg = sparse_feat_space_cfg,
        self.cross_layer_num = cross_layer_num,
        self.dnn_layer_cfg = dnn_layer_cfg,
        self.label_num = label_num
        self.dropout_prob = dropout_prob

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `TransformerConfig` from a Python dictionary of parameters."""
        config = DCNConfig()
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `TransformerConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DCN:

    def __init__(self, mode, config):

        self._config = config
        self._mode = mode
        self._sparse_feat_num = len(self._config.sparse_feat_space_cfg)

    def __embedding_layer(self, dense_feat, sparse_feat):
        '''
        :param dense_feat: B x Nd
        :param sparse_feat: B x Ns
        :return:
        '''
        # 构造 cross layer 的输入
        # B x N x E
        sparse_feat_space_size = sum(self._config.sparse_feat_space_cfg)
        emb_matrix_s = tf.get_variable(name="emb_s",
                                       dtype=tf.float32,
                                       shape=[sparse_feat_space_size, self._config.emb_size],
                                       initializer=tf.truncated_normal_initializer(stddev=0.02))
        sparse_feat_emb = tf.nn.embedding_lookup(params=emb_matrix_s, ids=sparse_feat)
        # B x (Ns*E)
        sparse_feat_emb = tf.reshape(tensor=sparse_feat_emb,
                                     shape=[-1, self._sparse_feat_num * self._config.emb_size])

        # 构造 dnn layer 的输入
        feat_emb = tf.concat(values=[sparse_feat_emb, dense_feat], axis=-1)
        # B x (Ns*E+Nd)
        feat_emb = tf.reshape(tensor=feat_emb,
                         shape=[-1, (self._sparse_feat_num*self._config.emb_size + self._config.dense_feat_num)])
        return sparse_feat_emb, feat_emb

    def __cross_layer(self, l_0):
        '''
        :param l_0: B x Ns x 1
        :return: B x Ns
        '''
        # cross 网络
        l_0 = tf.expand_dims(input=l_0, axis=-1)
        l_n = l_0 # B x Ns x 1
        for i in range(1, self._config.cross_layer_num):

            W = tf.get_variable(name="layer-weight-"+str(i), shape=[self._sparse_feat_num*self._config.emb_size, 1])
            b = tf.get_variable(name="layer-bias-"+str(i), shape=[self._sparse_feat_num*self._config.emb_size, 1])
            tmp = tf.einsum("bik, kj->bij", tf.transpose(a=l_n, perm=[0, 2, 1]), W) # B x 1 x 1
            l_n = tf.matmul(a=l_0, b=tmp) + b + l_n

        return tf.squeeze(input=l_n, axis=[-1])

    def __dnn_layer(self, l_0_all):
        '''
        :param l_0_all: B x N
        :return: B x M
        '''
        keep_prob = 1.0 - self._config.dropout_prob if tf.estimator.ModeKeys.TRAIN == self._mode else 1.0
        y = l_0_all
        for i in range(len(self._config.dnn_layer_cfg)):
            y = tf.layers.dense(inputs=y, units=self._config.dnn_layer_cfg[i],
                                activation=tf.nn.relu, name='dense-' + str(i))
            y = tf.nn.dropout(x=y, keep_prob=keep_prob)
        return y

    def calculate_loss(self, logits, labels):

        batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        loss_per_sample = tf.reduce_mean(input_tensor=batch_loss, axis=-1, keep_dims=False)
        return loss_per_sample

    def create_model(self, dense_feat, sparse_feat):

        l_sparce_0, l_0_all = self.__embedding_layer(dense_feat=dense_feat, sparse_feat=sparse_feat)
        l_n = self.__cross_layer(l_0=l_sparce_0)
        y = self.__dnn_layer(l_0_all=l_0_all)
        tmp = tf.concat(values=[l_n, y], axis=-1)

        logits = tf.layers.dense(inputs=tmp, units=self._config.label_num) # 最后是二分类
        probs = tf.nn.softmax(logits=logits, axis=-1)

        return logits, probs

    '''
    test interfaces
    '''
    def test__embedding_layer(self, dense_feat, sparse_feat):

        return self.__embedding_layer(dense_feat=dense_feat, sparse_feat=sparse_feat)

    def test__cross_layer(self, l_0):

        return self.__cross_layer(l_0=l_0)

    def test__dnn_layer(self, l_0_all):

        return self.__dnn_layer(l_0_all=l_0_all)

    def test_create_model(self, dense_feat, sparse_feat):

        return self.create_model(dense_feat=dense_feat, sparse_feat=sparse_feat)

    def test_calculate_loss(self, logits, labels):

        return self.calculate_loss(logits=logits, labels=labels)





