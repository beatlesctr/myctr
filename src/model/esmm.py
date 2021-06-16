import copy
import json

import six
import tensorflow as tf

from model.dcn import DCN, DCNConfig
from model.deepfm import DeepFM, DeepFMConfig
from model.dnn import DNN, DNNConfig


class EsmmConfig:

    def __init__(self,
                 emb_size=12,
                 model_name="dcn",
                 model_set=None
                 ):
        self.emb_size = emb_size,
        self.model_name = model_name,
        self.model_set = model_set


    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `TransformerConfig` from a Python dictionary of parameters."""
        config = EsmmConfig()
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


class Esmm:
    MODEL_MAP={
        "dnn":[DNN, DNNConfig],
        "dcn": [DCN, DCNConfig],
        "deepfm": [DeepFM, DeepFMConfig]
    }

    def __init__(self, mode, model_config, feat_config):

        self._model_config = model_config
        self._feat_config = feat_config
        self._mode = mode

        MODEL, MODELConfig = Esmm.MODEL_MAP[model_config.model_name]
        model_config_inner = MODELConfig.from_dict(MODELConfig.model_set[model_config.model_name])
        self._model = MODEL(mode=mode, model_config=model_config_inner, feat_config=feat_config)

        self._probs_ctr = None
        self._probs_ctcvr = None
        self._labels_ctr = None
        self._labels_ctcvr = None

    def create_model(self, dense_feat, sparse_feat):
        l_sparce_0, l_0_all = self._model.embedding_layer(self, dense_feat, sparse_feat)

        with tf.variable_scope(name_or_scope='bi-encoder', reuse=tf.AUTO_REUSE):
            _, self._probs_ctr = self._model.create_model_by_emb(l_sparce_0=l_sparce_0, l_0_all=l_0_all)
            _, self._probs_cvr = self._model.create_model_by_emb(l_sparce_0=l_sparce_0, l_0_all=l_0_all)
            p_ctcvr = self._probs_ctr[:,1] * self._probs_cvr[:, 1]
            self._probs_ctcvr = tf.concat(values=[1.0 - p_ctcvr, p_ctcvr], axis=-1)

        return None, (self._probs_ctr, self._probs_ctcvr)

    def __calculate_loss(self, labels, probs):

        y_label = tf.squeeze(input=labels, axis=-1)
        y_probs = probs
        loss_m = -1.0 * tf.log(y_probs + 1.0e-8) * tf.one_hot(indices=y_label, depth=2)
        loss_sum = tf.reduce_sum(input_tensor=loss_m, axis=-1)
        loss_per_sample = tf.reduce_mean(input_tensor=loss_sum)
        return loss_per_sample

    def __preprocess_label(self, labels):

        ones = tf.ones_like(tensor=labels)
        zeros = tf.zeros_like(tensor=labels)

        self._labels_ctcvr = tf.where(condition=tf.greater(x=labels,y=ones), x=ones, y=zeros)
        self._labels_ctr = tf.where(condition=tf.greater(x=labels,y=ones), x=ones, y=labels)

        return self._labels_ctr, self._labels_ctcvr

    def calculate_loss(self, probs, labels):

        probs_ctr = probs[0]
        probs_ctcvr = probs[1]
        labels_ctr, labels_ctcvr = self.__preprocess_label(labels=labels)
        loss_ctr = self.__calculate_loss(labels=labels_ctr, probs=probs_ctr)
        loss_ctcvr = self.__calculate_loss(labels=labels_ctcvr, probs=probs_ctcvr)

        return  loss_ctr + loss_ctcvr

    def get_metrics(self):

        metrics = {
            'ctr_auc': tf.metrics.auc(
                labels=tf.cast(x=self._labels_ctr, dtype=tf.bool),
                predictions=self._probs_ctr[:, 1],
                name='ctr_auc',
                num_thresholds=2000
            ),
            'ctcvr_auc': tf.metrics.auc(
                labels=tf.cast(x=self._labels_ctcvr, dtype=tf.bool),
                predictions=self._probs_ctcvr[:, 1],
                name='ctcvr_auc',
                num_thresholds=2000
            )
        }

        return metrics

