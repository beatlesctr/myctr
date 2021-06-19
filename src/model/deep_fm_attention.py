import copy
import json
import six
import math

import tensorflow as tf


from feature.ml_1m import ML1MConfig
from model.deepfm import DeepFM, DeepFMConfig

class DeepFMAttention():

    def __init__(self, mode, model_config, feat_config):
        self._mode = mode
        self._model_config = model_config

        self._feat_config = feat_config
        ml1m_feat_config = ML1MConfig.from_dict(self._feat_config.ml1m_config)
        self._deepfm = DeepFM(mode, model_config=self._model_config, feat_config=ml1m_feat_config)

    @staticmethod
    def __scaled_dot_product_attention(q, k, v, mask_q, mask_k, mask_v,
                                       dk, attention_dropout=0.0, training=True):
        # Input:
        # 	k,q,v: B x S x Ha
        # 	mask_k,mask_q,mask_v: B x S
        # Return:
        #	B x S x Ha

        def attention_mask_before_softmax(matrix, to_mask):
            to_mask = tf.cast(x=to_mask, dtype=tf.float32)
            attention_adder = (1.0 - tf.expand_dims(input=to_mask, axis=1)) * (-2.0**28+1.0)
            return matrix + attention_adder	# here attention_adder will be broadcast according to axis 1

        # QK^T
        dot_product = tf.matmul(a=q, b=k, transpose_b=True)
        # scale
        dk = tf.cast(x=dk, dtype=tf.float32)
        scale_dot_product = dot_product / tf.sqrt(dk)
        # mask & softmax
        scale_dot_product = attention_mask_before_softmax(matrix=scale_dot_product,to_mask=mask_k)
        attention_weight_a = tf.nn.softmax(logits=scale_dot_product, axis=-1)
        attention_weight_a = tf.layers.dropout(inputs=attention_weight_a, rate=attention_dropout, training=training)

        # attention
        attention_score = tf.matmul(a=attention_weight_a, b=v)
        return 	attention_score

    def __seq_feat_emb(self, seq_feat):

        emb_list = list()
        dk = 0
        for (i, s) in enumerate(self._feat_config.behavior_attr_space_cfg):
            emb_size = int(math.log2(s) * 1.2)
            matrix_emb = tf.get_variable(name="hehavior_embedding_matrix_%d"%(i), shape=[s, emb_size], dtype=tf.float32)
            tmp_i = seq_feat[:, :, i]
            emb_i = tf.nn.embedding_lookup(params=matrix_emb, ids=tmp_i)
            emb_list.append(emb_i)
            dk += s[1]
        return tf.concat(values=emb_list, axis=-1), dk

    def __attention_layer(self, cur_item_feat, history_behavior_seq_feat, history_behavior_seq_feat_mask):

        # attention part
        cur_item_feat_seq, dk = tf.expand_dims(input=cur_item_feat, axis=1)
        q = self.__seq_feat_emb(seq_feat=cur_item_feat_seq)
        attention_score_list = list()
        k_i,_ = self.__seq_feat_emb(seq_feat=history_behavior_seq_feat)
        k_i_mask = history_behavior_seq_feat_mask
        attention_score_i = DeepFMAttention.__scaled_dot_product_attention(q=q, k=k_i, v=k_i,
                                                                mask_q=None, mask_k=k_i_mask, mask_v=k_i_mask,
                                                                dk=dk, training=tf.estimator.ModeKeys.TRAIN==self._mode)

        attention_score = tf.squeeze(attention_score_i, axis=1)

        return attention_score

    def create_model(self, sparse_feat, dense_feat,
                     cur_item_feat, history_behavior_seq_feat_list, history_behavior_seq_feat_mask_list):
        '''
        :param sparse_feat: B x Ns
        :type sparse_feat: tf.Tensor
        :param dense_feat: B x Nd
        :type dense_feat: tf.Tensor
        :param cur_item_feat: B x Ni
        :type cur_item_feat: tf.Tensor
        :param history_behavior_seq_feat_list: B x S x Ni
        :type history_behavior_seq_feat_list: tf.Tensor
        :param history_behavior_seq_feat_mask_list: B x S
        :type history_behavior_seq_feat_mask_list: tf.Tensor
        :return: logits, prob
        :rtype: tf.Tensor, tf.Tensor
        '''
        # attention part
        attention_score = self.__attention_layer(cur_item_feat=cur_item_feat,
                                                 history_behavior_seq_feat_list=history_behavior_seq_feat_list,
                                                 history_behavior_seq_feat_mask_list=history_behavior_seq_feat_mask_list)
        l_sparce_0, l_0_all = self._deepfm.embedding_layer(dense_feat=dense_feat, sparse_feat=sparse_feat)
        # fm part
        l_n = self.__fm_layer(l_0=l_sparce_0)
        # dnn part
        y = self.__dnn_layer(l_0_all=l_0_all)

        # 所部分组合
        tmp = tf.concat(values=[l_n, y, attention_score], axis=-1)
        logits = tf.layers.dense(inputs=tmp, units=self._feat_config.label_num, name='proj')  # 最后是二分类
        probs = tf.nn.softmax(logits=logits, axis=-1)

        return logits, probs


