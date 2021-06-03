import tensorflow as tf


class Esmm:

    def __init__(self, mode, model_config, feat_config):

        self._model_config = model_config
        self._feat_config = feat_config
        self._mode = mode

        self._sparse_feat_num = len(self._feat_config.sparse_feat_space_cfg)
        self._dense_feat_num = len(self._feat_config.dense_feat_col_name)
        self._sparse_feat_space_size = sum(self._feat_config.sparse_feat_space_cfg)

        sparse_feat_steps = [0 for ele in self._feat_config.sparse_feat_space_cfg]

        for i in range(1, len(self._feat_config.sparse_feat_space_cfg), 1):
            sparse_feat_steps[i] = sparse_feat_steps[i-1] + self._feat_config.sparse_feat_space_cfg[i-1]
        self._sparse_feat_steps = tf.constant(value=sparse_feat_steps, dtype=tf.int32, name='sparse_feat_steps')

        self._probs_ctr = None
        self._probs_ctcvr = None
        self._labels_ctr = None
        self._labels_ctcvr = None


    def __sparse_feature_preprocess(self, sparse_feat):
        input_tmp = list()
        for i in range(len(self._feat_config.sparse_feat_space_cfg)):
            feat = tf.to_int32(tf.string_to_hash_bucket_fast(
                input=sparse_feat[:, i],
                num_buckets=self._feat_config.sparse_feat_space_cfg[i]
            )) + self._sparse_feat_steps[i]
            input_tmp.append(feat)
        sparse_feat = tf.transpose(a=tf.stack(values=input_tmp, axis=0), perm=[1,0])
        return sparse_feat

    def __embedding_layer(self, dense_feat, sparse_feat):
        '''
        :param dense_feat: B x Nd
        :param sparse_feat: B x Ns
        :return:
        '''
        # 构造 cross layer 的输入
        # B x N x E

        sparse_feat = self.__sparse_feature_preprocess(sparse_feat=sparse_feat)
        emb_matrix_s = tf.get_variable(name="emb_s",
                                       dtype=tf.float32,
                                       shape=[self._sparse_feat_space_size, self._model_config.emb_size],
                                       initializer=tf.truncated_normal_initializer(stddev=0.02))
        sparse_feat_emb = tf.nn.embedding_lookup(params=emb_matrix_s, ids=sparse_feat)
        # B x (Ns*E)
        sparse_feat_emb = tf.reshape(tensor=sparse_feat_emb,
                                     shape=[-1, self._sparse_feat_num * self._model_config.emb_size])

        # 构造 dnn layer 的输入
        if dense_feat is not None:
            feat_emb = tf.concat(values=[sparse_feat_emb, dense_feat], axis=-1)
        else:
            feat_emb = sparse_feat_emb
        # B x (Ns*E+Nd)
        feat_emb = tf.reshape(tensor=feat_emb,
                         shape=[-1, (self._sparse_feat_num*self._model_config.emb_size + self._dense_feat_num)])
        return sparse_feat_emb, feat_emb

    def __dnn_layer(self, l_0_all):
        '''
        :param l_0_all: B x N
        :return: B x M
        '''
        keep_prob = 1.0 - self._model_config.dropout_prob if tf.estimator.ModeKeys.TRAIN == self._mode else 1.0
        y = l_0_all
        for i in range(len(self._model_config.dnn_layer_cfg)):
            y = tf.layers.dense(inputs=y, units=self._model_config.dnn_layer_cfg[i],
                                activation=tf.nn.relu, name='dense-' + str(i))
            y = tf.nn.dropout(x=y, keep_prob=keep_prob)
        l_n_all = y
        return l_n_all

    def create_model(self, dense_feat, sparse_feat):

        sparse_feat = self.__sparse_feature_preprocess(sparse_feat=sparse_feat)
        sparse_feat_emb, feat_emb = self.__embedding_layer(dense_feat=dense_feat, sparse_feat=sparse_feat)

        with tf.variable_scope(name_or_scope='bi-encoder', reuse=tf.AUTO_REUSE):
            logits_cvr = self.__dnn_layer(l_0_all=feat_emb)
            logits_ctr = self.__dnn_layer(l_0_all=feat_emb)

            logits_ctr = tf.layers.dense(inputs=logits_ctr, units=self._feat_config.label_num, name='ctr')
            logits_cvr = tf.layers.dense(inputs=logits_cvr, units=self._feat_config.label_num, name='cvr')

            probs_ctr = tf.nn.softmax(logits=logits_ctr, axis=-1)
            probs_cvr = tf.nn.softmax(logits=logits_cvr, axis=-1)

            self._p_ctcvr = probs_ctr[:,1] * probs_cvr[:, 1]
            self._probs_ctcvr = tf.concat(values=[1.0 - self._p_ctcvr, self._p_ctcvr], axis=-1)

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

