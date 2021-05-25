from feature.ml_1m import ML1MConfig
from model.dcn import DCNConfig, DCN
import tensorflow as tf
tf.enable_eager_execution()


class TestDCN():

    def __init__(self):

        model_config = DCNConfig.from_json_file("test/data/dcn.json")
        feat_config = ML1MConfig.from_json_file("test/data/config.json")
        self._inst = DCN(mode=tf.estimator.ModeKeys.TRAIN, model_config=model_config, feat_config=feat_config)

    def test__embedding_layer(self):

        B = 2
        dense_feat = None
        # [10, 10, 10, 10]
        values = [['5830','3689','1','M','25','92120','Starman (1984)','Adventure1','Drama1','Romance1'],
                  ['5829', '3688', '1', 'M', '24', '92119', 'Starman (1983)', 'Adventure', 'Drama', 'Romance']]
        sparse_feat = tf.constant(value=values, dtype=tf.string)
        l0_sparse, l0_all = self._inst.test__embedding_layer(dense_feat=dense_feat, sparse_feat=sparse_feat)

        print(l0_sparse)
        print(l0_all)
        return l0_sparse, l0_all

    def test__cross_layer(self):

        l0_sparse, _ = self.test__embedding_layer()
        l_n = self._inst.test__cross_layer(l_0=l0_sparse)
        print(l_n)
        return l_n

    def test__dense_layer(self):

        _, l_0_all = self.test__embedding_layer()
        y = self._inst.test__dnn_layer(l_0_all=l_0_all)
        print(y)
        return y

    def test_create_model(self):

        B = 2
        dense_feat = tf.constant(value=[[1.0, 2.0] for ele in range(B)], dtype=tf.float32)
        # [10, 10, 10, 10]
        sparse_feat = tf.constant(value=[[1, 12, 23, 34] for ele in range(B)], dtype=tf.int32)
        logits, probs = self._inst.create_model(dense_feat=dense_feat, sparse_feat=sparse_feat)
        print(logits, probs)
        return logits, probs

    def test_calculate_loss(self):

        logits, probs = self.test_create_model()
        labels = tf.constant(value=[1, 0], dtype=tf.int32)
        loss_per_sample = self._inst.test_calculate_loss(logits=logits, labels=labels)
        print(loss_per_sample)
        return loss_per_sample

if __name__ == '''__main__''':

    TestDCN().test__embedding_layer()
    #TestDCN().test__cross_layer()
    #TestDCN().test_create_model()
    #loss_per_sample = TestDCN().test_calculate_loss()