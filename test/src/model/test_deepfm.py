from feature.ml_1m import ML1MConfig

import tensorflow as tf

from model.deepfm import DeepFMConfig, DeepFM

tf.enable_eager_execution()


class TestDeepFM():

    def __init__(self):

        model_config = DeepFMConfig.from_json_file("./data/deepfm.json")
        feat_config = ML1MConfig.from_json_file("./data/config.json")
        self._inst = DeepFM(mode=tf.estimator.ModeKeys.TRAIN, model_config=model_config, feat_config=feat_config)

    def test__embedding_layer(self):

        B = 2
        dense_feat = None
        # [10, 10, 10, 10]
        values = [['5830','3689','1','M','25','92120','Starman (1984)','Adventure1','Drama1','Romance1'],
                  ['5829', '3688', '1', 'M', '24', '92119', 'Starman (1983)', 'Adventure', 'Drama', 'Romance']]
        sparse_feat = tf.constant(value=values, dtype=tf.string)
        sparse_feat = self._inst.test__sparse_feature_preprocess(sparse_feat=sparse_feat)
        l0_sparse, l0_all = self._inst.test__embedding_layer(dense_feat=dense_feat, sparse_feat=sparse_feat)

        print(l0_sparse)
        print(l0_all)
        return l0_sparse, l0_all

    def test__fm_layer(self):

        l0_sparse, _ = self.test__embedding_layer()
        l_n = self._inst.test__fm_layer(l_0=l0_sparse)
        print(l_n)
        return l_n

    def test__dense_layer(self):

        _, l_0_all = self.test__embedding_layer()
        y = self._inst.test__dnn_layer(l_0_all=l_0_all)
        print(y)
        return y



if __name__ == '''__main__''':

    #TestDeepFM().test__embedding_layer()
    #TestDeepFM().test__fm_layer()
    TestDeepFM().test__dense_layer()
    #TestDCN().test__cross_layer()
    #TestDCN().test_create_model()
    #loss_per_sample = TestDCN().test_calculate_loss()