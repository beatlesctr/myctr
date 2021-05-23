from src.feature.ml_1m import ML1MConfig, DataProcessor, file_based_convert_examples_to_features, \
    file_based_input_fn_builder


class DataProcessorTest:

    def __init__(self):
        feat_cfg = ML1MConfig.from_json_file(json_file='.\\dat\\ml-1m\\config.json')
        self._inst = DataProcessor(feat_cfg=feat_cfg)

    def test_get_train_examples(self):
        data_dir = ['.\\dat\\ml-1m\\eval.txt']
        tmp = self._inst.get_train_examples(data_dir=data_dir)
        for ele in tmp[:100]:
            print(ele)
        return tmp

    def test_file_based_convert_examples_to_features(self):
        examples = self.test_get_train_examples()
        file_based_convert_examples_to_features(examples=examples,
                                                output_dir=".\\test\\out\\tf\\")

    def test_file_based_input_fn_builder(self):
        input_files = [".\\test\\out\\tf\\0.tfrecord",
                      ".\\test\\out\\tf\\100000.tfrecord"]
        input_fn = file_based_input_fn_builder(input_files=input_files,
                                               is_training=True, has_dense_feat=False)
        d = input_fn(params={"batch_size": 2})
        iterator = d.make_one_shot_iterator()
        one_element = iterator.get_next()
        with tf.Session() as sess:
            for i in range(5):
                print(sess.run(one_element))

if __name__=='''__main__''':

    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.INFO)
    test = DataProcessorTest()
    test.test_file_based_convert_examples_to_features()
    test.test_file_based_input_fn_builder()