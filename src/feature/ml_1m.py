import collections
import copy
import json
import os
import pandas as pd
import six
import tensorflow as tf


class Example:
    def __init__(self, guid, sparse_feat, dense_feat, y_label):
        self.guid = guid
        self.sparse_feat = sparse_feat
        self.dense_feat = dense_feat
        self.y_label = y_label


class ML1MConfig(object):
    """Configuration for `TransformerModel`."""

    def __init__(self,
                 dense_feat_col_name=[''],
                 sparce_feat_col_name=['UserID','MovieID',
                                       'Gender','Age','Occupation',
                                       'Zip-code','Title','Genres1',
                                       'Genres2','Genres3'],
                 sparse_feat_space_cfg=[6040, 3706, 2, 7, 21, 3439, 3706, 18, 18, 18],
                 label_col_name=['Rating'],
                 label_num=2,
                 train_sample_num=102400,
                 test_sample_num=10240
                 ):
        self.dense_feat_col_name = dense_feat_col_name
        self.sparce_feat_col_name = sparce_feat_col_name
        self.sparse_feat_space_cfg = sparse_feat_space_cfg
        self.label_col_name = label_col_name
        self.label_num = label_num
        self.train_sample_num = train_sample_num
        self.test_sample_num = test_sample_num

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `TransformerConfig` from a Python dictionary of parameters."""
        config = ML1MConfig()
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


class DataProcessor(object):
    """Base class for dat converters for sequence classification dat sets."""

    def __init__(self, feat_cfg):

        self._feat_cfg = feat_cfg

    def get_train_examples(self, input_files):

        sparse_feats, dense_feats, y_labels = self._read_txt_as_dataframe(input_files=input_files)
        return self._create_examples(sparse_feats, dense_feats, y_labels)

    def get_dev_examples(self, input_files):

        sparse_feats, dense_feats, y_labels = self._read_txt_as_dataframe(input_files=input_files)
        return self._create_examples(sparse_feats, dense_feats, y_labels)


    def get_test_examples(self, input_files):

        sparse_feats, dense_feats, y_labels = self._read_txt_as_dataframe(input_files=input_files)
        return self._create_examples(sparse_feats, dense_feats, y_labels)


    def get_labels(self):
        """Gets the list of labels for this dat set."""
        raise NotImplementedError()

    def _read_txt_as_dataframe(self, input_files):
        df_list = list()
        for input_file in input_files:
            dataframe = pd.read_csv(filepath_or_buffer=input_file, sep=':')
            df_list.append(dataframe)
        dataframe = pd.concat(df_list)
        sparse_feats = dataframe[self._feat_cfg.sparce_feat_col_name]
        dense_feats = None
        if len(self._feat_cfg.dense_feat_col_name) != 0:
            dense_feats = dataframe[self._feat_cfg.dense_feat_col_name]
        y_labels = dataframe[self._feat_cfg.label_col_name]

        return sparse_feats, dense_feats, y_labels

    def _create_examples(self, sparse_feats, dense_feats, y_labels):

        """Creates examples for the training and dev sets."""
        size = len(y_labels)
        examples = []
        for i in range(size):
            if i % 10000 == 0:
                tf.logging.info('create %f examples of all'%(float(i)/float(size)))
            example = Example(guid='guid' + str(i)
                              , sparse_feat=sparse_feats.iloc[i].array
                              , dense_feat=dense_feats.iloc[i].array if dense_feats is not None else None
                              , y_label=y_labels.iloc[i].array)
            examples.append(example)
        return examples
        

def file_based_convert_examples_to_features(examples, output_dir, feature_type='train'):

    def create_int_feature(values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f

    def create_str_feature(values):
        f = tf.train.Feature(bytes_list=
                             tf.train.BytesList(value=[str(ele).encode('utf-8') for ele in values]))
        return f

    def create_float_feature(values):
        f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
        return f

    is_open = False
    writer = None
    for (ex_index, example) in enumerate(examples):

        if ex_index % 100000 == 0:
            if is_open:
                writer.close()
                is_open = False
            writer = tf.python_io.TFRecordWriter(
                os.path.join(output_dir, feature_type + str(ex_index)+'.tfrecord'))
        features = collections.OrderedDict()

        features['sparse_feature'] = create_str_feature(example.sparse_feat)
        if example.dense_feat is not None:
            features['dense_feature'] = create_float_feature(example.dense_feat)
        features['y_label'] = create_int_feature(example.y_label)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writer.write(tf_example.SerializeToString())

        is_open = True
    if is_open:
        writer.close()


def file_based_input_fn_builder(input_files, is_training, has_dense_feat):

    name_to_feature_sd = {
        'sparse_feature': tf.VarLenFeature(dtype=tf.string),
        'dense_feature': tf.VarLenFeature(dtype=tf.float32),
        'y_label': tf.VarLenFeature(dtype=tf.int64)
    }

    name_to_feature_s = {
        'sparse_feature': tf.VarLenFeature(dtype=tf.string),
        #'dense_feat': tf.VarLenFeature(dtype=tf.float32),
        'y_label': tf.VarLenFeature(dtype=tf.int64)
    }

    def _decode_record(record, name_to_feature):
        example = tf.parse_single_example(record, name_to_feature)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.sparse_tensor_to_dense(t, default_value=1)
                t = tf.to_int32(t)
            if t.dtype == tf.string:
                t = tf.sparse_tensor_to_dense(t, default_value='')
            if t.dtype == tf.float32:
                t = tf.sparse_tensor_to_dense(t, default_value=0.0)

            example[name] = t

        if 'dense_feature' in name_to_feature.keys():
            return (example['sparse_feature'], example['dense_feature']), example['y_label']
        return (example['sparse_feature']), example['y_label']

    def input_fn(params):

        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            d = d.repeat()
            d = d.shuffle(buffer_size=len(input_files))

            d = d.apply(
                tf.contrib.data.parallel_interleave(
                    tf.data.TFRecordDataset,
                    sloppy=is_training,
                    cycle_length=16
                )
            )
            d = d.shuffle(buffer_size=1024000)

            batch_size = params['train_batch_size']
        else:
            d = tf.data.TFRecordDataset(filenames=input_files)
            batch_size = params['eval_or_predict_batch_size']
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_feature_sd if has_dense_feat else name_to_feature_s),
                batch_size=batch_size,
                drop_remainder=not is_training
            )
        )
        return d

    return input_fn


def serving_input_fn_builder_v2(sparse_feat_num, dense_feat_num):

    def serving_input_fn():
        feature_spec = dict()
        sparse_feat = tf.compat.v1.placeholder(tf.string, [None, sparse_feat_num], name='sparse_feat')
        feature_spec['sparse_feat'] = sparse_feat
        if dense_feat_num > 0:
            dense_feat = tf.compat.v1.placeholder(tf.float32, [None, dense_feat_num], name='dense_feat')
            feature_spec['dense_feat'] = dense_feat

        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)()
        return serving_input_receiver_fn

    return serving_input_fn
