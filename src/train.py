# coding:utf-8
import tensorflow as tf
import os

from feature.ml_1m import file_based_input_fn_builder, ML1MConfig, DataProcessor, \
    file_based_convert_examples_to_features
from flag_center import FLAGS
from model.dcn import DCN, DCNConfig


def noam_scheme(init_lr, global_step):

    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr


def create_train_opt(loss, init_lr=0.001):
    global_steps_ = tf.train.get_or_create_global_step()
    global_step = tf.cast(x=global_steps_, dtype=tf.float32)
    learning_rate = noam_scheme(init_lr=init_lr, global_step=global_step)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss=loss, global_step=global_steps_)
    tf.summary.scalar('learning_rate', learning_rate)
    summaries = tf.summary.merge_all()
    return train_op, learning_rate


def my_model_fn(features, labels, mode, params):

    config = params['config']
    init_lr = params['init_lr']

    dense_feat, sparse_feat = features
    y_label = labels

    dcn = DCN(mode=mode, config=config)
    logits, probs = dcn.create_model(dense_feat=dense_feat, sparse_feat=sparse_feat)
    loss = dcn.calculate_loss(logits=logits, y_labels=y_label)

    for v in tf.trainable_variables():
        tf.logging.info(v.name)

    if mode == tf.estimator.ModeKeys.TRAIN:
        '''
        训练rnn 模型的时候推荐的方法
        '''
        train_op, learning_rate = create_train_opt(loss=loss, init_lr=init_lr)
        hook_dict = {
            'loss': loss,
            'learning_rate': learning_rate,
        }
        hook = tf.train.LoggingTensorHook(
            hook_dict,
            every_n_iter=10
        )
        return tf.estimator.EstimatorSpec(
            mode=mode,
            training_hooks=[hook],
            loss=loss,
            train_op=train_op)

    elif mode == tf.estimator.ModeKeys.EVAL:

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={'prediction': probs}
        )

    else:

        raise NotImplementedError('not implemented')


def convert_feature_from_txt_to_tfrecord(data_dir, feature_config, feature_type='train'):
    '''
    特征转化
    '''
    data_processor = DataProcessor(feat_cfg=feature_config)
    input_files = list()
    for root, ds, fs in os.walk(data_dir):
        for f in fs:
            if f.startswith(prefix=feature_type) and f.endswith(suffix='txt'):
                input_files.append(f)
    examples = data_processor.get_train_examples(data_dir=input_files)
    file_based_convert_examples_to_features(examples=examples, output_dir=data_dir, feature_type=feature_type)


def get_tfrecord_filelist_from_dir(data_dir, has_dense_feat, feature_type='train'):
    '''
    获取迭代器
    '''
    input_files = list()
    for root, ds, fs in os.walk(data_dir):
        for f in fs:
            if f.startswith(prefix=feature_type) and f.endswith(suffix='tfrecord'):
                input_files.append(f)
    train_input_fn = file_based_input_fn_builder(input_files=[], is_training=True,
                                                 has_dense_feat=has_dense_feat)
    return train_input_fn


def main(unused_params):
    # 加载配置
    feature_config = ML1MConfig.from_json_file(json_file=os.path.join(FLAGS.feature_dir, 'config.json'))
    model_config = DCNConfig.from_json_file(FLAGS.model_config)
    tf.logging.info(model_config.to_json_string())
    tf.logging.info(feature_config.to_json_string())
    # 处理特征
    if FLAGS.mk_feature:
        convert_feature_from_txt_to_tfrecord(data_dir=FLAGS.feature_dir,
                                             feature_config=feature_config,
                                             feature_type='train')
        convert_feature_from_txt_to_tfrecord(data_dir=FLAGS.feature_dir,
                                             feature_config=feature_config,
                                             feature_type='eval')
    # 构造评估器
    train_step_num = int(feature_config.train_sample_num * FLAGS.epoch_num / FLAGS.batch_size)
    eval_step_num = int(feature_config.test_sample_num / FLAGS.batch_size)
    run_config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir,
                                        save_checkpoints_steps=FLAGS.save_checkpoint_steps,
                                        keep_checkpoint_max=FLAGS.keep_checkpoint_max)
    params = {
        'train_step_num': train_step_num,
        'eval_step_num': eval_step_num,
        'epoch_num': FLAGS.epoch_num,
        'model_config': model_config,
        'feature_config': feature_config,
        'init_lr': FLAGS.init_lr,
        'train_batch_size': FLAGS.batch_size,
        'predict_batch_size': FLAGS.batch_size
    }

    estimator = tf.estimator.Estimator(model_dir=FLAGS.model_dir,
                                       model_fn=my_model_fn,
                                       config=run_config,
                                       params=params)
    # 启动实验
    has_dense_feat = len(feature_config.dense_feat_col_name) > 0
    train_input_fn = get_tfrecord_filelist_from_dir(data_dir=FLAGS.feature_dir,
                                                    has_dense_feat=has_dense_feat,
                                                    feature_type='train')
    eval_input_fn = get_tfrecord_filelist_from_dir(data_dir=FLAGS.feature_dir,
                                                   has_dense_feat=has_dense_feat,
                                                   feature_type='eval')

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=train_step_num,
        eval_steps=eval_step_num,
        min_eval_frequency=FLAGS.save_checkpoint_steps
        #train_monitors=[train_input_hook],  # Hooks for training
        #eval_hooks=[eval_input_hook],  # Hooks for evaluation
    )
    experiment.train_and_eval()


if __name__ == '''__main__''':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()