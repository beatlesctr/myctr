# coding:utf-8
import tensorflow as tf
import os

from feature.ml_1m import ML1MConfig, DataProcessor, \
    file_based_convert_examples_to_features, file_based_input_fn_builder, serving_input_fn_builder_v2
from flag_center import FLAGS
from model.dcn import DCN, DCNConfig
from model.deepfm import DeepFM, DeepFMConfig
from model.esmm import Esmm


def noam_scheme(init_lr, global_step):

    #step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr


def create_train_opt(loss, init_lr=0.001):

    global_step = tf.train.get_or_create_global_step()
    learning_rate = noam_scheme(init_lr=init_lr, global_step=global_step)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss=loss, global_step=global_step)
    return train_op, global_step


def my_model_fn(features, labels, mode, params):

    model_config = params['model_config']
    model_type = params['model_type']
    feat_config = params['feature_config']

    init_lr = params['init_lr']

    if type(features) == tf.Tensor:
        sparse_feat = features
        dense_feat = None
    else:
        sparse_feat, dense_feat = features
    y_label = labels
    if model_type == 'esmm':
        inst = Esmm(mode=mode, model_config=model_config, feat_config=feat_config)
        _, probs = inst.create_model(dense_feat=dense_feat, sparse_feat=sparse_feat)
        loss = inst.calculate_loss(probs=probs, labels=y_label)
    else:
        if model_type == 'dcn':
            inst = DCN(mode=mode, model_config=model_config, feat_config=feat_config)
        else:
            inst = DeepFM(mode=mode, model_config=model_config, feat_config=feat_config)
        logits, probs = inst.create_model(dense_feat=dense_feat, sparse_feat=sparse_feat)
        loss = DeepFM.calculate_loss(logits=logits, labels=y_label)

    for v in tf.trainable_variables():
        tf.logging.info(v.name)

    tf.summary.scalar('loss', loss)

    '''
    可以通过summary 来看看参数训练过程中的数据分布
    '''
    tf.summary.merge_all()

    if mode == tf.estimator.ModeKeys.TRAIN:
        '''
        模型训练
        '''
        train_op, global_step = create_train_opt(loss=loss, init_lr=init_lr)
        hook_dict = {
            'loss': loss,
            'step': global_step,
            #'sparse': sparse_,
            #'sparse_ori': sparse_feat
        }
        hook = tf.train.LoggingTensorHook(
            hook_dict,
            every_n_iter=1000,
        )
        return tf.estimator.EstimatorSpec(
            mode=mode,
            training_hooks=[hook],
            loss=loss,
            train_op=train_op)

    elif mode == tf.estimator.ModeKeys.EVAL:
        '''
        模型评估
        '''
        if model_type == 'esmm':
            metrics = inst.get_metrics()
        else:
            metrics = {
                'auc': tf.metrics.auc(
                    labels=tf.cast(x=labels, dtype=tf.bool),
                    predictions=probs[:, 1],
                    name='auc',
                    num_thresholds=2000
                )
            }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            eval_metric_ops=metrics,
            loss=loss
        )

    elif mode == tf.estimator.ModeKeys.PREDICT:
        '''
        模型预测
        '''
        if model_type == 'esmm':
            probs = probs[1]
        export_outputs = {
            'y': tf.estimator.export.PredictOutput(
                {"predict": probs}
            )
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={'predict': probs[:, 1]},
            export_outputs=export_outputs
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
            if f.startswith(feature_type) and f.endswith('txt'):
                input_files.append(os.path.join(data_dir, f))
    examples = data_processor.get_train_examples(input_files=input_files)
    file_based_convert_examples_to_features(examples=examples, output_dir=data_dir, feature_type=feature_type)


def get_tfrecord_filelist_from_dir(data_dir, has_dense_feat, feature_type='train'):
    '''
    获取迭代器
    '''
    input_files = list()
    for root, ds, fs in os.walk(data_dir):
        for f in fs:
            if f.startswith(feature_type) and f.endswith('tfrecord'):
                input_files.append(os.path.join(data_dir,f))
    train_input_fn = file_based_input_fn_builder(input_files=input_files, is_training=True,
                                                 has_dense_feat=has_dense_feat)
    return train_input_fn


def main(unused_params):
    # 加载配置
    feature_config = ML1MConfig.from_json_file(json_file=os.path.join(FLAGS.feature_dir, 'config.json'))
    model_config = DeepFMConfig.from_json_file(FLAGS.model_config_file)
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
        'model_config': model_config,
        'model_type': FLAGS.model_type,
        'feature_config': feature_config,
        'init_lr': FLAGS.init_lr,
        'train_batch_size': FLAGS.batch_size,
        'eval_or_predict_batch_size': FLAGS.batch_size

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
        #train_monitors=[train_input_hook],
        #eval_hooks=[eval_input_hook]
    )
    experiment.train_and_evaluate()

    if FLAGS.do_export:
        estimator._export_to_tpu = False
        serving_input_fn = serving_input_fn_builder_v2(sparse_feat_num=len(feature_config.sparce_feat_col_name),
                                                       dense_feat_num=len(feature_config.dense_feat_col_name))
        estimator.export_savedmodel(os.path.join(FLAGS.model_dir, "pbtxt_model"), serving_input_fn, as_text=True)


if __name__ == '''__main__''':

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()