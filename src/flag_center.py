import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(name='model_config_file', default='./cfg/dcn.json', help='模型配置文件位置')

flags.DEFINE_integer(name='epoch_num', default=10, help='epoch数量')
flags.DEFINE_integer(name='batch_size', default=1024, help='batch 大小')
flags.DEFINE_float(name='init_lr', default=0.001, help='初始学习率')

flags.DEFINE_integer(name='save_checkpoint_steps', default=1024, help='每隔多少步保存模型')
flags.DEFINE_integer(name='keep_checkpoint_max', default=3, help='最多保存多少checkpoint')

flags.DEFINE_string(name='model_dir', default='./out/dcn', help='模型存储的位置')
flags.DEFINE_string(name='feature_dir', default='./dat/ml-1m', help='特征存放的位置')

flags.DEFINE_boolean(name='mk_feature', default=False, help='是否需要处理特征')
flags.DEFINE_enum(name='model_type', default='deepfm', enum_values=['deepfm', 'dcn', 'esmm'], help='支持的模型')
