import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(name='model_config_file', default='', help='模型配置文件位置')

flags.DEFINE_integer(name='epoch_num', default=1, help='epoch数量')
flags.DEFINE_integer(name='batch_size', default=16, help='batch 大小')
flags.DEFINE_float(name='init_lr', default=0.0001, help='初始学习率')

flags.DEFINE_integer(name='save_checkpoints_steps', default=1024, help='每隔多少步保存模型')
flags.DEFINE_integer(name='keep_checkpoint_max', default=3, help='最多保存多少checkpoint')

flags.DEFINE_string(name='model_dir', default='', help='模型存储的位置')
flags.DEFINE_string(name='feature_dir', default='', help='特征存放的位置')

flags.DEFINE_boolean(name='mk_feature', default=True, help='是否需要处理特征')
