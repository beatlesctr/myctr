# 实验配置



|  Experiment ID  | Learning rate  | Batch size | Epoch | Cross config | Deep config |auc|备注|
|  ----  | ----  | ---- | ---- | ---- | ---- | ---- | ---- |
|0 | 0.001 | 128 | 10 | "cross_layer_num":2 | "dnn_layer_cfg":[256, 256, 256] |0.832|根据时间戳分割样本集|
|0 | 0.001 | 512 | 10 | "cross_layer_num":2 | "dnn_layer_cfg":[256, 256, 256] |0.828|根据时间戳分割样本集|
|0 | 0.001 | 1024 | 10 | "cross_layer_num":2 | "dnn_layer_cfg":[256, 256, 256] |0.902|随机分割样本集|



