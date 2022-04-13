import tensorflow.compat.v1 as tf
from tensorflow.python import pywrap_tensorflow
 
#saver = tf.train.Saver()
reader = pywrap_tensorflow.NewCheckpointReader('./models/densefuse_gray/densefuse_model_bs2_epoch4_all_weight_1e0.ckpt')#model_checkpoint_path是保存模型的路径加上模型名
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor name:",key)
    saver.restore(sess, './models/densefuse_gray/densefuse_model_bs2_epoch4_all_weight_1e0.ckpt')