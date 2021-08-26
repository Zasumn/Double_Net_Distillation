import tensorflow as tf
import numpy as np
from tensorflow.python import pywrap_tensorflow

model_reader = pywrap_tensorflow.NewCheckpointReader(r"/home/shuming/double_net_distillation/double_net/double_net.ckpt-30000")

#Convert the reader to a dict like form
var_dict = model_reader.get_variable_to_shape_map()

#Print out the loop
for key in var_dict:
    print("variable name: ", key)
        #print(model_reader.get_tensor(key))
