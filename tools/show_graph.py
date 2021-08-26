import tensorflow as tf

g = tf.Graph()
with g.as_default() as g:
    tf.train.import_meta_graph('/home/shuming/double_net_distillation/double_net/double_net.ckpt-15000.meta')

with tf.Session(graph=g) as sess:
    file_writer = tf.summary.FileWriter(logdir='./your_out_file', graph=g)
