import tensorflow as tf

ckpt = r"/home/shuming/double_net_distillation/double_net/double_net.ckpt-15000"
# read node name way 1
with tf.Session() as sess:
    saver = tf.train.import_meta_graph(ckpt + '.meta', clear_devices=True)
    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
    node_list = [n.name for n in graph_def.node]
    for node in node_list:
        if 'nms' in node:
            print("node_name", node)
