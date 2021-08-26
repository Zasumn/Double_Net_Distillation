import tensorflow as tf
from tensorflow.python.framework import graph_util

def freeze_graph(input_checkpoint, output_node_names, output_graph):
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        for n in tf.get_default_graph().as_graph_def().node:
            print(n.name)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=tf.get_default_graph().as_graph_def(),
            output_node_names=output_node_names.split(","))
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

if __name__ == '__main__':
    ## ----rnet----
    input_checkpoint = r"/home/shuming/double_net_distillation/double_net/double_net.ckpt-30000"
    out_pb_path = "../DND.pb"
    output_node_names = "student_detector/transpose_2,student_detector/Squeeze,student_descriptor/transpose_1,student_descriptor/l2_normalize"
    freeze_graph(input_checkpoint, output_node_names, out_pb_path)
