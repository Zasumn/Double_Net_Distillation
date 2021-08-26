import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # 使用 GPU 1
trained_checkpoint_prefix = r"/home/shuming/double_net_distillation/double_net/double_net.ckpt-10000"
#trained_checkpoint_prefix = r"/home/shared_data2/eventcamera/exper/darkpoint_coco/model.ckpt-600000"
export_dir = '/home/shuming/DnD7'
graph = tf.Graph()
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
with tf.compat.v1.Session(graph=graph, config=config) as sess:
    # Restore from checkpoint
    loader = tf.compat.v1.train.import_meta_graph(trained_checkpoint_prefix + '.meta')
    loader.restore(sess, trained_checkpoint_prefix)
    # Export checkpoint to SavedModel
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.SERVING],
                                         strip_default_attrs=True)
    builder.save()