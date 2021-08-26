import tensorflow as tf
weights_dir = '/home/shared_data2/eventcamera/exper/saved_models/sp_v6'

graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    tf.saved_model.loader.load(sess,
                               [tf.saved_model.tag_constants.SERVING],
                               str(weights_dir))
    tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]


    for i in tensor_name_list:
        #if 'kernel' in i:
        print(i)