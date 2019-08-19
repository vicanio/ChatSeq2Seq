# encoding: utf-8

import DataProcessing
import os
import tensorflow as tf
from SequenceToSequence import Seq2Seq
from tqdm import tqdm
import numpy as np
from CONFIG import BASE_MODEL_DIR, MODEL_NAME, data_config, model_config,PB_PB_DIR, \
    n_epoch, batch_size, keep_prob
from tensorflow.python.framework import graph_util

# 是否在原有模型的基础上继续训练
continue_train = True


def train():
    """
    训练模型
    :return:
    """
    du = DataProcessing.DataUnit(**data_config)
    save_path = os.path.join(BASE_MODEL_DIR, MODEL_NAME)
    steps = int(len(du) / batch_size) + 1

    # 创建session的时候设置显存根据需要动态申请
    tf.reset_default_graph()
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True

    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            # 定义模型
            model = Seq2Seq(batch_size=batch_size,
                            encoder_vocab_size=du.vocab_size,
                            decoder_vocab_size=du.vocab_size,
                            mode='train',
                            **model_config)

            init = tf.global_variables_initializer()
            writer=tf.summary.FileWriter('./graph/nlp',sess.graph)
            sess.run(init)
            if continue_train:
                model.load(sess, save_path)
            model.export(sess)
            '''
            for epoch in range(1, n_epoch + 1):
                costs = []
                bar = tqdm(range(steps), total=steps,
                           desc='epoch {}, loss=0.000000'.format(epoch))
                for _ in bar:
                    x, xl, y, yl = du.next_batch(batch_size)
                    max_len = np.max(yl)
                    y = y[:, 0:max_len]
                    cost, lr = model.train(sess, x, xl, y, yl, keep_prob)
                    costs.append(cost)
                    bar.set_description('epoch {} loss={:.6f} lr={:.6f}'.format(epoch, np.mean(costs), lr))
                model.save(sess, save_path=save_path)
                '''

def savepb():
    metapath = os.path.join(BASE_MODEL_DIR, "chatbot_model.ckpt.meta")
    save_path = os.path.join(BASE_MODEL_DIR, PB_PB_DIR)
    tf.reset_default_graph()
    config = tf.ConfigProto()
    output_node_names = "global_step,global_step/Assign"
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    saver = tf.train.import_meta_graph(metapath)
    with tf.Session(config=config) as sess:
        saver.restore(sess, tf.train.latest_checkpoint('model/'))
        print("Restoring Done .. ")
        print("Saving the model to Protobuf format: ", save_path)

        tf.train.write_graph(sess.graph_def, save_path, "Binary_Protobuf.pb", False)
        tf.train.write_graph(sess.graph_def, save_path, "Text_Protobuf.pbtxt", True)
    print("Saving Done .. ")

def freeze_graph():
    save_path = "./model.pb"
    # 检查目录下ckpt文件状态是否可用
    checkpoint = tf.train.get_checkpoint_state('model/')
    input_checkpoint = checkpoint.model_checkpoint_path
    output_node_names = "init,save/restore_all" #nodename,一定要在graph内
    clear_devices = True
    # We import the meta graph and retrieve a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            input_graph_def,  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(save_path, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

        for op in graph.get_operations():
            print(op.name, op.values())

        print("output_graph:", save_path)
        print("all done")

def printNode():
    from tensorflow.python import pywrap_tensorflow
    checkpoint_path = os.path.join( "D:\code\Seq2SeqModel\model\chatbot_model.ckpt")
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("tensor_name: ", key)

def input_feed():
    encoder_inputs = tf.placeholder(tf.int32, shape=[batch_size, None], name='encoder_inputs')
    encoder_inputs_length = tf.placeholder(tf.int32, shape=[batch_size, ], name='encoder_inputs_length')
    keep_prob = tf.placeholder(tf.float32, shape=(), name='keep_prob')
    decoder_inputs = tf.placeholder(tf.int32, shape=[batch_size, None], name='decoder_inputs')
    decoder_inputs_length = tf.placeholder(tf.int32, shape=[batch_size, ],
                                                    name='decoder_inputs_length')
    input_feed = {encoder_inputs.name: encoder_inputs,
                  encoder_inputs_length.name: encoder_inputs_length}
    input_feed[keep_prob.name] = keep_prob
    input_feed[decoder_inputs.name] = decoder_inputs
    input_feed[decoder_inputs_length.name] = decoder_inputs_length
    return tf.estimator.export.build_raw_serving_input_receiver_fn(input_feed)


def savedmodel():
    export_dir = os.path.join(BASE_MODEL_DIR, "saved_model")
    tf.estimator.Estimator.export_savedmodel(export_dir_base=export_dir, serving_input_receiver_fn=input_feed(),strip_default_attrs=True)
    print("all done")

if __name__ == '__main__':
    #freeze_graph()
    train()
    #savedmodel()
