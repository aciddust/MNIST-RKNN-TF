'''
rk_predict.py, tf_predict.py를 실행하여 tf 모델, rknn 모델 및 양자화 된 rknn 모델의 실행 결과를 비교할 수 있음.
'''

#! -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
origin_test = mnist.test.images
reshape_test = []
for t in origin_test:
    b = t.reshape(28,28)
    reshape_test.append(b)
for length in [100,500,1000,10000]:
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        output_graph_path = './mnist_frozen_graph.pb'

        with open(output_graph_path, 'rb') as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")
     
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            input = sess.graph.get_tensor_by_name("x:0")
            output = sess.graph.get_tensor_by_name("y_conv:0")
            y_conv_2 = sess.run(output, feed_dict={input:reshape_test[0:length]})
            y_2 = mnist.test.labels[0:length]
            print("first image:",y_conv_2[0])
            correct_prediction_2 = tf.equal(tf.argmax(y_conv_2, 1), tf.argmax(y_2, 1))
            accuracy_2 = tf.reduce_mean(tf.cast(correct_prediction_2, "float"))
            print(accuracy_2)
            print('%d:'%length,"check accuracy %g" % sess.run(accuracy_2))