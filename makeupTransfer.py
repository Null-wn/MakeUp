# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
from imageio import imread, imsave
import cv2


def preprocess(img):
    return (img / 255. - 0.5) * 2

def deprocess(img):
    return (img + 1) / 2

def makeup_transfer(before_makeup_path,after_makeup_path):
    img_size = 256
    no_makeup = cv2.resize(imread(before_makeup_path), (img_size, img_size))
    X_img = np.expand_dims(preprocess(no_makeup), 0)   #增加维度
    makeup = cv2.resize(imread(after_makeup_path), (img_size, img_size))
    Y_img = np.expand_dims(preprocess(makeup), 0)

    result = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
    result = deprocess(result)

    imsave('result.jpg', result[0])
    return 
  
    
#tf.reset_default_graph()
tf.compat.v1.reset_default_graph()
#sess = tf.Session()
sess = tf.compat.v1.Session()
#sess.run(tf.global_variables_initializer())
tf.compat.v1.disable_eager_execution()
sess.run(tf.compat.v1.global_variables_initializer())

saver = tf.compat.v1.train.import_meta_graph(os.path.join('model', 'model.meta'))
saver.restore(sess, tf.train.latest_checkpoint('model'))

graph = tf.compat.v1.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')