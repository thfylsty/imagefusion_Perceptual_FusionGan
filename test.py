# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import scipy.misc
import time
import os
import glob
import cv2
from tensorflow.python.keras import backend as K
#reader = tf.train.NewCheckpointReader("./checkpoint_20/CGAN_120/CGAN.model-9")
os.environ['CUDA_VISIBLE_DEVICES']='3'
# os.environ['CUDA_VISIBLE_DEVICES']='0'
log_device_placement=True
allow_soft_placement=True
tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.99
config.gpu_options.allow_growth = True
K.set_session(tf.Session(graph=tf.get_default_graph(),config=config))

def imread(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:

    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def imsave(image, path):
  return scipy.misc.imsave(path, image)


def prepare_data(dataset):
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data = glob.glob(os.path.join(data_dir, "*.tif"))
    data.extend(glob.glob(os.path.join(data_dir, "*.bmp")))
    data.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
    # data.sort(key=lambda x:int(x[len(data_dir)+1:-4]))

    return data

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def fusion_model(img):
    with tf.variable_scope('fusion_model'):
        with tf.variable_scope('layer1'):
            weights=tf.get_variable("w1",initializer=tf.constant(reader.get_tensor('fusion_model/layer1/w1')))
            bias=tf.get_variable("b1",initializer=tf.constant(reader.get_tensor('fusion_model/layer1/b1')))
            conv1_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(img, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1_ir = lrelu(conv1_ir)
        with tf.variable_scope('layer2'):
            weights=tf.get_variable("w2",initializer=tf.constant(reader.get_tensor('fusion_model/layer2/w2')))
            bias=tf.get_variable("b2",initializer=tf.constant(reader.get_tensor('fusion_model/layer2/b2')))
            vivi = tf.concat([images_vi, images_vi], axis=-1)
            conv2_add =tf.concat([vivi,conv1_ir],axis=-1)
            # conv2_add =conv1_ir  #without add
            # conv2_add = tf.concat([vivi, conv1_ir], axis=-1)
            conv2_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2_add, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2_ir = lrelu(conv2_ir)
        with tf.variable_scope('layer3'):
            weights=tf.get_variable("w3",initializer=tf.constant(reader.get_tensor('fusion_model/layer3/w3')))
            bias=tf.get_variable("b3",initializer=tf.constant(reader.get_tensor('fusion_model/layer3/b3')))
            conv3_add = tf.concat([conv2_add, conv2_ir], axis=-1)
            # conv3_add = tf.concat([vivi, conv2_ir], axis=-1)
            conv3_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv3_add, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3_ir = lrelu(conv3_ir)
        with tf.variable_scope('layer4'):
            weights=tf.get_variable("w4",initializer=tf.constant(reader.get_tensor('fusion_model/layer4/w4')))
            bias=tf.get_variable("b4",initializer=tf.constant(reader.get_tensor('fusion_model/layer4/b4')))
            conv4_add = tf.concat([conv3_add, conv3_ir], axis=-1)
            # conv4_add = tf.concat([vivi, conv3_ir], axis=-1)
            conv4_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv4_add, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4_ir = lrelu(conv4_ir)
        with tf.variable_scope('layer5'):
            weights=tf.get_variable("w5",initializer=tf.constant(reader.get_tensor('fusion_model/layer5/w5')))
            bias=tf.get_variable("b5",initializer=tf.constant(reader.get_tensor('fusion_model/layer5/b5')))
            conv5_add = tf.concat([conv4_add, conv4_ir], axis=-1)
            # conv5_add = tf.concat([vivi, conv4_ir], axis=-1)
            conv5_ir= tf.nn.conv2d(conv5_add, weights, strides=[1,1,1,1], padding='VALID') + bias
            conv5_ir=tf.nn.tanh(conv5_ir)
    return conv5_ir




def input_setup(index):
    padding=6
    sub_ir_sequence = []
    sub_vi_sequence = []
    input_ir=(imread(data_ir[index])-127.5)/127.5

    w,h=input_ir.shape
    input_ir=input_ir.reshape([w,h,1])
    input_vi=(imread(data_vi[index])-127.5)/127.5

    w,h=input_vi.shape
    input_vi=input_vi.reshape([w,h,1])
    sub_ir_sequence.append(input_ir)
    sub_vi_sequence.append(input_vi)
    train_data_ir= np.asarray(sub_ir_sequence)
    train_data_vi= np.asarray(sub_vi_sequence)
    return train_data_ir,train_data_vi


num_epoch=17
path = '_120'
while(num_epoch<=24):
    print("num_epoch",num_epoch)
    reader = tf.train.NewCheckpointReader('./CGAN'+path+'/CGAN.model-'+ str(num_epoch))
    print('./checkpoint_20/CGAN'+path+'/CGAN.model-'+ str(num_epoch))
    # reader = tf.train.NewCheckpointReader('./checkpoint_20/CGAN_100_onlyadd/CGAN.model-' + str(num_epoch))

    with tf.name_scope('IR_input'):

        images_ir = tf.placeholder(tf.float32, [1,None,None,None], name='images_ir')
    with tf.name_scope('VI_input'):

        images_vi = tf.placeholder(tf.float32, [1,None,None,None], name='images_vi')
        #self.labels_vi_gradient=gradient(self.labels_vi)

    with tf.name_scope('input'):
        #resize_ir=tf.image.resize_images(images_ir, (512, 512), method=2)
        input_image=tf.concat([images_ir,images_vi],axis=-1)
    with tf.name_scope('fusion'):
        fusion_image=fusion_model(input_image)

    with tf.Session() as sess:
        init_op=tf.global_variables_initializer()
        sess.run(init_op)
        data_ir=prepare_data('Test_ir')
        data_vi=prepare_data('Test_vi')
        # data_ir = prepare_data('road/ir')
        # data_vi = prepare_data('road/vi')
        for i in range(len(data_ir)):
            start=time.time()
            train_data_ir,train_data_vi=input_setup(i)
            result =sess.run(fusion_image,feed_dict={images_ir: train_data_ir,images_vi: train_data_vi})
            result=result*127.5+127.5
            result = result.squeeze()
            # image_path = os.path.join(os.getcwd(), 'result/grd'+path,'epoch'+str(num_epoch))
            image_path = './result/'+'epoch'+str(num_epoch)
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            image_path = os.path.join(image_path,str(i+1)+".bmp")
            end=time.time()
            # print(out.shape)
            imsave(result, image_path)
            print("Testing [%d] success,Testing time is [%f]"%(i,end-start))
    tf.reset_default_graph()
    num_epoch=num_epoch+1
