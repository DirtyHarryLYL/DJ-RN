from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ult.config import cfg
from ult.ult import Get_Next_Instance_HO_Neg_HICO_3D
from ult.timer import Timer

import cPickle as pickle
import numpy as np
import os
import sys
import glob
import time

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.training.learning_rate_decay import cosine_decay_restarts

class SolverWrapper(object):
    """
    A wrapper class for the training process
    """

    def __init__(self, sess, network, Trainval_GT, Trainval_N, output_dir, tbdir, Pos_augment, Neg_select, Restore_flag, pretrained_model):

        self.net               = network
        self.Trainval_GT       = Trainval_GT
        self.Trainval_N        = Trainval_N
        self.output_dir        = output_dir
        self.tbdir             = tbdir
        self.Pos_augment       = Pos_augment
        self.Neg_select        = Neg_select
        self.Restore_flag      = Restore_flag
        self.pretrained_model  = pretrained_model

    def snapshot(self, sess, iter):

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        filename = 'HOI' + '_iter_{:d}'.format(iter) + '.ckpt'
        filename = os.path.join(self.output_dir, filename)
        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

    def construct_graph(self, sess):
        with sess.graph.as_default():

            tf.set_random_seed(cfg.RNG_SEED)

            layers = self.net.create_architecture(True)

            loss = layers['total_loss']

            if cfg.TRAIN_MODULE_CONTINUE == 1:
                path_iter = self.pretrained_model.split('.ckpt')[0]
                iter_num = path_iter.split('_')[-1]
                global_step    = tf.Variable(int(iter_num), trainable=False)

            if cfg.TRAIN_MODULE_CONTINUE == 2:
                global_step    = tf.Variable(0, trainable=False)

            first_decay_steps = 2 * len(self.Trainval_GT)
            lr = cosine_decay_restarts(cfg.TRAIN.LEARNING_RATE * 10, global_step, first_decay_steps, t_mul=cfg.LR_DECAY.T_MUL, m_mul=cfg.LR_DECAY.M_MUL, alpha=0.0) 

            self.optimizer = tf.train.GradientDescentOptimizer(lr)

            list_var_to_update = []
            if cfg.TRAIN_MODULE_UPDATE == 1:
                list_var_to_update = tf.trainable_variables()
            elif cfg.TRAIN_MODULE_UPDATE == 2:
                list_var_to_update = [var for var in tf.trainable_variables() if 'Att_sp' in var.name\
                                                                              or 'bottleneck_sp' in var.name\
                                                                              or 'conv1_sp' in var.name\
                                                                              or 'conv2_sp' in var.name\
                                                                              or 'Concat_SHsp' in var.name\
                                                                              or 'fc7_SHsp' in var.name\
                                                                              or 'body_to_head' in var.name\
                                                                              or 'attention_3D' in var.name\
                                                                              or 'triplet_align' in var.name\
                                                                              or 'space_classification' in var.name\
                                                                              or 'joint_classification' in var.name]

            grads_and_vars = self.optimizer.compute_gradients(loss, list_var_to_update)
            capped_gvs     = [(tf.clip_by_norm(grad, 1.), var) for grad, var in grads_and_vars]

            train_op = self.optimizer.apply_gradients(capped_gvs,global_step=global_step)
            self.saver = tf.train.Saver(max_to_keep=cfg.TRAIN.SNAPSHOT_KEPT)
            self.writer = tf.summary.FileWriter(self.tbdir, sess.graph)

        return lr, train_op

    def from_snapshot(self, sess):

        if self.Restore_flag == 0:
            saver_t  = [var for var in tf.model_variables() if 'conv1' in var.name and 'conv1_sp' not in var.name]
            saver_t += [var for var in tf.model_variables() if 'conv2' in var.name and 'conv2_sp' not in var.name]
            saver_t += [var for var in tf.model_variables() if 'conv3' in var.name]
            saver_t += [var for var in tf.model_variables() if 'conv4' in var.name]
            saver_t += [var for var in tf.model_variables() if 'conv5' in var.name]
            saver_t += [var for var in tf.model_variables() if 'shortcut' in var.name]

            print('Restoring model snapshots from {:s}'.format(self.pretrained_model))

            self.saver_restore = tf.train.Saver(saver_t)
            self.saver_restore.restore(sess, self.pretrained_model)

            for var in tf.trainable_variables():
                print(var.name, var.eval().mean())

        if self.Restore_flag == 5 or self.Restore_flag == 6 or self.Restore_flag == 7:

            print('Restoring model snapshots from {:s}'.format(self.pretrained_model))
            saver_t = {}
            
            for ele in tf.model_variables():
                if 'resnet_v1_50/conv1/weights' in ele.name or 'resnet_v1_50/conv1/BatchNorm/beta' in ele.name or 'resnet_v1_50/conv1/BatchNorm/gamma' in ele.name or 'resnet_v1_50/conv1/BatchNorm/moving_mean' in ele.name or 'resnet_v1_50/conv1/BatchNorm/moving_variance' in ele.name:
                    saver_t[ele.name[:-2]] = ele
            for ele in tf.model_variables():
                if 'block1' in ele.name:
                    saver_t[ele.name[:-2]] = ele
            for ele in tf.model_variables():
                if 'block2' in ele.name:
                    saver_t[ele.name[:-2]] = ele
            for ele in tf.model_variables():
                if 'block3' in ele.name:
                    saver_t[ele.name[:-2]] = ele
            for ele in tf.model_variables():
                if 'block4' in ele.name:
                    saver_t[ele.name[:-2]] = ele
            
            self.saver_restore = tf.train.Saver(saver_t)
            self.saver_restore.restore(sess, self.pretrained_model)
            
            if self.Restore_flag >= 5:
                saver_t = {}
                for ele in tf.model_variables():
                    if 'block4' in ele.name:
                        saver_t[ele.name[:-2]] = [var for var in tf.model_variables() if ele.name[:-2].replace('block4','block5') in var.name][0]
                self.saver_restore = tf.train.Saver(saver_t)
                self.saver_restore.restore(sess, self.pretrained_model)
            
            if self.Restore_flag >= 6:
                saver_t = {}
                for ele in tf.model_variables():
                    if 'block4' in ele.name:
                        saver_t[ele.name[:-2]] = [var for var in tf.model_variables() if ele.name[:-2].replace('block4','block6') in var.name][0]
                self.saver_restore = tf.train.Saver(saver_t)
                self.saver_restore.restore(sess, self.pretrained_model)
    
            if self.Restore_flag >= 7:
                saver_t = {}
                for ele in tf.model_variables():
                    if 'block4' in ele.name:
                        saver_t[ele.name[:-2]] = [var for var in tf.model_variables() if ele.name[:-2].replace('block4','block7') in var.name][0]
                self.saver_restore = tf.train.Saver(saver_t)
                self.saver_restore.restore(sess, self.pretrained_model)

    def from_previous_ckpt(self,sess):
        print('Restoring model snapshots from {:s}'.format(self.pretrained_model))
        saver_t = {}
        saver_t  = tf.model_variables()
        self.saver_restore = tf.train.Saver(saver_t)
        self.saver_restore.restore(sess, self.pretrained_model)

    def from_best_trained_model(self, sess):

        for var in tf.trainable_variables(): # trainable weights, we need surgery
            print(var.name, var.eval().mean())

        print('Restoring model snapshots from {:s}'.format(self.pretrained_model))
        saver_t = {}

        saver_t  = [var for var in tf.model_variables() if 'Att_sp' not in var.name\
                                                        and 'bottleneck_sp' not in var.name\
                                                        and 'conv1_sp' not in var.name\
                                                        and 'conv2_sp' not in var.name\
                                                        and 'Concat_SHsp' not in var.name\
                                                        and 'fc7_SHsp' not in var.name\
                                                        and 'body_to_head' not in var.name\
                                                        and 'attention_3D' not in var.name\
                                                        and 'triplet_align' not in var.name\
                                                        and 'space_classification' not in var.name\
                                                        and 'joint_classification' not in var.name\
                                                        ]
        self.saver_restore = tf.train.Saver(saver_t)
        self.saver_restore.restore(sess, self.pretrained_model)
        print("the variables is being trained now \n")

    def train_model(self, sess, max_iters):
        lr, train_op = self.construct_graph(sess)

        sess.run(tf.global_variables_initializer())
        if cfg.TRAIN_MODULE_CONTINUE == 1:
            self.from_previous_ckpt(sess)
        else:
            if cfg.TRAIN_INIT_WEIGHT == 1:
                self.from_snapshot(sess)
            elif cfg.TRAIN_INIT_WEIGHT == 2:
                self.from_previous_ckpt(sess) 
            elif cfg.TRAIN_INIT_WEIGHT == 3:
                self.from_best_trained_model(sess) 

        sess.graph.finalize()

        timer = Timer()
        Data_length = len(self.Trainval_GT)
        keys = self.Trainval_GT.keys()
        idx = range(Data_length)

        if cfg.TRAIN_MODULE_CONTINUE == 2:
            iter = 0
        elif cfg.TRAIN_MODULE_CONTINUE == 1:
            path_iter = self.pretrained_model.split('.ckpt')[0]
            iter_num = path_iter.split('_')[-1]
            iter = int(iter_num) + 1

        while iter < max_iters + 1:
            timer.tic()
            if iter % Data_length == 0:
                np.random.shuffle(idx)
            image_id = keys[idx[iter % Data_length]]

            blobs = Get_Next_Instance_HO_Neg_HICO_3D(self.Trainval_GT, self.Trainval_N, image_id, self.Pos_augment, self.Neg_select)

            if (iter % cfg.TRAIN.SUMMARY_INTERVAL == 0) or (iter < 20):
                total_loss, summary = self.net.train_step_with_summary(sess, blobs, lr.eval(), train_op)
                self.writer.add_summary(summary, float(iter))
            else:
                total_loss = self.net.train_step(sess, blobs, lr.eval(), train_op)
            del blobs
            timer.toc()

            if iter % (cfg.TRAIN.DISPLAY) == 0:
                print('iter: %d / %d, im_id: %6u, total loss: %.6f, lr: %f, speed: %.3f s/iter' % \
                            (iter, max_iters, image_id, total_loss, lr.eval(), timer.average_time))

            if (iter % cfg.TRAIN.SNAPSHOT_ITERS == 0 and iter != 0) or (iter == 10):
                self.snapshot(sess, iter)

            iter += 1

        self.writer.close()


def train_net(network, Trainval_GT, Trainval_N, output_dir, tb_dir, Pos_augment, Neg_select, Restore_flag, pretrained_model, max_iters=300000):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
        
    if cfg.TRAIN_MODULE_CONTINUE == 2:
        filelist = [ f for f in os.listdir(tb_dir)]
        for f in filelist:
            os.remove(os.path.join(tb_dir, f))
        filelist = [ f for f in os.listdir(output_dir)]
        for f in filelist:
            os.remove(os.path.join(output_dir, f))                
        
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:
        sw = SolverWrapper(sess, network, Trainval_GT, Trainval_N, output_dir, tb_dir, Pos_augment, Neg_select, Restore_flag, pretrained_model)
        print('Solving..., Pos augment = ' + str(Pos_augment) + ', Neg augment = ' + str(Neg_select) + ', Restore_flag = ' + str(Restore_flag))
        sw.train_model(sess, max_iters)
        print('done solving')
