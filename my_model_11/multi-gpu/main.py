import tensorflow as tf
import numpy as np
import argparse
import datetime
import logging
import time
import os
from models import Generator, Discriminator, flownet, initialize_flownet
from config_reader import Config
import dataset_util
from train_util import init_or_restore, average_gradients, average_losses
from test_util import compute_auc, compute_eer, plot_score, plot_event, plot_heatmap
from loss_functions import compute_flow_loss, compute_intensity_loss, compute_gradient_loss, \
    compute_psnr, compute_mse_error, compute_reconstr_loss, compute_pixel_loss
from tvnet.tvnet_model import TVNet

"""HYPER-PARAMETER"""
minima_threshold = 0.25
window_length = 50

use_trick = 1
for_remote = 0
for_pc = 1
use_rgb = 1
file_name = 'myModel_11'

#if epsilon is too big, training of DCGAN is failure.
epsilon = 1e-14 

# adam beta1 default is 0.9, wgan-gp, SNGAN try 0.5?
beta1 = 0.9

# 2e-4 1e-4
LRATE_G = [0.0002, 0.0002]
LRATE_G_BOUNDARIES = [20000]

LRATE_D = [0.0002, 0.0002]
LRATE_D_BOUNDARIES = [20000]
# # For rgb color scale video,
# # such as avenue, learning rate of G and D star from 2e-4 and 2e-5, respectively.
# LRATE_G = [0.0002, 0.00002]
# LRATE_G_BOUNDARIES = [100000]
#
# LRATE_D = [0.00002, 0.000002]
# LRATE_D_BOUNDARIES = [100000]

# # For gray scale video, such as Ped2 and Ped1, learning rate of G and D star from 1e-4 and 1e-5, respectively.
# LRATE_G = [0.0001, 0.00001]
# LRATE_G_BOUNDARIES = [7000]
#
# LRATE_D = [0.00001, 0.000001]
# LRATE_D_BOUNDARIES = [7000]

# for lp loss. e.g, 1 or 2 for l1 and l2 loss, respectively)
l_num = 2
# the power to which each gradient term is raised in GDL loss
alpha_num = 1
# the percentage of the lp loss to use in the combined loss,
# we found in smaller lp is slightly better in avenue,
# but not too much difference
# LAM_LP = 1 is 84.9, LAM_LP = 0.001 may arrive to 85.1
# lam_lp = 0.001
lam_lp = 1
# lam_gdl = 1
lam_gdl = 1
# the percentage of the adversarial loss to use in the combined loss
# lam_adv = 0.05
lam_adv = 0.05
# the percentage of the different frame loss
# lam_flow = 2
lam_flow = 2
adversarial = (lam_adv != 0)
# two stream
lam_two_stream = 1
use_tvnet = 1
GAN_type = 'SNGAN'
use_skip = 1
assert use_rgb in [0, 1],'[!!!] wrong use_rgb'

assert GAN_type in ['LSGAN', 'WGAN', 'WGAN-GP', 'SNGAN'], 'wrong GAN type choice!'

if use_tvnet != 0:
    tv_scale = 1
    tv_warp = 1
    tv_iter = 50
else:
    tv_scale = 0
    tv_warp = 0
    tv_iter = 0   

if for_remote == 1:
    """for remote"""
    """file_name need to be modified"""
    dataset_root_dir = '/data/zyl/graduate'
    exp_data_root_dir = '/data/zyl/graduate/exp_data/' + file_name
    npy_root_dir = dataset_root_dir + '/cgan_data/npy_dataset'
    tfrecord_root_dir = dataset_root_dir + '/cgan_data/tfrecord_dataset'
    gt_root_dir = dataset_root_dir + '/cgan_data'
    checkpoint_dir = exp_data_root_dir + '/training_checkpoints'
    summary_dir = exp_data_root_dir + '/summary'
    regularity_score_root_dir = exp_data_root_dir + '/regularity_score'
    log_path = exp_data_root_dir + '/log'
    FLOWNET_CHECKPOINT = dataset_root_dir + '/cgan_data/pretrained_flownet/flownet-SD.ckpt-0'
    gpu_nums = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
else:
    if for_pc == 1:
        """for local"""
        """file_name need to be modified"""
        dataset_root_dir = '/home/orli/Blue-HDD/1_final_lab_/Dataset'
        exp_data_root_dir = '/home/orli/Blue-HDD/1_final_lab_/exp_data/' + file_name
        npy_root_dir = dataset_root_dir + '/cgan_data/npy_dataset'
        tfrecord_root_dir = dataset_root_dir + '/cgan_data/tfrecord_dataset'
        gt_root_dir = dataset_root_dir + '/cgan_data'
        checkpoint_dir = exp_data_root_dir + '/training_checkpoints'
        summary_dir = exp_data_root_dir + '/summary'
        regularity_score_root_dir = exp_data_root_dir + '/regularity_score'
        log_path = exp_data_root_dir + '/log'
        FLOWNET_CHECKPOINT = dataset_root_dir + '/cgan_data/pretrained_flownet/flownet-SD.ckpt-0'
        gpu_nums = 1
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    else:
        """for notebook"""
        dataset_root_dir = 'D:/study/337_lab/thesis/data'
        exp_data_root_dir = 'D:/study/337_lab/thesis/exp_data/' + file_name
        npy_root_dir = dataset_root_dir + '/cgan_data/npy_dataset'
        tfrecord_root_dir = dataset_root_dir + '/cgan_data/tfrecord_dataset'
        gt_root_dir = dataset_root_dir + '/cgan_data'
        checkpoint_dir = exp_data_root_dir + '/training_checkpoints'
        summary_dir = exp_data_root_dir + '/summary'
        regularity_score_root_dir = exp_data_root_dir + '/regularity_score'
        log_path = exp_data_root_dir + '/log'
        FLOWNET_CHECKPOINT = dataset_root_dir + '/cgan_data/pretrained_flownet/flownet-SD.ckpt-0'
        gpu_nums = 1
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"


# 按照PCI_BUS_ID顺序从0开始排列GPU设备
os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"


parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--dataset', dest='dataset', default='avenue', help='dataset name')
args = parser.parse_args()

if use_rgb == 1:
    assert args.dataset not in ['ped1', 'ped2'], '[!!!] Dataset:{} can not use RGB mode'.format(args.dataset)
    
"""
Train phase
"""
def train(cfg, logger, model_name):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model_dir = os.path.join(checkpoint_dir, model_name)
    print('[!!!] model name:{}'.format(model_dir))
    logger.info('[!!!] model name:{}'.format(model_dir))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    ckpt_path = os.path.join(model_dir, model_name)
    best_model_ckpt_dir = os.path.join(model_dir, 'best_model')
    
    if not os.path.exists(best_model_ckpt_dir):
        os.makedirs(best_model_ckpt_dir)
    best_ckpt_path = os.path.join(best_model_ckpt_dir, 'best_model')
    
    trick_model_ckpt_dir = os.path.join(model_dir, 'trick_model')
    if not os.path.exists(trick_model_ckpt_dir):
        os.makedirs(trick_model_ckpt_dir)
    trick_ckpt_path = os.path.join(trick_model_ckpt_dir, 'trick_model')
    
    with tf.device('/cpu:0'):
        g_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='g_step')
        # # lr = tf.train.polynomial_decay(hparams.learning_rate, global_step, 100000)
        # if cfg.opt == 'Adam':
        #     opt = tf.train.AdamOptimizer()
        # else:
        #     opt = tf.train.GradientDescentOptimizer()

        # 分段设置学习率，根据传入的step，自行更改返回的学习率tensor的值
        g_lrate = tf.train.piecewise_constant(g_step, boundaries=LRATE_G_BOUNDARIES, values=LRATE_G)
        g_optimizer = tf.train.AdamOptimizer(learning_rate=g_lrate, beta1 = beta1, name='g_optimizer')

        if adversarial:
            # training discriminator
            # 变量保存global_step:d_step
            d_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='d_step')
            d_lrate = tf.train.piecewise_constant(d_step,
                                                  boundaries=LRATE_D_BOUNDARIES,
                                                  values=LRATE_D)
            d_optimizer = tf.train.AdamOptimizer(learning_rate=d_lrate, beta1 = beta1,
                                                 name='d_optimizer')
        else:
            d_step = None
            d_optimizer = None


        small_batch = cfg.batch_size // gpu_nums

        # validation_split = 0.15
        dataset = dataset_util.Dataset(args.dataset, cfg, small_batch, use_rgb=use_rgb,
                                       tfrecord_root_dir=tfrecord_root_dir, logger=logger,
                                       validation_split=0.15)
        train_iter = dataset.get_train_iterator()
        val_iter = dataset.get_val_iterator()
        assert train_iter is not None and val_iter is not None
        # generator_tower_grads
        g_tower_grads = []
        # discriminator_tower_grads
        d_tower_grads = []
        # generator losses
        g_losses = []
        # discriminator losses
        d_losses = []

        is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
        batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        for i in range(gpu_nums):
            with tf.device('/gpu:{}'.format(i)):
                with tf.name_scope('tower_{}'.format(i)) as scope:
                    train_volumes = dataset.get_next_volume('train')
                    val_volumes = dataset.get_next_volume('val')
                    data = tf.cond(is_training, lambda: train_volumes, lambda: val_volumes)
                    # 前 clip_lenth-1 帧 [batch, clip_length-1, h, w, c]
                    input = data[:,:-1,...]
                    # 最后一帧， [batch, h, w, c]
                    input_gt = tf.squeeze(data[:,-1:-2:-1,...], axis=[1])
                    # with tf.variable_scope('generator', reuse=None):
                    with tf.variable_scope('generator', reuse=(i>0)):
                        generator = Generator(cfg, input, batch_size, use_rgb, use_skip)
                        # [batch, h, w, c]
                        pred_output = generator._pred_out
                        if lam_two_stream != 0:
                            reconstr_output = generator._reconstr_out

                    # define reconstruct loss
                    if lam_two_stream != 0:
                        reconstr_loss = compute_reconstr_loss(input, reconstr_output)
                    else:
                        reconstr_loss = tf.constant(0.0, dtype=tf.float32)

                    # define intensity loss
                    if lam_lp != 0:
                        lp_loss = compute_intensity_loss(gen_frames=pred_output, gt_frames=input_gt,
                                                 l_num=l_num)
                    else:
                        lp_loss = tf.constant(0.0, dtype=tf.float32)

                    # define gdl loss
                    if lam_gdl != 0:
                        gdl_loss = compute_gradient_loss(gen_frames=pred_output, gt_frames=input_gt,
                                                 alpha=alpha_num)
                    else:
                        gdl_loss = tf.constant(0.0, dtype=tf.float32)

                    # define flow loss
                    if lam_flow != 0:
                        # cv2读取灰度图像，函数也会按bgr三个通道读取。这三个通道的像素值是相同的。
                        # flownet的input需是 3 channel 的
                        input_final_frame = tf.squeeze(input[:, -1:-2:-1, ...], axis=[1])
                        # 输入的最后一帧
                        if use_rgb == 1:
                            input_final_rgb = input_final_frame
                            input_gt_rgb = input_gt
                            output_rgb = pred_output
                        else:
                            input_final_rgb = tf.tile(input_final_frame, [1, 1, 1, 3])
                            input_gt_rgb = tf.tile(input_gt, [1, 1, 1, 3])
                            output_rgb = tf.tile(pred_output, [1, 1, 1, 3])
                        if use_tvnet == 0:
                            train_gt_flow = flownet(input_a=input_final_rgb,
                                                    input_b=input_gt_rgb,
                                                    height=cfg.flow_height,
                                                    width=cfg.flow_width,
                                                    reuse=None)
                            train_pred_flow = flownet(input_a=input_final_rgb,
                                                      input_b=output_rgb,
                                                      height=cfg.flow_height,
                                                      width=cfg.flow_width,
                                                      reuse=True)
                            # Computes the mean of elements across dimensions of a tensor
                            flow_loss = compute_flow_loss(train_gt_flow, train_pred_flow)
                        else:
                            tvnet = TVNet()
                            # start_time = time.time()
                            with tf.variable_scope('flow_1', reuse=(i > 0)):
                                flow1_u1, flow1_u2, _ = tvnet.tvnet_flow(input_final_rgb,
                                                                         input_gt_rgb,
                                                                         max_scales=tv_scale,
                                                                         warps=tv_warp,
                                                                         max_iterations=tv_iter)
                                train_gt_flow = tf.concat(values=[flow1_u1, flow1_u2], axis=3)
                            # duration = time.time() - start_time
                            # print(
                            #     'calculate first batch flows use [{:05f} s], fps = {:05f}'.format(
                            #         duration, cfg.batch_size / (gpu_nums * duration)))
                            # start_time = time.time()
                            with tf.variable_scope('flow_1', reuse=True):
                                flow2_u1, flow2_u2, _ = tvnet.tvnet_flow(input_final_rgb,
                                                                         output_rgb,
                                                                         max_scales=tv_scale,
                                                                         warps=tv_warp,
                                                                         max_iterations=tv_iter)
                            # duration = time.time() - start_time
                            # print(
                            #    'calculate second batch flows use [{:05f} s], fps = {:05f}'.format(
                            #         duration, cfg.batch_size / (gpu_nums * duration)))
                            train_pred_flow = tf.concat(values=[flow2_u1, flow2_u2], axis=3)
                            # Computes the mean of elements across dimensions of a tensor
                            flow_loss = compute_flow_loss(train_gt_flow, train_pred_flow)
                    else:
                        flow_loss = tf.constant(0.0, dtype=tf.float32)

                    # define adversarial loss
                    if adversarial:
                        # clip_lenth 帧 [batch, clip_length, h, w, c]
                        real_video = data
                        fake_video = tf.concat([data[:,:-1,...],
                                                tf.expand_dims(pred_output, axis=1)], axis=1)
                        # with tf.variable_scope('discriminator', reuse=None):
                        with tf.variable_scope('discriminator', reuse=(i > 0)):
                            if GAN_type == 'SNGAN':
                                discriminator = Discriminator(real_video, batch_size, is_sn=True)
                            else:
                                discriminator = Discriminator(real_video, batch_size)
                            real_outputs = discriminator.outputs
                        # 将参数reuse设置为True时，tf.get_variable 将只能获取已经创建过的变量。
                        with tf.variable_scope('discriminator', reuse=True):
                            if GAN_type == 'SNGAN':
                                discriminator = Discriminator(fake_video, batch_size, is_sn=True)
                            else:
                                discriminator = Discriminator(fake_video, batch_size)
                            fake_outputs = discriminator.outputs

                        # print('real_outputs = {}'.format(real_outputs))
                        # print('fake_outputs = {}'.format(fake_outputs))

                        if GAN_type == 'LSGAN':
                            # LSGAN, paper: Least Squares Generative Adversarial Networks
                            # adv_G_loss
                            adv_loss = tf.reduce_mean(tf.square(fake_outputs - 1) / 2)
                            # adv_D_loss
                            dis_loss = tf.reduce_mean(
                                tf.square(real_outputs - 1) / 2) + tf.reduce_mean(
                                tf.square(fake_outputs) / 2)
                        elif GAN_type == 'WGAN':
                            # WGAN, paper: Wasserstein GAN
                            # adv_G_loss
                            adv_loss = -tf.reduce_mean(fake_outputs)
                            # adv_D_loss, 负值
                            dis_loss = -tf.reduce_mean(real_outputs - fake_outputs)
                        elif GAN_type == 'WGAN-GP':
                            # WGAN-GP, paper: Improved Training of Wasserstein GANs
                            # gradient penalty 相较 weight penalty则可以让梯度在后向传播的过程中保持平稳
                            lambda_gp = 1
                            e = tf.random_uniform([batch_size, 1, 1, 1], 0, 1)
                            x_hat = e * input_gt + (1 - e) * pred_output

                            with tf.variable_scope('discriminator', reuse=True):
                                discriminator = Discriminator(x_hat, batch_size)
                                grad = tf.gradients(discriminator.outputs, x_hat)[0]

                            gradient_penalty = tf.reduce_mean(tf.square(
                                tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3])) - 1))

                            dis_loss = tf.reduce_mean(fake_outputs - real_outputs) \
                                       + lambda_gp * gradient_penalty

                            adv_loss = -tf.reduce_mean(fake_outputs)
                        elif GAN_type == 'SNGAN':
                            # SNGAN, paper: SPECTRAL NORMALIZATION FOR GENERATIVE ADVERSARIAL NETWORKS
                            dis_loss = - (tf.reduce_mean(tf.log(real_outputs + epsilon) + tf.log(
                                    1 - fake_outputs + epsilon)))
                            adv_loss = - tf.reduce_mean(tf.log(fake_outputs + epsilon))
                        else:
                            print('[!!!] wrong GAN_TYPE')
                            break

                    else:
                        adv_loss = tf.constant(0.0, dtype=tf.float32)
                        dis_loss = tf.constant(0.0, dtype=tf.float32)

                    g_loss = tf.add_n([reconstr_loss * lam_two_stream, lp_loss * lam_lp,
                                       gdl_loss * lam_gdl, adv_loss * lam_adv,
                                       flow_loss * lam_flow], name='g_loss')
                    g_losses.append(g_loss)
                    d_losses.append(dis_loss)

                    # # 重用variable
                    tf.get_variable_scope().reuse_variables()

                    # add summaries
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                    summaries.append(tf.summary.scalar(tensor=g_lrate,
                                                       name='g_lrate'))
                    summaries.append(tf.summary.scalar(tensor=d_lrate,
                    name='d_lrate'))
                    summaries.append(tf.summary.scalar(tensor=reconstr_loss,
                                                       name='reconstr_loss'))
                    summaries.append(tf.summary.scalar(tensor=g_loss,
                                                       name='generator_total_loss'))
                    summaries.append(tf.summary.scalar(tensor=dis_loss,
                                                       name='discriminator_loss'))
                    summaries.append(tf.summary.scalar(tensor=adv_loss,
                                                       name='generator_adverse_loss'))
                    summaries.append(tf.summary.scalar(tensor=lp_loss,
                                                       name='generator_intensorty_loss'))
                    summaries.append(tf.summary.scalar(tensor=gdl_loss,
                                                       name='generator_gradient_loss'))
                    summaries.append(tf.summary.scalar(tensor=flow_loss,
                                                       name='generator_flow_loss'))
                    summaries.append(tf.summary.image(tensor=pred_output, name='generator_output'))
                    summaries.append(tf.summary.image(tensor=input_gt, name='generator_gt'))
                    if lam_two_stream != 0:
                        # reconstr_output is [batch, 4, h, w, c]
                        # 第一帧 reconstr_output_first is  [batch, h, w, c]
                        reconstr_output_first = tf.squeeze(reconstr_output[:,0:1,...], axis=[1])
                        summaries.append(tf.summary.image(tensor=reconstr_output_first, 
                        name='reconstr_output_first'))
                        # 最后一帧 reconstr_output_last is  [batch, h, w, c]
                        reconstr_output_last = tf.squeeze(reconstr_output[:,-1:-2:-1,...], axis=[1])
                        summaries.append(tf.summary.image(tensor=reconstr_output_last, 
                        name='reconstr_output_last'))
                        
                    g_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                               scope='generator')
                    if lam_two_stream == 0:
                        g_vars = [x for x in g_vars if 'reconstruct' not in x.op.name]
                    generator_grads = g_optimizer.compute_gradients(g_loss, var_list=g_vars)
                    g_tower_grads.append(generator_grads)
                    if adversarial:
                        d_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                                   scope='discriminator')
                        discriminator_grads = d_optimizer.compute_gradients(dis_loss, var_list=
                        d_vars)
                        d_tower_grads.append(discriminator_grads)

        # 计算所有loss
        average_g_loss = average_losses(g_losses)
        average_d_loss = average_losses(d_losses)
        # cpu 上计算平均梯度
        g_grads = average_gradients(g_tower_grads)
        d_grads = average_gradients(d_tower_grads)
        # 更新
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            # calculate gradients
            train_g_op = g_optimizer.apply_gradients(g_grads, global_step=g_step)
            if adversarial:
                train_d_op = d_optimizer.apply_gradients(d_grads, global_step=d_step)
            else:
                train_d_op = None

        # add history for variables and gradients in genrator
        for var in tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope='generator'):
            summaries.append(
                tf.summary.histogram('G/' + var.op.name, var)
            )
        for grad, var in g_grads:
            if grad is not None:
                summaries.append(
                    tf.summary.histogram('G/' + var.op.name + '/gradients', grad)
                )
        # add history for variables and gradients in discriminator
        for var in tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope='discriminator'):
            summaries.append(
                tf.summary.histogram('D/' + var.op.name, var)
            )
        for grad, var in d_grads:
            if grad is not None:
                summaries.append(
                    tf.summary.histogram('D/' + var.op.name + '/gradients', grad)
                )

        # clip variable in Discriminator
        if GAN_type == 'LSGAN':
            clip_D = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]

        # create a saver
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
        saver_for_best = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=3)
        if use_trick:
            saver_for_trick = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)

        # build summary
        summary_op = tf.summary.merge(summaries)

    # start training session
    # "allow_soft_placement" must be set to True to build towers on GPU,
    # as some of the ops do not have GPU implementations.
    # "log_device_placement" set to True will print device place
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    
    if for_remote != 1:
        # fraction of overall amount of memory that each GPU should be allocated
        config.gpu_options.per_process_gpu_memory_fraction = 0.8

    with tf.Session(config=config) as sess:
        # summaries
        summary_path = os.path.join(summary_dir, model_name)
        summary_writer = tf.summary.FileWriter(summary_path, graph=sess.graph)

        init_or_restore(sess, saver, ckpt_path, logger)

        if lam_flow != 0 and use_tvnet == 0:
            # initialize flownet
            initialize_flownet(sess, FLOWNET_CHECKPOINT)

        volumes_per_step = small_batch * gpu_nums

        step = -1
        min_val_g_loss = np.inf
        min_val_loss_step = 0
        if use_trick == 1:
            min_val_loss_auc = 0
            max_psnr_auc = -np.inf
            best_auc_step = 0
        patient = 0
        if type(cfg.epochs) is int:
            for epoch in range(cfg.epochs):
                print("EPOCH: {}".format(epoch + 1))
                logger.info("EPOCH: {}".format(epoch + 1))
                sess.run(train_iter.initializer)
                sess.run(val_iter.initializer)
                while True:
                    try:
                        step += 1
                        # tf.cod的validate分支也会get_next， 避免先用完导致跳出循环
                        sess.run(val_iter.initializer)
                        # 一个step会并行运行所有的gpu
                        # 注意这里run的是cpu上的总的操作
                        # start_time = time.time()
                        if adversarial:
                            # print('Training discriminator...')
                            # logger.info('Training discriminator...')
                            if GAN_type == 'LSGAN':
                                _, _, average_d_loss_v, d_step_v = sess.run(
                                    [clip_D, train_d_op, average_d_loss, d_step],
                                    feed_dict={is_training: True,
                                               batch_size: cfg.batch_size // gpu_nums})
                                assert (not np.isnan(average_d_loss_v)), 'Model diverged with ' \
                                                                         'discriminator loss = NaN'
                            else:
                                _, average_d_loss_v, d_step_v = sess.run(
                                    [train_d_op, average_d_loss, d_step],
                                    feed_dict={is_training: True,
                                               batch_size: cfg.batch_size // gpu_nums})
                                assert (not np.isnan(average_d_loss_v)), 'Model diverged with ' \
                                                                         'discriminator loss = NaN'

                        # print('Training generator...')
                        # logger.info('Training generator...')
                        _, average_g_loss_v, g_step_v, summary_str = sess.run(
                            [train_g_op, average_g_loss, g_step, summary_op],
                            feed_dict={is_training: True,
                                       batch_size: cfg.batch_size // gpu_nums})

                        assert (not np.isnan(average_g_loss_v)), 'Model diverged with ' \
                                                                 'generator loss = NaN'
                        # duration = time.time() - start_time
                        # batch_per_sec = volumes_per_step / duration

                        if step % 10 == 0:
                            print("----- step:{} generator loss:{:09f}".format(
                                step, average_g_loss_v))
                            print("----- step:{} discriminator loss:{:09f}".format(
                                step, average_d_loss_v))
                            logger.info("----- step:{} generator loss:{:09f}".format(
                                step, average_g_loss_v))
                            logger.info("----- step:{} discriminator loss:{:09f}".format(
                                step, average_d_loss_v))
                        if step % 100 == 0:
                            summary_writer.add_summary(summary_str, step)
                        if step % 1000 == 0:
                            saver.save(sess, ckpt_path, global_step=step)
                    except tf.errors.OutOfRangeError:
                        print('train dataset finished')
                        logger.info('train dataset finished')
                        break
                    except:
                        import sys
                        print("Unexpected error:", sys.exc_info())
                # 这样写还是tf.cond的锅，两个batch都要有
                sess.run(train_iter.initializer)
                sess.run(val_iter.initializer)
                print('[!] start validate model ...')
                logger.info('[!] start validate model ...')
                val_g_losses = []
                while True:
                    try:
                        average_g_loss_v = sess.run(average_g_loss, feed_dict={is_training: False,
                                       batch_size: cfg.batch_size // gpu_nums})

                        assert (not np.isnan(average_g_loss_v)), 'Model diverged with ' \
                                                                 'generator loss = NaN'
                        val_g_losses.append(average_g_loss_v)
                    except tf.errors.OutOfRangeError:
                        break
                for loss in val_g_losses:
                    tmp = []
                    tmp.append(np.expand_dims(loss, axis=0))
                stacked_loss = np.concatenate(tuple(tmp), axis=0)
                cur_val_g_loss = np.mean(stacked_loss, axis=0)
                assert (not np.isnan(cur_val_g_loss)), 'Model diverged with cur_val_loss = NaN'
                print("----- step:[{}] validate loss:{:09f}".format(step, cur_val_g_loss))
                logger.info("----- step:{} validate loss:{:09f}".format(step, cur_val_g_loss))
                if cur_val_g_loss < min_val_g_loss:
                    min_val_g_loss = cur_val_g_loss
                    min_val_loss_step = step
                    patient = 0
                    saver_for_best.save(sess, best_ckpt_path, global_step=step)
                    print("update best model, min_val_g_loss is [{:09f}] in step [{}]".format(
                        cur_val_g_loss, step))
                    logger.info("update best model, min_val_g_loss is [{:09f}] in step [{}]".format(
                        cur_val_g_loss, step))
                    if use_trick == 1:
                        new_psnr_auc = test_in_train(cfg, logger, sess, model_name)
                        if step == min_val_loss_step:
                            min_val_loss_auc = new_psnr_auc
                            print("min_val_loss's auc is [{:09f}] in step [{}]".format(
                                    min_val_loss_auc, step))
                            logger.info("min_val_loss's auc is [{:09f}] in step [{}]".format(
                                min_val_loss_auc, step))
                        if new_psnr_auc > max_psnr_auc:
                            max_psnr_auc = new_psnr_auc
                            best_auc_step = step
                            print("best auc is [{:09f}] in step [{}]".format(max_psnr_auc, step))
                            logger.info("best auc is [{:09f}] in step [{}]".format(max_psnr_auc, 
                            step))
                            saver_for_trick.save(sess, trick_ckpt_path, global_step=step)
                else:
                    patient += 1
                    if patient >= cfg.patient:
                        print('[!!!] Early stop for no reducing in validate loss ' +
                                       'after [{}] epochs, '.format(cfg.patient) +
                                       'min_val_g_loss is [{:09f}]'.format(min_val_g_loss) + 
                                       ', in step [{}]'.format(min_val_loss_step))
                        logger.warning('[!!!] Early stop for no reducing in validate loss ' +
                                       'after [{}] epochs, '.format(cfg.patient) +
                                       'min_val_g_loss is [{:09f}]'.format(min_val_g_loss) + 
                                       ', in step [{}]'.format(min_val_loss_step))
                        # saver.save(sess, best_ckpt_path, global_step=step)
                        if use_trick == 1:
                            print("min_val_loss's auc is [{:09f}] in step [{}]".format(
                                    min_val_loss_auc, min_val_loss_step))
                            logger.info("min_val_loss's auc is [{:09f}] in step [{}]".format(
                                min_val_loss_auc, min_val_loss_step))
                            print("best auc is [{:09f}] in step [{}]".format(max_psnr_auc, 
                                best_auc_step))
                            logger.info("best auc is [{:09f}] in step [{}]".format(max_psnr_auc, 
                                best_auc_step))
                        break
    print('[!!!] model name:{}'.format(model_name))
    logger.info('[!!!] model name:{}'.format(model_name))


def compute_and_save_scores(errors, dir, video_id, error_name):
    # regularity score
    scores = errors - min(errors)
    scores = scores / max(scores)
    if error_name != 'psnr':
        # psnr原为正常值，转为异常值
        scores = 1 - scores
    regularity_score_file_dir = os.path.join(dir, error_name)
    if not os.path.exists(regularity_score_file_dir):
        os.makedirs(regularity_score_file_dir)
    regularity_score_file_path = os.path.join(regularity_score_file_dir,
                                        'scores_{:02d}.txt'.format(video_id+1))
    np.savetxt(regularity_score_file_path, scores)


def save_pixel_loss(errors, dir, video_id, error_name):

    regularity_score_file_dir = os.path.join(dir, error_name)
    if not os.path.exists(regularity_score_file_dir):
        os.makedirs(regularity_score_file_dir)
    np.save(os.path.join(regularity_score_file_dir, 'losses_{:02d}.npy'.format(video_id+1)),
            errors)
"""
Test phase
"""
def test(cfg, logger, model_name):
    size = '{}_{}'.format(cfg.width, cfg.height)
    if use_rgb == 1:
        npy_dir = os.path.join(npy_root_dir, args.dataset, size, 'rgb_testing_frames')
    else:
        npy_dir = os.path.join(npy_root_dir, args.dataset, size, 'testing_frames')
    video_nums = len(os.listdir(npy_dir))

    regularity_score_dir = os.path.join(regularity_score_root_dir, args.dataset, model_name, 
                                            'testing_frames')
    if not os.path.exists(regularity_score_dir):
        os.makedirs(regularity_score_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model_dir = os.path.join(checkpoint_dir, model_name)
    best_model_ckpt_dir = os.path.join(model_dir, 'best_model')
    assert os.path.exists(best_model_ckpt_dir)

    # [batch, clip_length, h, w, c]
    if use_rgb == 1:
        data = tf.placeholder(tf.float32, shape=[None, cfg.clip_length, cfg.height, cfg.width, 3])
    else:
        data = tf.placeholder(tf.float32, shape=[None, cfg.clip_length, cfg.height, cfg.width, 1])
    # [batch, clip_length-1,h, w, c]
    input = data[:, :-1, ...]
    # 最后一帧， [batch, h, w, c]
    input_gt = tf.squeeze(data[:, -1:-2:-1, ...], axis=[1])
    batch_size = tf.placeholder(tf.int32, [], name='batch_size')

    with tf.variable_scope('generator'):
        generator = Generator(cfg, input, batch_size, use_rgb, use_skip)
        # [batch, h, w, c]
        pred_output = generator._pred_out
        if lam_two_stream != 0:
            reconstr_output = generator._reconstr_out
                        

    saver_for_best = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=3)

    # comput loss
    psnr = compute_psnr(gen_frames=pred_output, gt_frames=input_gt)
    mse_error = compute_mse_error(gen_frames=pred_output, gt_frames=input_gt)

    # compute pixle loss
    pixel_loss = compute_pixel_loss(gen_frames=pred_output, gt_frames=input_gt)

    if lam_two_stream != 0:
        reconstr_loss = compute_reconstr_loss(input, reconstr_output)
    else:
        reconstr_loss = tf.constant(0.0, dtype=tf.float32)
    if adversarial:
        fake_video = tf.concat([data[:, :-1, ...],
                                tf.expand_dims(pred_output, axis=1)], axis=1)
        with tf.variable_scope('discriminator'):
            if GAN_type == 'SNGAN':
                discriminator = Discriminator(fake_video, batch_size, is_sn=True)
            else:
                discriminator = Discriminator(fake_video, batch_size)
            fake_outputs = discriminator.outputs

        # print('real_outputs = {}'.format(real_outputs))
        # print('fake_outputs = {}'.format(fake_outputs))

        dis_loss = -tf.reduce_mean(fake_outputs + epsilon)      
    else:
        dis_loss = tf.constant(0.0, dtype=tf.float32)
        

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    
    if for_remote != 1:
        # fraction of overall amount of memory that each GPU should be allocated
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
    
    with tf.Session(config=config) as sess:
        init_or_restore(sess, saver_for_best, best_model_ckpt_dir, logger)

        start_time = time.time()
        try:
            assert os.path.isfile(os.path.join(regularity_score_dir, 'ignored_frames_list.txt'))
            assert os.path.isfile(os.path.join(regularity_score_dir, 'video_length_list.txt'))
        except:
            print('[!!!] 未发现video_length_list.txt 或 ignored_frames_list.txt， 开始计算')
            logger.info('[!!!] 未发现video_length_list.txt 或 ignored_frames_list.txt， 开始计算')
            # total_frame_nums = 0
            IGNORED_FRAMES_LIST = []
            VIDEO_LENGTH_LIST = []
            for i in range(video_nums):
                npy_name = 'testing_frames_{:02d}.npy'.format(i + 1)
                if use_rgb == 1:
                    npy_name = 'rgb_' + npy_name
                data_frames = np.load(os.path.join(npy_dir, npy_name))
                frame_nums = data_frames.shape[0]
                used_frame_nums = frame_nums
                IGNORED_FRAMES_LIST.append(cfg.time_length)
                VIDEO_LENGTH_LIST.append(used_frame_nums)
                # total_frame_nums +=used_frame_nums
            np.savetxt(os.path.join(regularity_score_dir, 'ignored_frames_list.txt'),
                       np.array(IGNORED_FRAMES_LIST), fmt='%d')
            np.savetxt(os.path.join(regularity_score_dir, 'video_length_list.txt'),
                       np.array(VIDEO_LENGTH_LIST), fmt='%d')

        for i in range(video_nums):
            npy_name = 'testing_frames_{:02d}.npy'.format(i + 1)
            if use_rgb == 1:
                npy_name = 'rgb_' + npy_name
            data_frames = np.load(os.path.join(npy_dir, npy_name))
            # frame_nums = data_frames.shape[0]
            # used_frame_nums = frame_nums
            # IGNORED_FRAMES_LIST.append(cfg.clip_length-1)
            # VIDEO_LENGTH_LIST.append(used_frame_nums)
            # total_frame_nums +=used_frame_nums
            score_file = [x for x in os.listdir(regularity_score_dir)
                          if x.startswith('scores_') == True]
            if (len(score_file) != video_nums):
                pixel_loss_l = []
                mse_errors_l = []
                psnr_l = []
                # if lam_two_stream != 0:
                #     reconstr_loss_l = []
                #     psnr_rec_l = []
                dis_loss_l = []
                psnr_dis_l = []
                psnr_half_dis_l = []

                psnr_half_rec_l = []

                for j in range(len(data_frames)-cfg.clip_length+1):
                    # [clip_length, h, w]
                    tested_data = data_frames[j:j + cfg.clip_length]
                    # [n, clip_length, h, w] for gray or [n, clip_length, h, w, c] for rgb
                    tested_data = np.expand_dims(tested_data, axis=0)
                    if use_rgb == 0:
                        # [n, clip_length, h, w, 1]
                        tested_data = np.expand_dims(tested_data, axis=-1)

                    pixel_loss_v, dis_loss_v, mse_error_v, psnr_v = sess.run(
                            [pixel_loss, dis_loss, mse_error, psnr],
                            feed_dict={data: tested_data,  batch_size:1})

                    pixel_loss_l.append(pixel_loss_v)
                    mse_errors_l.append(mse_error_v)
                    psnr_l.append(psnr_v)
                    dis_loss_l.append(dis_loss_v)
                    psnr_dis_l.append(-psnr_v + dis_loss_v)
                    psnr_half_dis_l.append(-psnr_v + 0.5*dis_loss_v)
                    # psnr_half_rec_l.append(-psnr_v + 0.5*reconstr_loss_v)
                    # if lam_two_stream != 0:
                    #     reconstr_loss_l.append(reconstr_loss_v)
                    #     psnr_rec_l.append(-psnr_v + reconstr_loss_v)

                pixel_losses = np.array(pixel_loss_l)
                mse_errors = np.array(mse_errors_l)
                psnrs = np.array(psnr_l)
                dis_losses = np.array(dis_loss_l)
                # if lam_two_stream != 0:
                #     reconstr_losses = np.array(reconstr_loss_l)
                #     psnr_recs = np.array(psnr_rec_l)
                psnr_half_recs = np.array(psnr_half_rec_l)
                psnr_diss = np.array(psnr_dis_l)
                psnr_half_diss = np.array(psnr_half_dis_l)

                save_pixel_loss(pixel_losses, regularity_score_dir, i, 'pixel_loss')
                compute_and_save_scores(mse_errors, regularity_score_dir, i, 'mse')
                compute_and_save_scores(psnrs, regularity_score_dir, i, 'psnr')
                compute_and_save_scores(dis_losses, regularity_score_dir, i, 'dis')
                # compute_and_save_scores(psnr_half_recs, regularity_score_dir, i, 'psnr_half_rec')
                compute_and_save_scores(psnr_diss, regularity_score_dir, i, 'psnr_dis')
                compute_and_save_scores(psnr_half_diss, regularity_score_dir, i, 'psnr_half_dis')
                # if lam_two_stream != 0:
                #     compute_and_save_scores(reconstr_losses, regularity_score_dir, i, 'rec')
                #     compute_and_save_scores(psnr_recs, regularity_score_dir, i, 'psnr_rec')

    print('AUC and EER result:')
    logger.info('AUC and EER result:')
    # compute auc and eer
    for error_name in ['mse', 'psnr', 'dis', 'psnr_dis', 'psnr_half_dis']:
        assert os.path.exists(os.path.join(regularity_score_dir, error_name)) is True\
            , '[!!!] error_name:{} is non-existent.'.format(error_name)
        print('---- error_name:{}'.format(error_name))
        logger.info('---- error_name:{}'.format(error_name))
        auc = compute_auc(video_nums, regularity_score_dir, error_name, args.dataset,
                          gt_root_dir)
        print('auc = {:09f}'.format(auc))
        logger.info('auc = {:09f}'.format(auc))
        eer = compute_eer(video_nums, regularity_score_dir, error_name, args.dataset,
                          gt_root_dir)
        print('eer = {:09f}'.format(eer))
        logger.info('eer = {:09f}'.format(eer))

        # plot score
        plot_score(video_nums, args.dataset, regularity_score_dir, error_name, logger,
                   gt_root_dir, cfg.clip_length - 1)

    # plot heatmap
    for error_name in ['pixel_loss']:
        assert os.path.exists(os.path.join(regularity_score_dir, error_name)) is True\
            , '[!!!] error_name:{} is non-existent.'.format(error_name)
        print('---- error_name:{}'.format(error_name))
        logger.info('---- error_name:{}'.format(error_name))

        # plot score
        plot_heatmap(video_nums, args.dataset, regularity_score_dir, error_name, logger,
                     cfg.clip_length - 1, dataset_root_dir, cfg, gt_root_dir)
    print('[!!!] model name:{}'.format(model_name))
    logger.info('[!!!] model name:{}'.format(model_name))

"""
Test_in_train phase
"""
def test_in_train(cfg, logger, sess, model_name):
    size = '{}_{}'.format(cfg.width, cfg.height)
    with tf.variable_scope('trick_for_auc', reuse=tf.AUTO_REUSE):
        # [batch, clip_length, h, w, c]
        if use_rgb == 1:
            data = tf.placeholder(tf.float32,
                                  shape=[None, cfg.clip_length, cfg.height, cfg.width, 3])
        else:
            data = tf.placeholder(tf.float32,
                                  shape=[None, cfg.clip_length, cfg.height, cfg.width, 1])
        # [batch, clip_length-1,h, w, c]
        input = data[:, :-1, ...]
        # 最后一帧， [batch, h, w, c]
        input_gt = tf.squeeze(data[:, -1:-2:-1, ...], axis=[1])
        batch_size = tf.placeholder(tf.int32, [], name='batch_size')

    with tf.variable_scope('generator', reuse=True):
        generator = Generator(cfg, input, batch_size, use_rgb, use_skip)
        # [batch, h, w, c]
        pred_output = generator._pred_out

    # comput loss
    psnr = compute_psnr(gen_frames=pred_output, gt_frames=input_gt)
    # mse_error = compute_mse_error(gen_frames=pred_output, gt_frames=input_gt)

    if use_rgb == 1:
        npy_dir = os.path.join(npy_root_dir, args.dataset, size, 'rgb_testing_frames')
    else:
        npy_dir = os.path.join(npy_root_dir, args.dataset, size, 'testing_frames')
    video_nums = len(os.listdir(npy_dir))
    # [error_video_1, error_video_1, ...]

    regularity_score_dir = os.path.join(regularity_score_root_dir, args.dataset, model_name,
                                            'testing_frames')
    if not os.path.exists(regularity_score_dir):
        os.makedirs(regularity_score_dir)

    try:
        assert os.path.isfile(os.path.join(regularity_score_dir, 'ignored_frames_list.txt'))
        assert os.path.isfile(os.path.join(regularity_score_dir, 'video_length_list.txt'))
    except:
        print('[!!!] 未发现video_length_list.txt 或 ignored_frames_list.txt， 开始计算')
        logger.info('[!!!] 未发现video_length_list.txt 或 ignored_frames_list.txt， 开始计算')
        # total_frame_nums = 0
        IGNORED_FRAMES_LIST = []
        VIDEO_LENGTH_LIST = []
        for i in range(video_nums):
            npy_name = 'testing_frames_{:02d}.npy'.format(i+1)
            if use_rgb == 1:
                npy_name = 'rgb_' + npy_name
            data_frames = np.load(os.path.join(npy_dir, npy_name))
            frame_nums = data_frames.shape[0]
            used_frame_nums = frame_nums
            IGNORED_FRAMES_LIST.append(cfg.clip_length-1)
            VIDEO_LENGTH_LIST.append(used_frame_nums)
            # total_frame_nums +=used_frame_nums
        np.savetxt(os.path.join(regularity_score_dir, 'ignored_frames_list.txt'),
                    np.array(IGNORED_FRAMES_LIST), fmt='%d')
        np.savetxt(os.path.join(regularity_score_dir, 'video_length_list.txt'),
                    np.array(VIDEO_LENGTH_LIST), fmt='%d')
    
    for i in range(video_nums):
        npy_name = 'testing_frames_{:02d}.npy'.format(i+1)
        if use_rgb == 1:
            npy_name = 'rgb_' + npy_name
        data_frames = np.load(os.path.join(npy_dir, npy_name))
        # frame_nums = data_frames.shape[0]
        # used_frame_nums = frame_nums
        # IGNORED_FRAMES_LIST.append(cfg.clip_length-1)
        # VIDEO_LENGTH_LIST.append(used_frame_nums)
        # total_frame_nums +=used_frame_nums
        score_file = [x for x in os.listdir(regularity_score_dir)
                        if x.startswith('scores_')==True]
        if(len(score_file)!=video_nums):
            psnr_l = []
            for j in range(len(data_frames)-cfg.clip_length+1):
                # [clip_length, h, w]
                tested_data = data_frames[j:j + cfg.clip_length]
                # [n, clip_length, h, w] for gray or [n, clip_length, h, w, c] for rgb
                tested_data = np.expand_dims(tested_data, axis=0)
                if use_rgb == 0:
                    # [n, clip_length, h, w, 1]
                    tested_data = np.expand_dims(tested_data, axis=-1)
                psnr_v = sess.run([psnr], feed_dict={data: tested_data,
                                                                batch_size:1})
                psnr_l.append(psnr_v)

            psnrs = np.array(psnr_l)
            compute_and_save_scores(psnrs, regularity_score_dir, i, 'psnr')
        
    print('AUC and EER result:')
    logger.info('AUC and EER result:')
    # compute auc and eer
    for error_name in ['psnr']:
        assert os.path.exists(os.path.join(regularity_score_dir, error_name)) is True\
            , '[!!!] error_name:{} is non-existent.'.format(error_name)
        print('---- error_name:{}'.format(error_name))
        logger.info('---- error_name:{}'.format(error_name))
        auc = compute_auc(video_nums, regularity_score_dir, error_name, args.dataset,
                            gt_root_dir)
        print('auc = {:09f}'.format(auc))
        logger.info('auc = {:09f}'.format(auc))
        eer = compute_eer(video_nums, regularity_score_dir, error_name, args.dataset,
                            gt_root_dir)
        print('eer = {:09f}'.format(eer))
        logger.info('eer = {:09f}'.format(eer))
    return auc


"""
Detect event phase
"""
def detect_event(cfg, logger, model_name):
    if use_rgb == 1:
        npy_dir = os.path.join(npy_root_dir, args.dataset, 'rgb_testing_frames')
    else:
        npy_dir = os.path.join(npy_root_dir, args.dataset, 'testing_frames')
    video_nums = len(os.listdir(npy_dir))

    regularity_score_dir = os.path.join(regularity_score_root_dir, model_name, args.dataset,
                                        'testing_frames')
    if not os.path.exists(regularity_score_dir):
        os.makedirs(regularity_score_dir)

    # plot event
    error_name = 'psnr'
    minima_dir = os.path.join(regularity_score_dir, error_name, 'minima_{:.2f}/'.format(
        minima_threshold))
    assert os.path.exists(minima_dir), '[!!!] No minima dir exists.'
    plot_event(video_nums, args.dataset, regularity_score_dir, error_name, logger, gt_root_dir,
               cfg.clip_length - 1, minima_dir, window_length)


def main(*argc, **argv):
    cfg = Config()

    model_name = '{}-{}-{}-{}_{}_{}_{}_{}-{}_{}_{}_{}_{}_{}_{}_{}_{}_{}-{}_{}_{}-{}_{}_{}_{}'.format(
        args.dataset,
        use_rgb, use_skip, cfg.batch_size, cfg.clip_length, cfg.opt, cfg.epochs, cfg.patient,
        l_num, alpha_num, lam_lp, lam_gdl, lam_adv, lam_flow, lam_two_stream, use_tvnet, GAN_type,
        beta1,
        tv_scale, tv_warp, tv_iter,
        LRATE_G[0], LRATE_G[1], LRATE_D[0], LRATE_D[1])
    if LRATE_G[0] != LRATE_G[1] or LRATE_D[0] != LRATE_D[1]:
        model_name += '-{}_LRdecay'.format(LRATE_D_BOUNDARIES[0])
    if cfg.quick_train == 0:
        model_name += '-full'

    log_dir = os.path.join(log_path, model_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(filename=os.path.join(log_dir, '{}.log'.format(datetime.datetime.now().
                                                                        strftime('%Y%m%d-%H%M%S'))),
                        level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger()

    if args.phase == 'train':
        train(cfg, logger, model_name)
    elif args.phase == 'test':
        test(cfg, logger, model_name)
    elif args.phase == 'event':
        detect_event(cfg, logger, model_name)
    else:
        print('[!!!] wrong key word')

        
if __name__  == "__main__":
    tf.app.run()
