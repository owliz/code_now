import tensorflow as tf
import numpy as np



def compute_reconstr_loss(input_frames, gen_frames):
    """
    Calculate reconstuct losses between the reconstructed frames and input frames.
    :param input_frames:
    :param gen_frames:
    :return: batch_mean_reconstr_loss.
    """
    return 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(input_frames - gen_frames), [1, 2, 3, 4]))


# def compute_flow_loss(gt_flows, gen_flows):
#     return tf.reduce_mean(tf.abs(gt_flows - gen_flows))
def compute_flow_loss(gt_flows, gen_flows):
    return tf.reduce_mean(tf.reduce_sum(tf.abs(gt_flows - gen_flows), axis=[1,2,3]))


# def compute_intensity_loss(gen_frames, gt_frames, l_num):
#     """
#     Calculates the sum of lp losses between the predicted and ground truth frames.
#
#     @param gen_frames: The predicted frames at each scale.
#     @param gt_frames: The ground truth frames at each scale
#     @param l_num: 1 or 2 for l1 and l2 loss, respectively).
#
#     @return: The lp loss.
#     """
#     return tf.reduce_mean(tf.abs((gen_frames - gt_frames) ** l_num))
def compute_intensity_loss(gen_frames, gt_frames, l_num):
    """
    Calculates the sum of lp losses between the predicted and ground truth frames.

    @param gen_frames: The predicted frames at each scale.
    @param gt_frames: The ground truth frames at each scale
    @param l_num: 1 or 2 for l1 and l2 loss, respectively).

    @return: The lp loss.
    """
    return tf.reduce_mean(tf.reduce_sum(tf.abs((gen_frames - gt_frames) ** l_num), axis=[1,2,3]))


# def compute_gradient_loss(gen_frames, gt_frames, alpha):
#     """
#     Calculates the sum of GDL losses between the predicted and ground truth frames.
#
#     @param gen_frames: The predicted frames at each scale.
#     @param gt_frames: The ground truth frames at each s cale
#     @param alpha: The power to which each gradient term is raised.
#
#     @return: The GDL loss.
#     """
#     # calculate the loss for each scale
#     # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.
#     #水平和垂直方向各自3个卷积核，即每次只对一个通道进行卷积求梯度。水平卷积核大小[1x2x3]比如[[[-1,0,0],
#     #[1,0,0]]]是R通道 1对应h方向，2对应w方向，3对应channel方向
#     #垂直的[2x1x3] 同上
#     channels = gen_frames.get_shape().as_list()[-1]
#     pos = tf.constant(np.identity(channels), dtype=tf.float32)     # 3 x 3
#     neg = -1 * pos
#     filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)  # [-1, 1]
#     filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]
#     strides = [1, 1, 1, 1]  # stride of (1, 1)
#     padding = 'SAME'
#     #每个的大小为height*width*3
#     gen_dx = tf.abs(tf.nn.conv2d(gen_frames, filter_x, strides, padding=padding))
#     gen_dy = tf.abs(tf.nn.conv2d(gen_frames, filter_y, strides, padding=padding))
#     gt_dx = tf.abs(tf.nn.conv2d(gt_frames, filter_x, strides, padding=padding))
#     gt_dy = tf.abs(tf.nn.conv2d(gt_frames, filter_y, strides, padding=padding))
#
#     grad_diff_x = tf.abs(gt_dx - gen_dx)
#     grad_diff_y = tf.abs(gt_dy - gen_dy)
#
#     # condense into one tensor and avg
#      #return gen_dx, gen_dy
#     return tf.reduce_mean(grad_diff_x ** alpha + grad_diff_y ** alpha)
def compute_gradient_loss(gen_frames, gt_frames, alpha):
    """
    Calculates the sum of GDL losses between the predicted and ground truth frames.

    @param gen_frames: The predicted frames at each scale.
    @param gt_frames: The ground truth frames at each s cale
    @param alpha: The power to which each gradient term is raised.

    @return: The GDL loss.
    """
    # calculate the loss for each scale
    # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.
    #水平和垂直方向各自3个卷积核，即每次只对一个通道进行卷积求梯度。水平卷积核大小[1x2x3]比如[[[-1,0,0],
    #[1,0,0]]]是R通道 1对应h方向，2对应w方向，3对应channel方向
    #垂直的[2x1x3] 同上
    channels = gen_frames.get_shape().as_list()[-1]
    pos = tf.constant(np.identity(channels), dtype=tf.float32)     # 3 x 3
    neg = -1 * pos
    filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)  # [-1, 1]
    filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]
    strides = [1, 1, 1, 1]  # stride of (1, 1)
    padding = 'SAME'
    #每个的大小为height*width*3
    gen_dx = tf.abs(tf.nn.conv2d(gen_frames, filter_x, strides, padding=padding))
    gen_dy = tf.abs(tf.nn.conv2d(gen_frames, filter_y, strides, padding=padding))
    gt_dx = tf.abs(tf.nn.conv2d(gt_frames, filter_x, strides, padding=padding))
    gt_dy = tf.abs(tf.nn.conv2d(gt_frames, filter_y, strides, padding=padding))

    grad_diff_x = tf.abs(gt_dx - gen_dx)
    grad_diff_y = tf.abs(gt_dy - gen_dy)

    # condense into one tensor and avg
     #return gen_dx, gen_dy
    return tf.reduce_mean(tf.reduce_sum(grad_diff_x ** alpha + grad_diff_y ** alpha, axis=[1,2,3]))


def log10(t):
    """
    Calculates the base-10 log of each element in t.

    @param t: The tensor from which to calculate the base-10 log.

    @return: A tensor with the base-10 log of each element in t.
    """

    numerator = tf.log(t)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def compute_psnr(gen_frames, gt_frames):
    """
    Computes the Peak Signal to Noise Ratio between the generated images and the ground
    truth images.

    @param gen_frames: A tensor of shape [batch_size, height, width, 3]. The frames generated by the
                       generator model.
    @param gt_frames: A tensor of shape [batch_size, height, width, 3]. The ground-truth frames for
                      each frame in gen_frames.

    @return: A scalar tensor. The mean Peak Signal to Noise Ratio
                over each frame in the batch.
    """
    shape = tf.shape(gen_frames)
    num_pixels = tf.to_float(shape[1] * shape[2] * shape[3])
    gt_frames = (gt_frames + 1.0) / 2.0
    gen_frames = (gen_frames + 1.0) / 2.0
    square_diff = tf.square(gt_frames - gen_frames)

    batch_psnr = 10 * log10(1 / ((1 / num_pixels) * tf.reduce_sum(square_diff, [1, 2, 3])))
    return tf.reduce_mean(batch_psnr)


def compute_mse_error(gen_frames, gt_frames):
    """
    Compute the mse error between generated frame and gt frame
    :param gen_frames: A tensor of shape [batch_size, height, width, channel]. Frame generated by
                        generator model.
    :param gt_frames: A tensor of shape [batch_size, height, width, 3]. The ground-truth frames for
                      each frame in gen_frames.
    :return: A scalar tensor.
    """
    batch_errors = tf.reduce_sum(tf.sqrt(tf.square(gen_frames-gt_frames)), [1,2,3])
    return tf.reduce_mean(batch_errors)


def compute_pixel_loss(gen_frames, gt_frames):
    """
    Compute the pixel loss between generated frame and gt frame
    :param gen_frames: A tensor of shape [batch_size, height, width, channel]. Frame generated by
                        generator model.
    :param gt_frames: A tensor of shape [batch_size, height, width, 3]. The ground-truth frames for
                      each frame in gen_frames.
    :return: A scalar tensor.
    """
    batch_errors = tf.reduce_sum(tf.sqrt(tf.square(gen_frames-gt_frames)), [3])
    return tf.reduce_mean(batch_errors, axis=0)