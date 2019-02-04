import tensorflow as tf
import os




def init_or_restore(sess, saver, checkpoint_dir, logger):
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        # model_checkpoint_path 自动保存最新model_path,注意路径
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('[*] checkpoint load successful')
        logger.info('[*] checkpoint load successful')
    else:
        print('[!] no checkpoint, model is initialized...')
        logger.info('[!] no checkpoint, model is initialized...')

def average_losses(losses):
    """Calculate the average loss for input list of tensors
    Args:
     losses: List of average loss for each batch.
    Returns:
       average loss for all gpu's loss.
    """
    new_losses = []
    for loss in losses:
        new_losses.append(tf.expand_dims(loss, 0))
    stacked_loss = tf.concat(values=new_losses, axis=0)
    mean_loss = tf.reduce_mean(stacked_loss, 0)
    return mean_loss
    

# 所有tower上的梯度取平均
def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
