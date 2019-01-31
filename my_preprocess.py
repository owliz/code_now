import logging
import coloredlogs
import datetime
import os
import uuid
# from skimage.io import imread
import cv2
import numpy as np
import tensorflow as tf
import sys


device = 'gpu0'

job_uuid = str(uuid.uuid4())
log_path = os.path.join('logs')
os.makedirs(log_path, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_path,
                                          "{}.log".format(datetime.datetime.now().strftime(
                                              "%Y%m%d-%H%M%S"))),
                    level=logging.DEBUG,
                    format="%(asctime)s [%(levelname)s] %(message)s")
coloredlogs.install(level=logging.INFO)
logger = logging.getLogger()


if device == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    logger.debug("Using CPU only")
elif device == 'gpu0':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    logger.debug("Using GPU 0")


def to_npy(dataset, dataset_path, height, width, frames_type='training_frames', img_type='gray'):
    """
    transfer frames into grey, then into npy.
    :param dataset: dataset name
    :param dataset_path: dataset root dir
    :return: .npy  file
    """
    size = '{}_{}'.format(height, width)
    target_frames_type = frames_type
    if img_type == 'rgb':
        target_frames_type = 'rgb_' + target_frames_type
    logger.info("[{}] to npy for [{}]".format(frames_type, dataset))
    frame_path = os.path.join(dataset_path, dataset, frames_type)

    npy_dir_path = os.path.join(dataset_path, 'npy_dataset', dataset, size, target_frames_type)
    os.makedirs(npy_dir_path, exist_ok=True)

    for frames_folder in os.listdir(frame_path):
        print('==> ' + os.path.join(frame_path, frames_folder))
        training_frames_vid = []
        for frame_file in sorted(os.listdir(os.path.join(frame_path, frames_folder))):
            frame_file_name = os.path.join(frame_path, frames_folder, frame_file)
            # 灰度 [-1, 1]
            if img_type == 'gray':
                frame_value = cv2.imread(frame_file_name, cv2.IMREAD_GRAYSCALE)
            else:
                # BGR, 0~255，通道格式为(W, H, C)
                frame_value = cv2.imread(frame_file_name)
                frame_value = cv2.cvtColor(frame_value, cv2.COLOR_BGR2RGB)

            frame_value = cv2.resize(frame_value, (width, height))
            frame_value = frame_value.astype(dtype=np.float32)
            frame_value = (frame_value / 127.5) - 1.0
            assert(-1. <= frame_value.all() <= 1.)
            training_frames_vid.append(frame_value)
        training_frames_vid = np.array(training_frames_vid)

        np.save(os.path.join(npy_dir_path, '{}_{}.npy'.format(target_frames_type, frames_folder)),
                training_frames_vid)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tostring()]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))


def volumes_counter(dataset, dataset_path, size, img_type='gray', frames_type='training_frames',
                    clip_length=10, stride=1):
    """
    transfer to volume unit and saved in tfrecord
    :param dataset:
    :param dataset_path:
    :param frames_type:
    :param clip_length:
    :return:
    """
    if img_type == 'rgb':
        frames_type = 'rgb_' + frames_type
    print("to_volume_stride_[{}] for [{}] [{}]".format(stride, frames_type, dataset))
    logger.info("to_volume_stride_[{}] for [{}] [{}]".format(stride, frames_type, dataset))
    num_videos = len(os.listdir(os.path.join(dataset_path, 'npy_dataset', dataset, size,
                                             frames_type)))
    if img_type == 'rgb':
        tfrecord_dir = os.path.join(dataset_path, 'tfrecord_dataset', dataset, size,
                                'rgb_clip_length_{:02d}'.format(clip_length))
    else:
        tfrecord_dir = os.path.join(dataset_path, 'tfrecord_dataset', dataset, size,
                                    'clip_length_{:02d}'.format(clip_length))
    os.makedirs(tfrecord_dir, exist_ok=True)

    all_vol_num = 0
    for i in range(num_videos):
        data_frames = np.load(os.path.join(dataset_path, 'npy_dataset', dataset, size, frames_type,
                                           '{}_{:02d}.npy'.format(frames_type, i+1)))
        num_frames = data_frames.shape[0]
        vol_num = num_frames - ((clip_length-1)*stride + 1) + 1
        all_vol_num += vol_num

    save_path = os.path.join(tfrecord_dir, 'num_of_volumes_in_{}_stride_{}.txt'.format(frames_type,
                                                                                       stride))
    np.savetxt(save_path, np.array([all_vol_num + 1]), fmt='%d')
    print('volumes num is [{}] for [stride_{}]'.format(all_vol_num + 1, stride))
    logger.info('volumes num is [{}] for [stride_{}]'.format(all_vol_num + 1, stride))


def to_tfrecord(dataset, dataset_path, frames_type='training_frames', clip_length=10, stride=1):
    """
    transfer to volume unit and saved in tfrecord
    :param dataset:
    :param dataset_path:
    :param frames_type:
    :param clip_length:
    :return:
    """
    print("to_volume_stride_[{}] for [{}] [{}]".format(stride, frames_type, dataset))
    logger.info("to_volume_stride_[{}] for [{}] [{}]".format(stride, frames_type, dataset))
    num_videos = len(os.listdir(os.path.join(dataset_path, 'npy_dataset', dataset, frames_type)))
    tfrecord_dir = os.path.join(dataset_path, 'tfrecord_dataset', dataset,
                                'clip_length_{:02d}'.format(clip_length))
    os.makedirs(tfrecord_dir, exist_ok=True)
    desfile = os.path.join(tfrecord_dir, '{}_stride_{}.tfrecords'.format(frames_type, stride))

    with tf.python_io.TFRecordWriter(desfile) as writer:
        all_vol_num = 0
        for i in range(num_videos):
            data_frames = np.load(os.path.join(dataset_path, 'npy_dataset', dataset, frames_type,
                                               '{}_{:02d}.npy'.format(frames_type, i+1)))
            # 末尾增加一个维度
            # data_frames = np.expand_dims(data_frames, axis=-1)
            num_frames = data_frames.shape[0]
            vol = 0
            vol_num = num_frames - ((clip_length-1)*stride + 1) + 1
            volumes = np.zeros((vol_num, clip_length, resize_height, resize_width)).astype('float32')

            for j in range(vol_num):
                volumes[vol] = data_frames[j:j + (clip_length - 1) * stride + 1:stride]
                saved_volume = volumes[vol]
                # Create an example protocol buffer
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={'volume': _bytes_feature(saved_volume)}
                    )
                )
                # example = tf.train.Example(
                #     features=tf.train.Features(
                #         feature={'volume': _bytes_feature(saved_volume),
                #                  'shape': _int64_feature(saved_volume.shape)}
                #     )
                # )
                writer.write(example.SerializeToString())
            all_vol_num += vol_num

    print("to_volume_stride_[{}] for [{}] [{}] finished".format(stride, frames_type, dataset))
    logger.info("to_volume_stride_[{}] for [{}] [{}] finished".format(stride, frames_type, dataset))
    save_path = os.path.join(tfrecord_dir, 'num_of_volumes_in_{}_stride_{}.txt'.format(frames_type,
                                                                                       stride))
    np.savetxt(save_path, np.array([vol_num + 1]), fmt='%d')
    print('[{}] volumes are saved'.format(vol_num + 1))
    logger.info('[{}] volumes are saved'.format(vol_num + 1))


def to_tfrecord_split(dataset, dataset_path, target_dir, img_type, size,
                      frames_type='training_frames',
                      clip_length=10,
                      stride=1, validate_split=0.15):
    """
    transfer to volume unit and saved in tfrecord
    :param dataset:
    :param dataset_path:
    :param frames_type:
    :param clip_length:
    :return:
    """
    if img_type == 'rgb':
        frames_type = 'rgb_' + frames_type
    num_videos = len(os.listdir(os.path.join(dataset_path, 'npy_dataset', dataset, size,
                                             frames_type)))
    os.makedirs(target_dir, exist_ok=True)
    desfile_1 = os.path.join(target_dir, 'train_stride_{}_validate_split_{}.tfrecords'.format(
        stride, validate_split))
    desfile_2 = os.path.join(target_dir, 'val_stride_{}_validate_split_{}.tfrecords'.format(
        stride, validate_split))

    save_path = os.path.join(target_dir, 'num_of_volumes_in_{}_stride_{}.txt'.format(
        frames_type, stride))
    all_vol_num = np.loadtxt(save_path, dtype=int)

    train_size = int((1 - validate_split)*all_vol_num)

    write_1 = tf.python_io.TFRecordWriter(desfile_1)
    write_2 = tf.python_io.TFRecordWriter(desfile_2)
    count = 0
    for i in range(num_videos):
        data_frames = np.load(os.path.join(dataset_path, 'npy_dataset', dataset, size, frames_type,
                                           '{}_{:02d}.npy'.format(frames_type, i+1)))
        # 末尾增加一个维度
        # data_frames = np.expand_dims(data_frames, axis=-1)
        num_frames = data_frames.shape[0]
        vol_num = num_frames - ((clip_length - 1) * stride + 1) + 1
        if img_type == 'rgb':
            volumes = np.zeros((vol_num, clip_length, resize_height, resize_width, 3)).astype(
                'float32')
        else:
            volumes = np.zeros((vol_num, clip_length, resize_height, resize_width)).astype(
                'float32')
        vol = 0
        for j in range(vol_num):
            volumes[vol] = data_frames[j:j + (clip_length - 1) * stride + 1:stride]
            saved_volume = volumes[vol]
            # Create an example protocol buffer
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={'volume': _bytes_feature(saved_volume)}
                )
            )
            count += 1
            vol += 1
            if count <= train_size:
                write_1.write(example.SerializeToString())
                if count == train_size:
                    write_1.close()
            else:
                write_2.write(example.SerializeToString())

    write_2.close()
    print('train_num:{}'.format(train_size))
    logger.info('train_num:{}'.format(train_size))
    print('val_num:{}'.format(all_vol_num - train_size))
    logger.info('val_num:{}'.format(all_vol_num - train_size))
    np.savetxt(os.path.join(target_dir,
                            'num_of_volumes_in_train_stride_{}_validate_split_{}.txt'.format(
                                stride, validate_split)),
               [train_size],
               fmt='%d')
    np.savetxt(os.path.join(target_dir,
                            'num_of_volumes_in_val_stride_{}_validate_split_{}.txt'.format(
                                stride, validate_split)),
               [all_vol_num - train_size],
               fmt='%d')
    print("finished split for stride_{}".format(stride))
    logger.info("finished split for stride_{}".format(stride))



def preprocess(logger, dataset, clip_length, dataset_path, img_type='gray',height=256, width=256):
    """
    1. frames to npy file
    2. npy into volumes' collection
    :param logger:
    :param dataset:
    :param clip_length:
    :param dataset_path:
    :return:
    """
    print("preprocess for [{}]".format(dataset))
    logger.info("preprocess for [{}]".format(dataset))
    assert img_type in ['gray', 'rgb'], '[!!!] wrong image type'

    frames_type_list = ['training_frames', 'testing_frames']
    size = '{}_{}'.format(height, width)
    # to npy
    try:
        for frames_type in frames_type_list:
            frame_path = os.path.join(dataset_path, dataset, frames_type)
            if not os.path.exists(frame_path):
                os.makedirs(frame_path)
            if img_type == 'rgb':
                frames_type = 'rgb_' + frames_type
            for frames_folder in os.listdir(frame_path):
                npy_file_path = os.path.join(dataset_path, 'npy_dataset', dataset, size,
                                            frames_type,
                                            '{}_{}.npy'.format(frames_type, frames_folder))
                assert(os.path.isfile(npy_file_path))
    except AssertionError:
        for frames_type in frames_type_list:
            to_npy(dataset, dataset_path, height, width, frames_type, img_type)
    except BaseException as e:
        print("unexpected error:{}".format(e))

    # only for training frames, split training volumes into train and validate
    try:
        for stride in [1]:
            if img_type == 'gray':
                tfrecord_dir = os.path.join(dataset_path, 'tfrecord_dataset', dataset, size,
                                        'clip_length_{:02d}'.format(clip_length))
            else:
                tfrecord_dir = os.path.join(dataset_path, 'tfrecord_dataset', dataset, size,
                                            'rgb_clip_length_{:02d}'.format(clip_length))
            path = os.path.join(tfrecord_dir,
                                'num_of_volumes_in_training_frames_stride_{}.txt'.format(stride))
            assert os.path.isfile(path)
    except AssertionError:
        # generate tfrecord for training_frames
        volumes_counter(dataset, dataset_path, size, img_type, 'training_frames', clip_length,
                        stride=1)
        # volumes_counter(dataset, dataset_path, 'training_frames', clip_length, stride=2)
        # volumes_counter(dataset, dataset_path, 'training_frames', clip_length, stride=3)
    except BaseException as e:
        print("[!!!]unexcepted error:{}".format(e))

    validate_split = 0.15
    try:
        for stride in [1]:
            desfile_1 = os.path.join(tfrecord_dir,
                                     'train_stride_{}_validate_split_{}.tfrecords'.format(stride,
                                                                                  validate_split))
            desfile_2 = os.path.join(tfrecord_dir,
                                     'val_stride_{}_validate_split_{}.tfrecords'.format(stride,
                                                                                validate_split))
            assert os.path.isfile(desfile_1) and os.path.isfile(desfile_2)
    except AssertionError:
        if img_type =='gray':
            to_tfrecord_split(dataset, dataset_path, tfrecord_dir, img_type, size,
                              'training_frames',
                              clip_length, stride=1,
                              validate_split=0.15)
        else:
            to_tfrecord_split(dataset, dataset_path, tfrecord_dir, img_type, size,
                              'training_frames',
                              clip_length, stride=1,
                              validate_split=0.15)
        # to_tfrecord_split(dataset, dataset_path, 'training_frames', clip_length, stride=2, validate_split=0.15)
        # to_tfrecord_split(dataset, dataset_path, 'training_frames', clip_length, stride=3, validate_split=0.15)
    except BaseException as e:
        print("[!!!] unexcepted error:{}".format(e))

    # confuse tfrecord

    print('complete [{}]'.format(dataset))
    logger.info('complete [{}]'.format(dataset))


if __name__ == '__main__':
    clip_length = 5
    resize_height = 256
    resize_width = 256
    img_type = 'gray'
    assert img_type in ['gray', 'rgb'], '[!!!] wrong rgb type'
    # dataset_path = 'test'
    # dataset_path = '/home/orli/Blue-HDD/1_final_lab_/Dataset/cgan_data'
    dataset_path = r'D:\study\337_lab\thesis\data'
    # preprocess(logger=logger, dataset='avenue', clip_length=clip_length, dataset_path=dataset_path,
    #            img_type=img_type)
    preprocess(logger=logger, dataset='ped2', clip_length=clip_length, dataset_path=dataset_path,
               img_type=img_type, height=resize_height, width=resize_width)