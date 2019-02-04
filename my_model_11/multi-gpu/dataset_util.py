import tensorflow as tf
import os
import numpy as np


            
class Dataset(object):
    def _parse_example(self, serial_exmp):
        # tf.FixedLenFeature([], tf.string)  # 0D, 标量
        # tf.FixedLenFeature([3]volume...)   1D，长度为3
        features = {'volume': tf.FixedLenFeature([], tf.string)}
        parsed_features = tf.parse_single_example(serial_exmp, features)
        volume = tf.decode_raw(parsed_features['volume'], tf.float32)
        return volume


    def get_train_iterator(self):
        return self.train_iterator

    def get_val_iterator(self):
        return self.val_iterator

    def get_next_volume(self, dataset_type):
        """
        return [batch_size, clip_length, h, w, input_c] format volume
        """
        if dataset_type =='train' and self.train_iterator is not None:
            volume = self.train_iterator.get_next()
        elif dataset_type =='val' and self.val_iterator is not None:
            volume = self.val_iterator.get_next()

        if self.use_rgb == 1:
            volume = tf.reshape(volume, [-1, self.clip_length, self.height, self.width, 3])
        else:
            volume = tf.reshape(volume, [-1, self.clip_length, self.height, self.width])
            volume = tf.expand_dims(volume, axis=-1)

        return volume


    def __init__(self, dataset, cfg, batch_size, use_rgb, tfrecord_root_dir, logger,
                 validation_split=0.15):
        assert validation_split and 0. < validation_split < 1., "validate_split in wrong num"
        self.batch_size = batch_size
        self.clip_length = cfg.clip_length
        self.logger = logger
        self.width = cfg.width
        self.height = cfg.height
        self.use_rgb = use_rgb
        train_file_dir_list = []
        train_volume_num = 0
        val_file_dir_list = []
        val_volume_num = 0

        size = '{}_{}'.format(cfg.width, cfg.height)
        dir_name =  'clip_length_{:02d}'.format(cfg.clip_length)
        if use_rgb == 1:
            dir_name = 'rgb_' + dir_name
        # load training data (stride-1,2,3)
        dir_path = os.path.join(tfrecord_root_dir, dataset, size, dir_name)
        for i in range(1):
            train_file_dir = os.path.join(dir_path,
                                          'train_stride_{}_validate_split_{}.tfrecords'.format(
                                              i+1, validation_split))
            train_file_dir_list.append(train_file_dir)

            train_num_dir = os.path.join(dir_path,
            'num_of_volumes_in_train_stride_{}_validate_split_{}.txt'.format(
                i+1, validation_split))
            train_volume_num += np.loadtxt(train_num_dir, dtype=int)
        
        # load validating data (stride-1)
        for i in range(1):
            val_file_dir = os.path.join(dir_path,
             'val_stride_{}_validate_split_{}.tfrecords'.format(
                i+1, validation_split))
            val_file_dir_list.append(val_file_dir)

            val_num_dir = os.path.join(dir_path,
               'num_of_volumes_in_val_stride_{}_validate_split_{}.txt'.format(
                i + 1, validation_split))
            val_volume_num += np.loadtxt(val_num_dir, dtype=int)

        print('[{}] volumes are loaded for training'.format(train_volume_num))
        self.logger.info('[{}] volumes are loaded for training'.format(train_volume_num))
        print('[{}] volumes are loaded for validating'.format(val_volume_num))
        self.logger.info('[{}] volumes are loaded for validating'.format(val_volume_num))

        try:
            self.train_dataset = tf.data.TFRecordDataset(train_file_dir_list)
            self.train_dataset = self.train_dataset.map(self._parse_example)
            
            if cfg.quick_train == 1:
                """
                for choose hyper-parameter quickly
                """
                percent = 0.2
                small_train_num = int(train_volume_num * percent)
                print('[!!!] {} of train set is used for quick training.'.format(small_train_num,
                                                                                percent))
                self.logger.info('[!!!] {} of train set is used for quick training.'.format(
                    small_train_num, percent))
                self.train_dataset = self.train_dataset.take(small_train_num)

            self.train_dataset = self.train_dataset.shuffle(buffer_size=self.batch_size)
            # drop reminder data with insufficient batch size
            self.train_dataset = self.train_dataset.batch(self.batch_size, drop_remainder=True)

            self.val_dataset = tf.data.TFRecordDataset(val_file_dir_list)
            self.val_dataset = self.val_dataset.map(self._parse_example)
            self.val_dataset = self.val_dataset.batch(self.batch_size,      drop_remainder=True)
            # Repeats this dataset count times.

            # shuffle then repeat,
            # as this will ensure that you see the whole dataset each epoch.
            # if type(cfg.epochs) is int:
            #     self.train_dataset = self.train_dataset.repeat(count=cfg.epochs)
            # else:
            #     self.train_dataset = self.train_dataset.repeat()
            self.train_iterator = self.train_dataset.make_initializable_iterator()
            self.val_iterator = self.val_dataset.make_initializable_iterator()
        except AssertionError:
            print('tfrecord load error!')
            self.train_iterator = None
            self.val_iterator = None
        except:
            import sys
            print("Unexpected error:", sys.exc_info())
            self.train_iterator = None
            self.val_iterator = None


    """
    def __init__(self, dataset, cfg, batch_size, frames_type, tfrecord_root_dir, validation_split):
        file_dir = os.path.join(tfrecord_root_dir, dataset, 'clip_length_{:02d}'.format(cfg.clip_length),
                                '{}.tfrecords'.format(frames_type))

        assert validation_split and 0. < validation_split < 1., "validate_split in wrong num"

        volumes_num_dir = os.path.join(tfrecord_root_dir, dataset, 'clip_length_{:02d}'.format(
            cfg.clip_length), 'num_of_volumes_in_tfrecord.txt')
        assert os.path.isfile(volumes_num_dir)
        volumes_num = np.loadtxt(volumes_num_dir, dtype=int)

        self.batch_size = batch_size
        self.clip_length = cfg.clip_length
        self.train_size = int((1 - validation_split) * volumes_num)
        try:
            assert(os.path.isfile(file_dir))
            self.dataset = tf.data.TFRecordDataset(file_dir)
            # 解析文件中所有记录
            self.dataset = self.dataset.map(self._parse_example)



            self.train_dataset = self.dataset.take(self.train_size)
            self.train_dataset = self.train_dataset.shuffle(buffer_size=self.batch_size)
            self.train_dataset = self.train_dataset.batch(self.batch_size)

            self.val_dataset = self.dataset.skip(self.train_size)
            self.val_dataset = self.val_dataset.batch(self.batch_size)
            # Repeats this dataset count times.

            # shuffle then repeat,
            # as this will ensure that you see the whole dataset each epoch.
            # if type(cfg.epochs) is int:
            #     self.train_dataset = self.train_dataset.repeat(count=cfg.epochs)
            # else:
            #     self.train_dataset = self.train_dataset.repeat()
            self.train_iterator = self.train_dataset.make_initializable_iterator()
            self.val_iterator = self.val_dataset.make_initializable_iterator()
        except AssertionError:
            print('tfrecord load error!')
            self.train_iterator = None
            self.val_iterator = None
        except:
            import sys
            print("Unexpected error:", sys.exc_info())
            self.train_iterator = None
            self.val_iterator = None
    """






