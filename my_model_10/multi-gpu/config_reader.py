import os
import yaml

class Config(object):
    def __init__(self):
        # config file
        with open(os.path.join('config.yml'), 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
        self.batch_size = cfg['batch_size']
        self.clip_length = cfg['clip_length']
        self.height = cfg['height']
        self.width = cfg['width']
        self.epochs = cfg['epochs']
        self.opt = cfg['optimizer']
        self.steps = cfg['steps']
        self.patient = cfg['patient']
        self.flow_height = cfg['flow_height']
        self.flow_width = cfg['flow_width']
        self.quick_train = cfg['quick_train']