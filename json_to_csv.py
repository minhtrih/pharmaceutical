# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

import cv2

import pandas as pd
from pandas import DataFrame
import yaml
import json

import numpy as np
import PIL.Image
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('annotation_file', 'via_region_data.json',
                    'Annotation file for raw dataset')
# flags.DEFINE_string('output', 'output',
#                     'Path to output TFRecord ex) tmp: output file = tmp_train, tmp_val')
FLAGS = flags.FLAGS


def main(_):

    annotation_path = FLAGS.annotation_file
    annotation_fid = open(annotation_path)

#     output_dir = os.path.dirname(FLAGS.output)
#     output_name = os.path.basename(FLAGS.output)
#     output_name, output_ext = os.path.splitext(output_name)
    _filename = []
    _width = []
    _height = []
    _class = []
    _xmin = []
    _ymin = []
    _xmax = []
    _ymax = []

    with open(annotation_path) as annotation_fid:

        annotations = yaml.safe_load(annotation_fid)
        annotation_size = len(annotations)

        annotation_keys = list(annotations.keys())
        for idx, key in enumerate(annotation_keys):
            if len(annotations[key]['regions']) == 1:
                img = cv2.imread('./images/train/' +
                                 (annotations[key]['filename']))
                rows, cols = img.shape[:2]
                _filename.append(annotations[key]['filename'])
                _width.append(int(cols))
                _height.append(int(rows))
                _class.append('blister')
                _xmin.append(int(annotations[key]['regions']
                                 ['0']['shape_attributes']['x']))
                _ymin.append(int(annotations[key]['regions']
                                 ['0']['shape_attributes']['y']))
                _xmax.append(int(annotations[key]['regions']['0']['shape_attributes']['x'] +
                                 annotations[key]['regions']['0']['shape_attributes']['width']))
                _ymax.append(int(annotations[key]['regions']['0']['shape_attributes']['y'] +
                                 annotations[key]['regions']['0']['shape_attributes']['height']))
            else:
                region_list = annotations[key]['regions']
                regions_keys = list(region_list.keys())
                for id_reg, key_reg in enumerate(regions_keys):
                    img = cv2.imread('./images/train/' +
                                     (annotations[key]['filename']))
                    rows, cols = img.shape[:2]
                    _filename.append(annotations[key]['filename'])
                    _width.append(int(cols))
                    _height.append(int(rows))
                    _class.append('blister')
                    _xmin.append(
                        int(region_list[key_reg]['shape_attributes']['x']))
                    _ymin.append(
                        int(region_list[key_reg]['shape_attributes']['y']))
                    _xmax.append(int(region_list[key_reg]['shape_attributes']['x'] +
                                     region_list[key_reg]['shape_attributes']['width']))
                    _ymax.append(int(region_list[key_reg]['shape_attributes']['y'] +
                                     region_list[key_reg]['shape_attributes']['height']))

    data_test = {'filename': _filename,
                 'width': _width,
                 'height': _height,
                 'class': _class,
                 'xmin': _xmin,
                 'ymin': _ymin,
                 'xmax': _xmax,
                 'ymax': _ymax,
                 }

    df = DataFrame(data_test,
                   columns=['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
    export_csv = df.to_csv('train_labels.csv', index=None, header=True)


if __name__ == '__main__':
    tf.app.run()
