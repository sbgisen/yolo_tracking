#!/usr/bin/env pipenv-shebang
# -*- coding:utf-8 -*-

# Copyright (c) 2023 SoftBank Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pathlib

import cv_bridge
import message_filters
import numpy as np
import rospy
from box_mot.msg import Features
from jsk_recognition_msgs.msg import ClassificationResult
from jsk_recognition_msgs.msg import RectArray
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

from boxmot.appearance.reid_multibackend import ReIDDetectMultiBackend


class EstimateFeature(object):

    def __init__(self) -> None:
        model_path = pathlib.Path(rospy.get_param('~model', 'osnet_x0_25_msmt17.pt'))
        self.__model = ReIDDetectMultiBackend(weights=model_path, device='cuda', fp16=True)
        self.__pub = rospy.Publisher('features', Features, queue_size=1)
        image_sub = message_filters.Subscriber('image', Image)
        rects_sub = message_filters.Subscriber('rects', RectArray)
        results_sub = message_filters.Subscriber('results', ClassificationResult)
        self.__sync = message_filters.TimeSynchronizer([image_sub, rects_sub, results_sub], 10)
        self.__sync.registerCallback(self.__callback)
        self.__bridge = cv_bridge.CvBridge()

    def __callback(self, img: Image, rects: RectArray, results: ClassificationResult) -> None:
        image = self.__bridge.imgmsg_to_cv2(img, desired_encoding='bgr8')
        detections = []
        for rect, name, prob, label_id in zip(rects.rects, results.label_names, results.label_proba, results.labels):
            if name != 'person':
                continue
            detections.append([rect.x, rect.y, rect.x + rect.width, rect.y + rect.height, prob, label_id])
        if not detections:
            return
        detections = np.array(detections)
        features = self.__model.get_features(detections[:, 0:4], image)
        features_msg = Features()
        features_msg.header = img.header
        for feature in features:
            features_msg.features.append(Float32MultiArray(data=feature))
        self.__pub.publish(features_msg)


if __name__ == '__main__':
    rospy.init_node('estimate_feature')
    _ = EstimateFeature()
    rospy.spin()
