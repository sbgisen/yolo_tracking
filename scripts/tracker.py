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

import cv2
import cv_bridge
import message_filters
import numpy as np
import rospy
from jsk_recognition_msgs.msg import ClassificationResult
from jsk_recognition_msgs.msg import RectArray
from sensor_msgs.msg import Image

from boxmot import BoTSORT


class Tracker(object):

    def __init__(self) -> None:
        model_path = pathlib.Path(rospy.get_param('~model', 'osnet_x0_25_msmt17.pt'))
        self.__tracker = BoTSORT(model_weights=model_path, device='cuda', fp16=True, frame_rate=10, track_buffer=1000)
        self.__pub = rospy.Publisher('vis', Image, queue_size=1)
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
        tracks = self.__tracker.update(np.array(detections), image)
        if tracks.shape[0] == 0:
            return
        xyxys = tracks[:, 0:4].astype('int')  # float64 to int

        ids = tracks[:, 4].astype('int')  # float64 to int
        confs = tracks[:, 5]
        clss = tracks[:, 6].astype('int')  # float64 to int
        inds = tracks[:, 7].astype('int')  # float64 to int
        color = (0, 0, 255)  # BGR
        thickness = 2
        fontscale = 0.5
        if tracks.shape[0] != 0:
            for xyxy, id, conf, cls in zip(xyxys, ids, confs, clss):
                image = cv2.rectangle(image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, thickness)
                cv2.putText(image, f'id: {id}, conf: {conf}, c: {cls}', (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, fontscale, color, thickness)

        self.__pub.publish(self.__bridge.cv2_to_imgmsg(image, encoding='bgr8'))


if __name__ == '__main__':
    rospy.init_node('tracker')
    _ = Tracker()
    rospy.spin()
