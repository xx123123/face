# -*- coding: utf-8 -*-
import sys
import dlib
import cv2
import os
import numpy as np
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def comparePersonData(facedata1, facedata2):
    diff = 0
    for i in range(len(facedata1)):
        diff += (facedata1[i] - facedata2[i]) ** 2
    diff = np.sqrt(diff)
    return diff

def getPersonData(faceRecClass, faceDescriptor):
    if faceRecClass.name == None or faceDescriptor == None:
        return
    vectors = np.array([])
    for i, num in enumerate(faceDescriptor):
        vectors = np.append(vectors, num)
    return vectors

class faceRecognition(object):
    def __init__(self):
        self.predictorPath = './faceapp/model/shape_predictor_68_face_landmarks.dat'
        self.faceRecModelPath = './faceapp/model/dlib_face_recognition_resnet_model_v1.dat'
        self.detector = dlib.get_frontal_face_detector()#加载正脸检测器
        self.shapePredictor = dlib.shape_predictor(self.predictorPath)#加载人脸关键点检测器
        self.faceRecModel = dlib.face_recognition_model_v1(self.faceRecModelPath)#加载人脸识别模型
        self.name = None
        self.imgBgr = None
        self.imgRgb = None

    def inputPerson(self, name = 'people', imgPath = None):
        if imgPath == None:
            print('No file!\n')
            return

        self.imgPath = imgPath
        self.name = name
        self.imgBgr = cv2.imread(self.imgPath)
        b, g, r = cv2.split(self.imgBgr)
        self.imgRgb = cv2.merge([r, g, b])


    def create128DVectorSpace(self):
        dets = self.detector(self.imgRgb, 1)
        print("Number of faces detected: {}".format(len(dets)))
        for index, face in enumerate(dets):
            print('face {}; left {}; top {}; right {}; bottom {}'.format(index, face.left(), face.top(), face.right(),
                                                                         face.bottom()))
            shape = self.shapePredictor(self.imgRgb, face)
            faceDescriptor = self.faceRecModel.compute_face_descriptor(self.imgRgb, shape)
            return faceDescriptor

