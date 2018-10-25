from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
import numpy as np
import cv2
import dlib
import base64
import os
import faceapp.face_rec as fc

# Create your views here.
def index(request):
    return render(request, "index.html")

def face(request):
    return render(request, "face.html", {"feature": False})

def compare(request):
    return render(request, "compare.html", {"score" : None})


def getFeature(request):
    if request.method == 'POST':
        imgBase64 =  request.POST.get('imageBase64')
        imgBase64 = imgBase64[imgBase64.find(',') + 1 : ]
        #print(imgBase64)
        imgData = base64.b64decode(imgBase64.encode('utf-8'))
        with open('./faceapp/static/tmp1.jpg', 'wb') as f:
            f.write(imgData)

    detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor('./faceapp/model/shape_predictor_68_face_landmarks.dat')
    img = cv2.imread('./faceapp/static/tmp1.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = detector(img_gray, 0)
    if len(faces) > 0:
        for k, d in enumerate(faces):
            cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (255, 255, 255))
            shape = landmark_predictor(img, d)
            for i in range(68):
                cv2.circle(img, (shape.part(i).x, shape.part(i).y), 5, (0, 255, 0), -1, 8)
                cv2.putText(img, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 2555, 255))
    cv2.imwrite("./faceapp/static/tmp1.jpg", img)

    return render(request, "feature.html")
    #return render(request, "face.html", {"feature": True})
    #return JsonResponse({'feature': True})


def faceCompare(request):
    if request.method == 'POST':
        img1Base64 = request.POST.get('img1Base64')
        img2Base64 = request.POST.get('img2Base64')
        img1Base64 = img1Base64[img1Base64.find(',') + 1:]
        img2Base64 = img2Base64[img2Base64.find(',') + 1:]
        img1Data = base64.b64decode(img1Base64.encode('utf-8'))
        img2Data = base64.b64decode(img2Base64.encode('utf-8'))
        with open('./faceapp/static/img1.jpg', 'wb') as f:
            f.write(img1Data)
        with open('./faceapp/static/img2.jpg', 'wb') as f:
            f.write(img2Data)

        faceRec = fc.faceRecognition()
        faceRec.inputPerson(name='image1', imgPath='./faceapp/static/img1.jpg')
        vector = faceRec.create128DVectorSpace()
        personData1 = fc.getPersonData(faceRec, vector)

        faceRec = fc.faceRecognition()
        faceRec.inputPerson(name='image2', imgPath='./faceapp/static/img2.jpg')
        vector = faceRec.create128DVectorSpace()
        personData2 = fc.getPersonData(faceRec, vector)
        score = fc.comparePersonData(personData1, personData2)
        result = {}
        result['score'] = score
        if score < 0.6:
            result['result'] = '是同一个人！'
        else:
            result['result'] = '不是同一个人！'
        #return render(request, 'compare.html', result)
        return JsonResponse(result)