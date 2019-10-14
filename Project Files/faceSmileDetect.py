# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 09:03:13 2018

@author: Anubhav
"""

import cv2

face_cascade_loaded = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade_loaded = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade_loaded = cv2.CascadeClassifier('haarcascade_smile.xml')

def detectFaceSmile(grayImage , orignalFrame):
    faces_tuple = face_cascade_loaded.detectMultiScale(grayImage, 1.3, 5)
    for (x, y, w, h) in faces_tuple:
        cv2.rectangle(orignalFrame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        roi_gray = grayImage[y:y+h, x:x+w]
        roi_color = orignalFrame[y:y+h, x:x+w]
        
        eyes_tuple = eye_cascade_loaded.detectMultiScale(roi_gray, 1.1, 22)
        for (ex, ey, ew, eh) in eyes_tuple:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
            
        smiles_tuple = smile_cascade_loaded.detectMultiScale(roi_gray, 1.7, 22)
        for (sx, sy, sw, sh) in smiles_tuple:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
    return orignalFrame
    
video_capture = cv2.VideoCapture(0)

while True:
    _, orignalFrame = video_capture.read()
    inputImage = cv2.cvtColor(orignalFrame, cv2.COLOR_BGR2GRAY)
    outputImage = detectFaceSmile(inputImage, orignalFrame)
    
    cv2.imshow('faceSmileDetect', outputImage)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()