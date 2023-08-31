import numpy as np 
import cv2
import sklearn
import pickle
from django.conf import settings
import os
import sys
sys.path.insert(0,'/home/eduardolacava/Área de Trabalho/fr_project/face_recognition/app')
from face_identifier import FaceIdentifier
from model_api.models import OutputTransform

STATIC_DIR = settings.STATIC_DIR


#face detection 

face_detector_model = cv2.dnn.readNetFromCaffe(os.path.join(STATIC_DIR,'models/deploy.prototxt.txt'),
                                         os.path.join(STATIC_DIR,'models/res10_300x300_ssd_iter_140000.caffemodel'))

#feature extraction 

face_feature_model = cv2.dnn.readNetFromTorch(os.path.join(STATIC_DIR,'models/openface.nn4.small2.v1.t7'))

# face recognition model 

face_recognition_model = pickle.load(
    open(os.path.join(STATIC_DIR,'models/machine_learning_face_person_identity.pkl'), mode ='rb'))

#emotion model 

emotion_recognition_model  = pickle.load(
    open(os.path.join(STATIC_DIR,'models/machinelearning_face_emotion.pkl'), mode ='rb'))


def pipeline_model(img):
    
    #pipeline model 

    image = img.copy()
    h,w = img.shape[:2]

    #face detection 

    img_blob = cv2.dnn.blobFromImage(img,1,(300,300),(104,177,123),
                                    swapRB=False,crop= False)
    face_detector_model.setInput(img_blob) 
    detections = face_detector_model.forward()

    results = dict(face_detect_score = [],
                   face_name = [],
                   face_name_score = [],
                   emotion_name = [],
                   emotion_name_score = [],
                   count = [])
    
    Flag = False

    if len(detections) > 0 :
        
        for i, confidence in enumerate(detections[0,0,:,2]):

            if confidence > 0.5 :
                Flag = True
                box  = detections[0,0,i,3:7].copy()
                box *= np.array([w,h,w,h])
                box = box.astype(int)
            
                startx,starty,endx,endy = box

                cv2.rectangle(image,(startx,starty),(endx,endy),(0,255,0))

                #feature extraction 
                
                face_roi = img[starty:endy,startx:endx].copy()
                face_blob= cv2.dnn.blobFromImage(face_roi,1/255,(96,96),(0,0,0),
                                                swapRB=True,crop=True)
                
                face_feature_model.setInput(face_blob)
                vectors = face_feature_model.forward()

                # predict with machine learning 

                #face_name = face_recognition_model.predict(vectors)[0]
                #face_score = face_recognition_model.predict_proba(vectors).max()

                #text_name = '{} : {:.2f} %'.format(face_name,face_score)
            
                #cv2.putText(image,text_name ,(startx,starty-10),
                #             cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
                
                ## emotion 

                emotion_name  = emotion_recognition_model.predict(vectors)[0]
                emotion_score = emotion_recognition_model.predict_proba(vectors).max()

                text_emotion = '{} : {:.2f} %'.format(emotion_name,emotion_score)
            
                cv2.putText(image,text_emotion ,(startx,endy),
                            cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
                
                results['count'].append(i)
                results['face_detect_score'].append(confidence)
                #results['face_name'].append(face_name)
                #results['face_name_score'].append(face_score)
                results['emotion_name'].append(emotion_name)
                results['emotion_name_score'].append(emotion_score)

                
    return Flag,image

def draw_detections(frame, frame_processor, detections, output_transform):
    size = frame.shape[:2]
    frame = output_transform.resize(frame)
    for roi, landmarks, identity in zip(*detections):
        text = frame_processor.face_identifier.get_identity_label(identity.id)
        if identity.id != FaceIdentifier.UNKNOWN_ID:
            text += ' %.2f%%' % (100.0 * (1 - identity.distance))

        xmin = max(int(roi.position[0]), 0)
        ymin = max(int(roi.position[1]), 0)
        xmax = min(int(roi.position[0] + roi.size[0]), size[1])
        ymax = min(int(roi.position[1] + roi.size[1]), size[0])
        xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 220, 0), 2)

        for point in landmarks:
            x = xmin + output_transform.scale(roi.size[0] * point[0])
            y = ymin + output_transform.scale(roi.size[1] * point[1])
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 255), 2)
        textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
        cv2.rectangle(frame, (xmin, ymin), (xmin + textsize[0], ymin - textsize[1]), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    return frame

def center_crop(frame, crop_size):
    fh, fw, _ = frame.shape
    crop_size[0], crop_size[1] = min(fw, crop_size[0]), min(fh, crop_size[1])
    return frame[(fh - crop_size[1]) // 2 : (fh + crop_size[1]) // 2,
                 (fw - crop_size[0]) // 2 : (fw + crop_size[0]) // 2,
                 :]


def pipeline_openvino(img,processor):

    output_transform = OutputTransform(img.shape[:2], None)
    detections = processor.process(img)
    img = draw_detections(img, processor, detections, output_transform)
    
    if len(detections[0]) > 0:
        return True, img, detections
    else:
        return False, img, None

