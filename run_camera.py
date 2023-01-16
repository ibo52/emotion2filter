from keras.models import load_model
from tensorflow.keras.utils import img_to_array#from keras.preprocessing.image import img_to_array
from keras.preprocessing import image

from utils import *
"""import cv2
import numpy as np
#utilsde import edildi"""
#my emotion detection model
classifier = load_model('model/EmotionDetectionModel.h5')#classifier = load_model('EmotionDetectionModel.h5')
#my facila landmark model
marker_model = load_model('model/LandmarkModel.h5')

"""
#https://stackoverflow.com/questions/6893968/how-to-get-the-return-value-from-a-thread
from threading import thread
from Queue import Queue
q=Queue()
mark_thread=Thread(target=detect_landmarks,args=(roi_color,marker_model,))
emotion_thread=Thread(target=classifier.predict,args=(roi,verbose=0,))
"""
#
#
#----------------------------------
#open camera
cap=cv2.VideoCapture(0)
roi_color=roi_gray=np.zeros((48,48,3))
canvas=np.zeros((100,100,3))
#----------------------------------
while True:
    ret,frame=cap.read()#get image frame from camera
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#grayed image
    faces=face_classifier.detectMultiScale(gray,1.3,5)#to detect multiple faces
    
    for (x,y,w,h) in faces:
        canvas=np.asarray(np.copy(frame))#canvas to lay image over backround

        #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
        cr=(20 if 20<=x and 20<=y else 0)#crop range
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)#circle the detected face
        roi_gray=gray[y:y+h,x:x+w]#detected face area of gray image
        roi_color=frame[y-cr:y+h+cr,x-cr:x+w+cr]#y ekseninde çeneyi de alsın diye +30 ekledim
        
        #print("face x,y,w,h:",x,y,w,h,"\ncr:",cr)
        roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)#resize to emotion model input size
        roi_color=cv2.resize(roi_color,(250,250),interpolation=cv2.INTER_AREA)#resize to marker model input size
        
        #facial landmark detection
        keypoints=detect_landmarks(roi_color,marker_model)
        put_landmarks_to_image(roi_color,keypoints)
        
        if np.sum([roi_gray])!=0:#if face detected
           roi=roi_gray.astype('float')/255.0#normalize data
           roi=img_to_array(roi)
           roi=np.expand_dims(roi,axis=0)
           preds=classifier.predict(roi,verbose=0)[0]#emotion detection model

           #get highest possible emotion
           predicted=preds.argmax()
           label=class_labels[predicted]
           label_position=(x,y)


           #merge emoji images with main frame
           #resize emoji to detected face area width and height to maintain change by face distance
           #canvas=add_transparent_image(canvas,cv2.resize(label_emojis[predicted],(w,h),interpolation=cv2.INTER_AREA),x,y-h)#insert emoji to
           frame=add_transparent_image(frame,cv2.resize(label_emojis[predicted],(w,h),interpolation=cv2.INTER_AREA),x,y-h)#insert emoji to

           #add appropriate filter by prediction
           #insert_filt(predicted,keypoints,canvas,x,y,w,h,roi_color)
           insert_filt(predicted,keypoints,frame,x,y,w,h,roi_color)
           
           #canvas=add_transparent_image(canvas,cv2.resize(label_filters[predicted],(64,64),interpolation=cv2.INTER_AREA),x+det_x,y+det_y)
           cv2.putText(frame,label,(40,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
           #cv2.putText(canvas,label,(40,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        else:
           cv2.putText(frame,'No Face Found',(40,20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
           
    #cv2.imshow('detected face',cv2.resize(roi_color,(0,0),fx=4,fy=4))
    """
    bone_canvas=np.zeros((250,300,3))
    tempx,tempy=roi_color.shape[0],roi_color.shape[1]
    bone_canvas[0:tempx,0:tempy,:]=roi_color
    #bone_canvas[tempx:tempx+roi_gray.shape[1] , tempy:tempy+roi_gray.shape[0],:]=roi_gray
    """
    
    roi_color=cv2.resize(roi_color,(320,320),interpolation=cv2.INTER_AREA)#resize to marker model input size
    cv2.imshow('landmarks-overlook',roi_color)#cv2.resize(frame,(0,0),fx=2,fy=2))
    cv2.imshow('Face detection and Emotion Analysis',frame)#cv2.resize(frame,(0,0),fx=2,fy=2))
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
cap.release()
cv2.destroyAllWindows()
