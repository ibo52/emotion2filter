from keras.models import load_model
from keras.utils import img_to_array#from keras.preprocessing.image import img_to_array
from keras.preprocessing import image

from utils import *
"""import cv2
import numpy as np
#utilsde import edildi"""
#my emotion detection model
classifier = load_model('model/EmotionDetectionModel.h5')#classifier = load_model('EmotionDetectionModel.h5')
#my facila landmark model
marker_model = load_model('model/LandmarkModel.h5')


class Camera:
    def __init__(self):
        self.cap=cv2.VideoCapture(0)#open camera
        
        self.canvas=np.zeros((100,100,3))#main frame to manipulate and show contents
        
    def detectFaces(self):
        ret,frame=self.cap.read()#get image frame from camera
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#grayed image
        faces=face_classifier.detectMultiScale(gray,1.3,5)#to detect multiple faces
        
        return frame, gray, faces
    
    def run(self):
        
        roi_color=roi_gray=np.zeros((48,48,3))#alloc mem for region of interest
        
        while True:
            frame, grayed, faces=self.detectFaces()

            for (x,y,w,h) in faces:
                self.canvas=np.asarray(np.copy(frame))#canvas to lay image over backround

                #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                
                cr=(20 if 20<=x and 20<=y else 0)#crop range
                #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)#circle the detected face
                roi_gray=grayed[y:y+h,x:x+w]#detected face area of gray image
                roi_color=frame[y-cr:y+h+cr,x-cr:x+w+cr]#y ekseninde çeneyi de alsın diye +30 ekledim
                
                #print("face x,y,w,h:",x,y,w,h,"\ncr:",cr)
                roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)#resize to emotion model input size
                roi_color=cv2.resize(roi_color,(250,250),interpolation=cv2.INTER_AREA)#resize to marker model input size
                
                #if np.sum([roi_gray])!=0:#if face detected
                    
                #facial landmark detection
                keypoints=self.detectLandmarks(roi_color,marker_model)
                
                #emotion detection: preparation
                roi=roi_gray.astype('float')/255.0#normalize data
                roi=img_to_array(roi)
                roi=np.expand_dims(roi,axis=0)
                    
                #emotion detection
                emotion= self.detectEmotion(roi, x, y)
                
                #apply filter according to detected emotion and facial keypoints
                self.applyEmotion(frame, roi_color, keypoints, emotion, x, y, w, h)
                    
                cv2.putText(frame,class_labels[emotion],(40,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
                    
            if len(faces)<1:
                cv2.putText(frame,'No Face Found',(40,20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        
            self.showCanvas(frame, roi_color)
                
                
    def detectLandmarks(self, roi, markerModel):
        keypoints= detect_landmarks(roi, markerModel)
        put_landmarks_to_image(roi, keypoints)
        
        return keypoints
    
    def detectEmotion(self, roi, x,y):
           preds=classifier.predict(roi,verbose=0)[0]#emotion detection model

           #get highest possible emotion
           predicted=preds.argmax()
           
           return predicted
           
    def applyEmotion(self, frame, roi, facialKeypoints, predictedEmotion, x, y, w, h):
        #merge emoji images with main frame
        #resize emoji to detected face area width and height to maintain change by face distance
        #canvas=add_transparent_image(canvas,cv2.resize(label_emojis[predicted],(w,h),interpolation=cv2.INTER_AREA),x,y-h)#insert emoji to
        frame=add_transparent_image(frame,cv2.resize(label_emojis[predictedEmotion],(w,h),interpolation=cv2.INTER_AREA),x,y-h)#insert emoji to

        #add appropriate filter by prediction
        #insert_filt(predicted,keypoints,canvas,x,y,w,h,roi_color)
        insert_filt(predictedEmotion,facialKeypoints,frame,x,y,w,h,roi)
        
    def showCanvas(self, frame, roi_color):
        
        roi_color=cv2.resize(roi_color,(320,320),interpolation=cv2.INTER_AREA)#resize to marker model input size
        cv2.imshow('landmarks-overlook',roi_color)#cv2.resize(frame,(0,0),fx=2,fy=2))
        cv2.imshow('Face detection and Emotion Analysis',frame)#cv2.resize(frame,(0,0),fx=2,fy=2))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            exit(0)


if __name__=="__main__":
    cam=Camera()
    cam.run()
    