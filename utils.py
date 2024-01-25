import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import rotate as nprotate# image array rotate
_ANIM_COUNTER=0
"""
importings, functions of motor program
"""
def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    """lay an image over another"""
    
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    #y_offset-=fg_h#emojiyi başın üstüne koymak içün#oarametrede gönderdim

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite

    return background

def detect_landmarks(in_image,model):
    """
    @param in_image: input image.
    @param model: cnn landmark deteciton model object
    """
    #shape image to 250*250,and normalize; to prepare to model
    image=cv2.resize(in_image,(250,250),interpolation=cv2.INTER_AREA)/255
    

    #resmi 4d yap(resimler listesi yani) model 4d giriş bekliyor
    image=np.reshape(image,(-1,250,250,3))

    # estimate 136 6coordinates and shape to 68*(x,y) coordinates of face
    keypoints=model.predict(image,verbose=0)#verbose=0:ekrana yazı yazma

    #model predictions is normalized [0,250] range to [0,1]
    #return non-normalized landmark cordinates
    return (np.reshape(keypoints,(-1,2))*250)

def put_landmarks_to_image(image,keypoints):
    """
    put facial keypoints to image
    """
    imx,imy=image.shape[0]/250,image.shape[1]/250
    
    for mark in keypoints:
        # getting the x and y coordinates from the landmarks or keypoints  
        m_x, m_y=mark[0],mark[1]
        cv2.circle(image, (int(m_x*imx), int(m_y*imy)), 1, (255,255,255),-1)
    
'''#üsttekiyle aynı fonksiyon
def overlay_transparent(background, overlay, x, y):

    x+=50
    y-=overlay.shape[1]#başın üstüne koymak içün
    
    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[:,:, :overlay.shape[2]]
    mask = overlay[:,:, 3:] / 255.0
    print("overlay:",overlay.shape,"overlay_image:",overlay_image.shape,"background:",background.shape)
    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background
'''
    
def insert_filt(emotion:int,keypoints,canvas,face_x,face_y,width,height,roi):
    """insert prepared filter to canvas
    @param face_x: detected face's x pos on canvas
    @param face_y: detected face's y pos on canvas
    @param width:  detected face's width
    @param height: detected face's height
    @param keypoints: landmarks of face. ranges[0, 250]
    @param canvas: ground that the filter applied
    @param emotion: which filter to be applied on that ground
    """
    global _ANIM_COUNTER

    if emotion==0:#angry
        #extract eye landmarks for filter
        marking=keypoints[ mark_loc["eyes"][0]:mark_loc["eyes"][1] ]
        len_=len(marking)//2

        #center of eyes by averaging eyepoints
        left_ex, left_ey, right_ex, right_ey = int(np.mean(marking[:len_ ,0 ])) , int(np.mean(marking[:len_ ,1 ])) , int(np.mean(marking[len_:,0])) ,int(np.mean(marking[len_:,1]))

        #veins filter scale
        size=[1, 1.1, 1.2, 1.3, 1.4, 1.3, 1.2, 1.1, 1]

        _ex, _ey=keypoints[26,0],keypoints[26,1]-64#sağ kaşın üstüne yerleştir

        fitx,fity=width/250, height/250
        f_w,f_h=size[_ANIM_COUNTER%9] *64 *fitx, size[_ANIM_COUNTER%9]*64 *fity#filter size(angry)
        
        filt_im=label_filters[emotion]
        flame_im=read_gif("emojis/flame.gif")
        
        add_transparent_image(canvas, cv2.resize(filt_im,(int(f_w*250/width),int(f_h*250/height)),interpolation=cv2.INTER_AREA) , int(face_x+_ex), int(face_y+_ey))

        add_transparent_image(roi, cv2.resize(filt_im,(int(f_w*250/width),int(f_h*250/height)),interpolation=cv2.INTER_AREA) ,int(_ex) ,int(_ey))

        cv2.circle(roi,( int(keypoints[26,0]) , int(keypoints[26,1]) ),3,(0,255,0),-1)

        #add flame images to eyes
        flame_w=96#filter width,height
        flame_h=160
        add_transparent_image(canvas, cv2.resize( flame_im[_ANIM_COUNTER%len(flame_im)] ,(int(flame_h*fity),int(flame_w*fitx)) ,interpolation=cv2.INTER_AREA) ,int(face_x+(left_ex- flame_w)*fitx ), int(face_y+(left_ey-flame_h/2)*fity ))
        add_transparent_image(canvas, cv2.resize( flame_im[_ANIM_COUNTER%len(flame_im)] ,(int(flame_h*fity),int(flame_w*fitx)) ,interpolation=cv2.INTER_AREA) ,int(face_x+(right_ex- flame_w)*fitx ), int(face_y+(right_ey-flame_h/2)*fity ))

        _ANIM_COUNTER+=1

    elif emotion==1:#disgust
        marking=keypoints[ mark_loc["mouth"][0]:mark_loc["mouth"][1] ]

        filt_x=np.mean(marking[:,0])
        filt_y=np.mean(marking[:,1])

        #cv2.circle(roi,(int(mouth_x),int(mouth_y)),2,(255,255,0),-1)

        fitx,fity=width/250, height/250
        #filter width:difference of x-axis of jawline points:1-17
        #filter height:difference of y-axis of left eyebrow point(19) and mean(eye) points
        f_w,f_h=abs( keypoints[48,0]-keypoints[54,0] )+30, (abs( keypoints[28,1]- keypoints[8,1] ))#filter size(sunglasses)
        #f_w,f_h=244,106
        filt_x-=f_w/2
        
        filt_im=label_filters[emotion]#vomit gif
        
        #canvas
        add_transparent_image(canvas, cv2.resize( filt_im[_ANIM_COUNTER%len(filt_im)], (int(f_w*fitx),int(f_h*fity)), interpolation=cv2.INTER_AREA) ,int(face_x+filt_x*fitx),int(face_y+filt_y*fity) )
        _ANIM_COUNTER+=1
        #
        #
        
    elif emotion==3:#happy
        #extract eye landmarks for filter
        marking=keypoints[ mark_loc["eyes"][0]:mark_loc["eyes"][1] ]
        len_=len(marking)//2

        #center of eyes by averaging eyepoints
        left_ex, left_ey, right_ex, right_ey = int(np.mean(marking[:len_ ,0 ])) , int(np.mean(marking[:len_ ,1 ])) , int(np.mean(marking[len_:,0])) ,int(np.mean(marking[len_:,1]))

        slope=(right_ey - left_ey ) / (right_ex - left_ex)
        angle=np.rad2deg(np.arctan( slope ))

        #put_landmarks_to_image(canvas,marking[:])
        #cv2.circle(canvas,(left_ex,left_ey),1,(0,255,0),-1)
        #cv2.circle(canvas,(right_ex,right_ey),1,(0,255,0),-1)

        #filter width:difference of x-axis of jawline points:1-17
        #filter height:difference of y-axis of left eyebrow point(19) and mean(eye) points
        fitx, fity=width/250, height/250
        f_w,f_h=abs( keypoints[0,0]-keypoints[16,0] ), (abs( keypoints[19,1]- keypoints[1,1] ))#filter size(sunglasses)

        filt_im=nprotate(label_filters[emotion],int(-angle))

        #filter placemet position is first eyebrow point(18)
        filt_x,filt_y= keypoints[17,0], keypoints[17,1]
        #roi
        add_transparent_image(roi, cv2.resize(filt_im,(int(f_w),int(f_h)),interpolation=cv2.INTER_AREA) ,int(filt_x),int(filt_y) )

        #canvas
        add_transparent_image(canvas, cv2.resize(filt_im,(int(f_w*fitx),int(f_h*fity)),interpolation=cv2.INTER_AREA) ,int(face_x+filt_x*fitx),int(face_y+filt_y*fity) )
        #
        #
    elif emotion==4:#neutral
        bird_im=read_gif("emojis/bird.gif")
        #extract eye landmarks to determine slope angle
        marking=keypoints[ mark_loc["eyes"][0]:mark_loc["eyes"][1] ]
        len_=len(marking)//2

        #center of eyes by averaging eyepoints
        left_ex, left_ey, right_ex, right_ey = int(np.mean(marking[:len_ ,0 ])) , int(np.mean(marking[:len_ ,1 ])) , int(np.mean(marking[len_:,0])) ,int(np.mean(marking[len_:,1]))

        slope=(right_ey - left_ey ) / (right_ex - left_ex)
        angle=np.rad2deg(np.arctan( slope ))

        #put_landmarks_to_image(canvas,marking[:])
        #cv2.circle(canvas,(left_ex,left_ey),1,(0,255,0),-1)
        #cv2.circle(canvas,(right_ex,right_ey),1,(0,255,0),-1)

        fitx,fity=width/250, height/250
        #filter width:difference of x-axis of jawline points:1-17
        #filter height:difference of y-axis of left eyebrow point(19) and mean(eye) points
        f_w,f_h=abs( keypoints[0,0]-keypoints[16,0])+20 , abs( keypoints[19,1]- keypoints[1,1])+10 #filter width and height over image(crown of inner peace)
    
        bird_w,bird_h=160, 128
        
        filt_im=nprotate(label_filters[emotion],int(-angle))

        #top left of jawline is filter place point
        filt_x,filt_y= int(keypoints[0,0])-10 , int(keypoints[0,1]-f_h)-40
        
        bird_pos=[0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 165, 150, 135, 120, 105, 90, 75, 60, 45, 30, 15, 0]
        len_b=len(bird_pos)

        '''
        #roi
        #crown (of inner peace)
        add_transparent_image(roi, cv2.resize(filt_im,(int(f_w),int(f_h)),interpolation=cv2.INTER_AREA) ,filt_x,filt_y)
        #bird image
        add_transparent_image(roi, cv2.resize(bird_im[_ANIM_COUNTER%len(bird_im)],(bird_w,bird_h),interpolation=cv2.INTER_AREA) ,filt_x+ int( len_b* np.sin( np.radians(bird_pos[_ANIM_COUNTER%len_b]) ) ), filt_y+ int( len_b* np.cos(np.radians(bird_pos[_ANIM_COUNTER%len_b]) ) ))
        add_transparent_image(roi, cv2.resize( np.flip(bird_im[_ANIM_COUNTER%len(bird_im)],1) ,(bird_w,bird_h) ,interpolation=cv2.INTER_AREA) ,filt_x+20+ int( len_b* np.sin( np.radians(bird_pos[(_ANIM_COUNTER+10) %len_b]) ) ), filt_y+20+ int( len_b* np.cos(np.radians(bird_pos[(_ANIM_COUNTER+10) %len_b]) ) ))
        '''
        #canvas
        #crown
        add_transparent_image(canvas, cv2.resize(filt_im,(int(f_w*fitx),int(f_h*fity)),interpolation=cv2.INTER_AREA) ,int(face_x+filt_x*fitx),int(face_y+filt_y*fity) )
        #bird image
        add_transparent_image(canvas, cv2.resize(bird_im[_ANIM_COUNTER%len(bird_im)],(int(bird_w*fitx),int(bird_h*fity)) ,interpolation=cv2.INTER_AREA) , int(face_x+filt_x*fitx)-60+ int( len_b* np.sin( np.radians(bird_pos[_ANIM_COUNTER %len_b]) ) ), int(face_y+filt_y*fity)-60+ int( len_b* np.cos(np.radians(bird_pos[_ANIM_COUNTER %len_b]) ) ))
        add_transparent_image(canvas, cv2.resize( np.flip(bird_im[_ANIM_COUNTER%len(bird_im)],1) ,(int(bird_w*fitx),int(bird_h*fity)) ,interpolation=cv2.INTER_AREA) ,int(face_x+filt_x*fitx)+80+ int( len_b* np.sin( np.radians(bird_pos[(_ANIM_COUNTER+10) %len_b]) ) ), int(face_y+filt_y*fity)-80+ int( len_b* np.cos(np.radians(bird_pos[(_ANIM_COUNTER+10) %len_b]) ) ))

        _ANIM_COUNTER+=1
        #add_transparent_image(canvas, cv2.resize(filt_im,(int(f_w),int(f_h)),interpolation=cv2.INTER_AREA) ,face_x+int(keypoints[0,0]),face_y+int(keypoints[0,1]) )
        #
        #
        
    elif emotion==5:#sad
        velocity=[0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66]
        #extract eye landmarks for filter
        marking=keypoints[ mark_loc["eyes"][0]:mark_loc["eyes"][1] ]
        len_=len(marking)//2

        #center of eyes by averaging eyepoints
        left_ex, left_ey, right_ex, right_ey = int(np.mean(marking[:len_ ,0 ])) , int(np.mean(marking[:len_ ,1 ])) , int(np.mean(marking[len_:,0])) ,int(np.mean(marking[len_:,1]))


        fitx,fity= width/250, height/250
        f_w,f_h=abs( keypoints[36,0]-keypoints[39,0] ) *width/250, (abs( keypoints[37,1] - keypoints[41,1] )+50) *height/250#filter size(sunglasses)

        filt_im=label_filters[emotion]

        #filt_x, filt_y==left_ex,left_ey

        add_transparent_image(canvas, cv2.resize(filt_im,(int(f_w),int(f_h)),interpolation=cv2.INTER_AREA) ,int(face_x+left_ex*fitx) , int(face_y+ left_ey*fity)+ velocity[_ANIM_COUNTER%12])

        add_transparent_image(canvas, cv2.resize(filt_im,(int(f_w),int(f_h)),interpolation=cv2.INTER_AREA) ,int(face_x+right_ex*fity) , int(face_y+ right_ey*fity)+ velocity[(_ANIM_COUNTER+3)%12])

        _ANIM_COUNTER+=1
    """
    elif emotion==6:#surprise
        marking=keypoints[ mark_loc["mouth"][0]:mark_loc["mouth"][1] ]

        filt_x=np.mean(marking[:,0])
        filt_y=np.mean(marking[:,1])

        #cv2.circle(roi,(int(mouth_x),int(mouth_y)),2,(255,255,0),-1)

        fitx,fity=width/250, height/250
        #filter width:difference of x-axis of jawline points:1-17
        #filter height:difference of y-axis of left eyebrow point(19) and mean(eye) points
        f_w,f_h=abs( keypoints[0,0]-keypoints[16,0])+20 , abs( keypoints[19,1]- keypoints[1,1])+10 #filter width and height over image(crown of inner peace)
        beard_x,beard_y=keypoints[0,0], keypoints[0,1]
        #
        #
        hair_w, hair_h=100,f_h
        
        
        hair_img=cv2.imread("/home/ibrahim/Downloads/hair.png",cv2.IMREAD_UNCHANGED)
        scale_f=1.5*f_w/hair_img.shape[0]
        
        
        hair_img=cv2.resize(hair_img ,None, fx=scale_f,fy=scale_f)
        print("hair:",hair_img.shape,"fw,fh=",f_w,f_h)
        
        hair_x,hair_y=keypoints[0,0]-10, keypoints[0,1]-hair_img.shape[0]+40
        
        #canvas cv2.resize(hair_img, (int(f_w*fitx),int(f_h*fity)), interpolation=cv2.INTER_AREA )
        add_transparent_image(canvas, hair_img ,int(face_x+hair_x*fitx),int(face_y+hair_y*fity) )
        #
        #
    """


def read_gif(path=""):
    if path=="":
        return -1
    """return frames of gif file
    PIL and numpy libraires required"""
    n_frames=0
    i=[]
    
    with Image.open(path) as file:
    # To iterate through the entire gif
        try:
            while 1:
                file.seek(file.tell()+1)
                n_frames += 1
                # do something to im
                arra=file.convert()
                arra=np.asarray(arra)
                i.append(arra)

        except EOFError:
            pass # end of sequence
    return i
#taken from IBM website and opecv github page
cascades={"face":'haarcascade_frontalface_default.xml',
          "left-eye":"haarcascade_eye.xml",
          "right-eye":"haarcascade_righteye_2splits.xml",
          "mouth":"haarcascade_smile.xml"}

casc_path="model/"

face_classifier=cv2.CascadeClassifier(casc_path + cascades["face"])

mark_loc={"jawline":(0,17),
          "eyebrows":(17,27),
          "nose":(27,36),
          "eyes":(36,48),
          "mouth":(48,68)
          }

#
#
#
#-----------------------------------
class_labels=['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
emoji_path="emojis/"#cv2.IMREAD_UNCHANGED alpha kanalıyla aç(saydamlık içün)
#emojis to certain emotion
label_emojis=[np.asarray(cv2.resize(cv2.imread(emoji_path+"Angry.png",cv2.IMREAD_UNCHANGED),(250,250))),
              np.asarray(cv2.resize(cv2.imread(emoji_path+"Disgust.png",cv2.IMREAD_UNCHANGED),(250,250))),
              np.asarray(cv2.resize(cv2.imread(emoji_path+"Fear.png",cv2.IMREAD_UNCHANGED),(250,250))),
              np.asarray(cv2.resize(cv2.imread(emoji_path+"Happy.png",cv2.IMREAD_UNCHANGED),(250,250))),
              np.asarray(cv2.resize(cv2.imread(emoji_path+"Neutral.png",cv2.IMREAD_UNCHANGED),(250,250))),
              np.asarray(cv2.resize(cv2.imread(emoji_path+"Sad.png",cv2.IMREAD_UNCHANGED),(250,250))),
              np.asarray(cv2.resize(cv2.imread(emoji_path+"Surprise.png",cv2.IMREAD_UNCHANGED),(250,250)))
              ]

label_filters=[np.asarray(cv2.resize(cv2.imread(emoji_path+"Angry-f.png",cv2.IMREAD_UNCHANGED),(250,250))),
              read_gif("emojis/Disgust-f.gif"),#np.asarray(cv2.resize(cv2.imread(emoji_path+"Disgust.png",cv2.IMREAD_UNCHANGED),(250,250))),
              np.asarray(cv2.resize(cv2.imread(emoji_path+"Fear.png",cv2.IMREAD_UNCHANGED),(250,250))),
              np.asarray(cv2.resize(cv2.imread(emoji_path+"Happy-f.png",cv2.IMREAD_UNCHANGED),(250,250))),
              np.asarray(cv2.resize(cv2.imread(emoji_path+"Neutral-f.png",cv2.IMREAD_UNCHANGED),(250,250))),
              np.asarray(cv2.resize(cv2.imread(emoji_path+"Sad-f.png",cv2.IMREAD_UNCHANGED),(250,250))),
              np.asarray(cv2.resize(cv2.imread(emoji_path+"Surprise.png",cv2.IMREAD_UNCHANGED),(250,250)))
              ]

spec_filt=[]
#-----------------------------------
#
#
"""
left_eye_classifier=cv2.CascadeClassifier(casc_path+ cascades["left-eye"])
right_eye_classifier=cv2.CascadeClassifier(casc_path+ cascades["right-eye"])
mouth_classifier=cv2.CascadeClassifier(casc_path+ cascades["mouth"])
"""
if __name__=="__main__":
    print("no main.Exit")
