from PIL import Image
import cv2
import numpy as np

def read_gif(path="bird.gif"):
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

i=read_gif()


c=0
while True:
    cv2.imshow("dd",i[c%len(i)])
    c+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

    
