import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
import os

from base_model import ANNClassifierModel

#a dataset reference to use:https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset

#i am inherittin my base model to easily manage and monitor my trained or progressed models
class EmotionModel(ANNClassifierModel):
    
    def __init__(self,name='emotionModel.h5',
                 data_path=os.path.join(os.getcwd(),"dataset/emotion-data"),
                 img_input_shape=(48,48,1),
                 batch_size=32,
                 epoch=100):

        super().__init__(name,data_path,img_input_shape,batch_size, epoch)
        
    #override the method to specify your own model
    def build_ANN(self):
        self.model = Sequential()
        #----------
        INPUT_IM_SHAPE = self.img_rows, self.img_cols, self.channels
        
        F_MAP_1=32
        F_MAP_2=64
        F_MAP_3=128
        F_MAP_4=256
        KERNEL_SIZE=(3,3)
        num_classes=len(self.labels) #7 emotions:natural,happy, ... as numeric OUTPUT

        #activation is elu, which is similar to elu. Look for link below as reference
        #https://datascience.stackexchange.com/questions/102483/difference-between-relu-elu-and-leaky-relu-their-pros-and-cons-majorly

        #stack-1
        self.model.add(Conv2D(F_MAP_1, KERNEL_SIZE ,padding='same',kernel_initializer='he_normal',
                        input_shape=INPUT_IM_SHAPE))
        self.model.add(Activation('elu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(F_MAP_1, KERNEL_SIZE,padding='same',kernel_initializer='he_normal',))
        self.model.add(Activation('elu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.2))#throw %20 of predictions to overcome possible overfitting

        #stack-2
        self.model.add(Conv2D(F_MAP_2, KERNEL_SIZE, padding='same',kernel_initializer='he_normal'))
        self.model.add(Activation('elu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(F_MAP_2, KERNEL_SIZE,padding='same',kernel_initializer='he_normal'))
        self.model.add(Activation('elu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.2))

        #stack-3
        self.model.add(Conv2D(F_MAP_3, KERNEL_SIZE, padding='same',kernel_initializer='he_normal'))
        self.model.add(Activation('elu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(F_MAP_3, KERNEL_SIZE,padding='same',kernel_initializer='he_normal'))
        self.model.add(Activation('elu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.2))
        #stack-4
        self.model.add(Conv2D(F_MAP_4, KERNEL_SIZE,padding='same',kernel_initializer='he_normal'))
        self.model.add(Activation('elu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(F_MAP_4, KERNEL_SIZE,padding='same',kernel_initializer='he_normal'))
        self.model.add(Activation('elu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.2))

        #FINAL Stack
        self.model.add(Flatten())
        self.model.add(Dense(64,kernel_initializer='he_normal'))
        self.model.add(Activation("elu"))
        self.model.add(Dropout(0.5))
        #Block-6
        self.model.add(Dense(64,kernel_initializer='he_normal'))#connect all layers before
        self.model.add(Activation("elu"))
        self.model.add(Dropout(0.5))
        #Block-7
        self.model.add(Dense(num_classes,kernel_initializer='he_normal'))
        self.model.add(Activation('softmax'))

if __name__=="__main__":
    print("""
    You can define new model by inheriting base class,
    you simply have to override 'build_ANN(self):' function
    """)
    
    _Model=EmotionModel(name="EmotionDetectionModel.h5",data_path="dataset/emotion-data")
    _Model.fit_model()
    with open("emotionLabels.py","w") as f:
        f.write(f"emotionLabels_={_Model.labels}")
