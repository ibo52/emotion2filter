import matplotlib.pyplot as plt
import numpy as np
from numpy import expand_dims
import cv2

from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D,BatchNormalization


# Load the np image data using np.load
data = np.load('dataset/training_images.npz', allow_pickle=True)
#şimdilik kullanmıyom zaten train büyük onu böldüm, bilgisayar zorlaniyor


#-----------------img data adjustments---------------------------np.
train_images = data['images']#2811 civari 250*250 rgb resimler
train_pts = data['points']#data points of face.w300-resimlerin tanımlanan noktaları:68 nokta koordinati

print("Image Data Shape: {}, Points Data Shape: {}".format(train_images.shape,train_pts.shape))
#nokta koordinatlarını 0-250 aralığına düzeltiyoruz.if x>250:x=250;if x<0:x=0

train_pts = np.clip(train_pts,0,250)#zaten0-250 aralığında olamlı ama ne olur ne olmaz diye yapıyoruz
train_pts=train_pts/250

train_images=train_images/255
X_train=train_images[:-500]
X_test=train_images[-500:]

Y_train=train_pts[:-500]
Y_test=train_pts[-500:]

Y_train=np.reshape(Y_train,(-1,136))
Y_test=np.reshape(Y_test,(-1,136))

print("""train Shape: {}, train Points Shape: {}
test Points Shape: {},test Points Shape: {}""".format(X_train.shape,
                                                      Y_train.shape,
                                                      X_test.shape,
                                                      Y_test.shape,))
#-----------------img data adjustments---------------------------
#
#
#build CNN model
def m1(input_shape):
    #size of filters that used in layers
    size_f1=34#on stack1
    size_f2=68#on stack2
    size_f3=136#on stack3

    kernel_size=(3,3)#filters kernel size that browse through input image.
    pool_size=(2,2)#pooling will half the output of layer to reduce complexitiy

    model=Sequential()

    if input_shape==None:
        input_shape=X_train.shape[1:]#250*250*3
        
    #stack 1
    #NOT:relu activation complexty'i azaltıyor. gönül isterdi tanh olsun
    model.add(Conv2D(size_f1,kernel_size,activation="relu",input_shape=input_shape))
    model.add(Conv2D(size_f1,kernel_size,activation="relu",))
    model.add(Conv2D(size_f1,kernel_size,activation="relu"))
    model.add(MaxPooling2D(pool_size=pool_size))

    #stack 2
    model.add(Conv2D(size_f2,kernel_size,activation="relu"))
    model.add(Conv2D(size_f2,kernel_size,activation="relu"))
    model.add(Conv2D(size_f2,kernel_size,activation="relu"))
    model.add(MaxPooling2D(pool_size=pool_size))

    #stack3:final fully connected layer to out
    model.add(Flatten())#2d img arrayı tek boyuta koyar?
    model.add(Dense(size_f3))#fully connected layer 68 tane x,y koordinatı vercek
    model.add(Activation("sigmoid"))#tanh. sigmoid aralığını genişletiyor.
    return model


model=m1(X_train.shape[1:])

#-------------derleme kısmı----------
epoch_size=5#re-calculate as epoch size
batch_size=32# her 32 datada 1 model weightleri update edilecek.yani her iterasyonda ele alınan data sayısıdır
#batch size 32 standarttır. çok büyüğü ekran kartını bitirir. neyse ki senin kart yok
from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint

loss_func="mean_squared_error"
model.compile(loss=loss_func,
              optimizer="adam",#checkpointte kayıtlı olanı sıfırlıyormuş
              metrics=["mse"])

print(model.summary())

#modeli derlerken kaydetmek içün
checkpoint=ModelCheckpoint(filepath="model/LandmarkModel.h5",
                           monitor="val_loss",
                           mode="min",
                           save_best_only=True,
                           verbose=1)#verbose=1;daha iyi model.h5 güncellenirse haber ver demek
callbacks = [checkpoint]

#model.fit_generator deprecated olmuş. Model.fit kullanın diyor
#kaldığın epochtan devam etmek için initial_epoch=numara
history=model.fit(X_train,Y_train,
                  validation_data=(X_test,Y_test),
                  epochs=epoch_size,
                  batch_size=batch_size,
                  callbacks=callbacks,
                  verbose=1,
                  )#initial_epoch=1)#final
