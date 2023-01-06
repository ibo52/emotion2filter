import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
import os

#a dataset reference to use:https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset

img_rows,img_cols=48,48 #48x48 pixel gray scale images

batch_size=32 #parse data to 32 chunks
path=os.path.join(os.getcwd(),"dataset/emotion-data")

train_path=os.path.join(path,"train")
validation_path=os.path.join(path,"test")

#artificially expand datasize by augmenting method
train_datagen = ImageDataGenerator(
    rescale=1./255, #scale to increase perf
    rotation_range=30,# to be able to detect rotated faces
    shear_range=0.3,            #distort images to be able to  better prediction
    zoom_range=0.3,             #distort images
    width_shift_range=0.4,      #distort images
    height_shift_range=0.4,     #distort images
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
 train_path,
 color_mode='grayscale',    #no need color, so decrease complexity by grayscaling
 target_size=(img_rows,img_cols),   #48x48 input, decrease complexity
 batch_size=batch_size,
 class_mode='categorical',
 shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
 validation_path,
 color_mode='grayscale',
 target_size=(img_rows,img_cols),
 batch_size=batch_size,
 class_mode='categorical',
 shuffle=True)

model = Sequential()
#----------
F_MAP_1=32
F_MAP_2=64
F_MAP_3=128
F_MAP_4=256
KERNEL_SIZE=(3,3)
num_classes=7 #7 emotions:natural,happy, ... as numeric OUTPUT

#activation is elu, which is similar to elu. Look for link below as reference
#https://datascience.stackexchange.com/questions/102483/difference-between-relu-elu-and-leaky-relu-their-pros-and-cons-majorly

#stack-1
model.add(Conv2D(F_MAP_1, KERNEL_SIZE ,padding='same',kernel_initializer='he_normal',
                 input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(F_MAP_1, KERNEL_SIZE,padding='same',kernel_initializer='he_normal',
                 input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))#throw %20 of predictions to overcome possible overfitting

#stack-2
model.add(Conv2D(F_MAP_2, KERNEL_SIZE, padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(F_MAP_2, KERNEL_SIZE,padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#stack-3
model.add(Conv2D(F_MAP_3, KERNEL_SIZE, padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(F_MAP_3, KERNEL_SIZE,padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
#stack-4
model.add(Conv2D(F_MAP_4, KERNEL_SIZE,padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(F_MAP_4, KERNEL_SIZE,padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#FINAL Stack
model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation("elu"))
model.add(Dropout(0.5))
#Block-6
model.add(Dense(64,kernel_initializer='he_normal'))#connect all layers before
model.add(Activation("elu"))
model.add(Dropout(0.5))
#Block-7
model.add(Dense(num_classes,kernel_initializer='he_normal'))
model.add(Activation('softmax'))


model.summary()

from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
#---------------
checkpoint = ModelCheckpoint('model/EmotionDetectionModel.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)
earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=5,
                          verbose=1,
                          restore_best_weights=True
                          )
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=5,
                              verbose=1,
                              min_delta=0.0001)
callbacks = [earlystop,checkpoint,reduce_lr]

model.compile(loss='categorical_crossentropy',
 optimizer = Adam(lr=0.001),
 metrics=['accuracy'])
nb_train_samples = 24176
nb_validation_samples = 3006
epochs=25

#model.fit_generator deprecated olmuş. Model.fit kullanın diyor
history=model.fit(
 train_generator,
 batch_size=batch_size,
 epochs=epochs,
 callbacks=callbacks,
 validation_data=validation_generator)
