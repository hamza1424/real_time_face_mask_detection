#!/usr/bin/env python
# coding: utf-8

# In[13]:


#importing Libs

import numpy as np
import tensorflow as tf
import pandas as pd
import keras
import keras.backend as k
from keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D, Flatten, Dropout, Dense
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing import image
import cv2
import datetime
import keras.utils


# In[2]:


# Building Model

model = Sequential()

model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = (150,150,3)))

model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), activation = 'relu', ))

model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), activation = 'relu'))

model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(100, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
         rescale = 1./255,
         shear_range= 0.2,
         zoom_range=0.2,
         horizontal_flip=True)

test_dataGen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'train',
        target_size=(150,150),
        batch_size=16 ,
        class_mode='binary')

test_set = test_dataGen.flow_from_directory(
'test', target_size=(150,150), batch_size=16, class_mode='binary')


# In[9]:


# Training Model 
model_saved = model.fit(
              training_set,
              epochs=10,
              validation_data=test_set)


# In[45]:


# Saving the model
my_model = load_model('my_model.h5')


# In[44]:


# Test for individual images

initial_image = image.load_img('without_mask_2.jpg', target_size= (150,150,3))

test_image = image.img_to_array(initial_image)

test_image = np.expand_dims(test_image, axis=0)

pred = model.predict_classes(test_image)[0][0]
if pred == 0:
    print("Wearing Mask")

else:
    print("No Mask")
initial_image


# In[49]:


# Using Real Time Web Cam

cap = cv2.VideoCapture(0)

face_casCade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while cap.isOpened():
    _, img = cap.read()
    
    faces = face_casCade.detectMultiScale(img, scaleFactor =1.1, minNeighbors = 4)
    
    for (x,y,w,h) in faces:
        face_img = img[y:y+h, x:x+w]
        cv2.imwrite('temp.jpg',face_img)
        test_image = image.load_img('temp.jpg', target_size=(150,150,3))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        pred = my_model.predict_classes(test_image)[0][0]
        
        if pred == 1:
            cv2.rectangle(img, (x,y), (x+w, y+h),(0,0,255),3)
            cv2.putText(img, 'NO Mask', ((x+w)//2, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),3)
            
        else:
            cv2.rectangle(img, (x,y), (x+w, y+h),(0,255,0),3)
            cv2.putText(img, 'Wearing Mask', ((x+w)//2, y+h+20), cv2.FONT_HERSHEY_PLAIN,1, (0,255,0), 3)
        
        datat = str(datetime.datetime.now())
        cv2.putText(img, datat,(400,450), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
        
    cv2.imshow('img',img)
    
    if cv2.waitKey(1) == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
    


# In[ ]:




